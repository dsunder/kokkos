/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 2.0
//              Copyright (2014) Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact  H. Carter Edwards (hcedwar@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

#include <Kokkos_Macros.hpp>

#include <impl/Kokkos_ThreadExecutionResource.hpp>

#if defined( _OPENMP )
#include <omp.h>
#endif

#if defined( KOKKOS_ENABLE_HWLOC )

#include <new>
#include <vector>
#include <algorithm>
#include <iostream>

#include <hwloc.h>

namespace Kokkos { namespace Impl {

struct ThreadExecutionResource::Impl
{
  hwloc_obj_t    m_obj            {nullptr};
  hwloc_cpuset_t m_cpuset         {nullptr};
  Impl *         m_member_of      {nullptr};
  Impl *         m_partitions     {nullptr};
  int            m_concurrency    {0};
  int            m_num_partitions {0};
  int            m_level          {0};
  int            m_member_id      {0};
  int            m_global_id      {0};
  int            m_pad            {0};  // used to pad sizeof
};

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------

namespace {

hwloc_topology_t g_topology   = nullptr;
hwloc_cpuset_t   g_process    = nullptr;

ThreadExecutionResource::Impl *  g_root   = nullptr;
std::vector<ThreadExecutionResource::Impl const *> g_leaves;


//------------------------------------------------------------------------------

bool intersect_process( hwloc_obj_t obj, hwloc_bitmap_t scratch ) noexcept
{
  hwloc_bitmap_and( scratch, g_process, obj->allowed_cpuset );
  return !hwloc_bitmap_iszero( scratch );
}

int get_num_partitions( hwloc_obj_t obj, hwloc_bitmap_t scratch ) noexcept
{
  int result = 0;

  for (int i=0; i < obj->arity; ++i) {
    if ( intersect_process( obj->children[i], scratch ) ) {
      ++result;
    }
  }

  return result;
}

hwloc_bitmap_t get_cpuset( hwloc_obj_t obj ) noexcept
{
  hwloc_bitmap_t result = hwloc_bitmap_alloc();
  hwloc_bitmap_and( result, obj->allowed_cpuset, g_process );

  return result;
}

void initialize_tree( ThreadExecutionResource::Impl * impl
                    , ThreadExecutionResource::Impl * member_of
                    , hwloc_obj_t obj
                    , hwloc_cpuset_t scratch
                    , int level
                    , int member_id
                    , int global_id
                    ) noexcept
{
  impl->m_cpuset         = get_cpuset( obj );
  impl->m_member_of      = member_of;
  impl->m_concurrency    = hwloc_bitmap_weight( impl->m_cpuset );
  impl->m_level          = level;
  impl->m_member_id      = member_id;
  impl->m_global_id      = global_id;

  // flatten out superfluous levels
  while ( obj->arity == 1 ) {
    obj = obj->children[0];
  }

  impl->m_obj            = obj;
  impl->m_num_partitions = get_num_partitions( obj, scratch );

  if ( impl->m_num_partitions > 0 ) {
    impl->m_partitions = new (std::nothrow) ThreadExecutionResource::Impl[impl->m_num_partitions]{};

    for (int i=0, j=0; i < obj->arity; ++i) {
      if ( intersect_process( obj->children[i], scratch) ) {
        initialize_tree( impl->m_partitions + j
                       , impl
                       , obj->children[i]
                       , scratch
                       , level + 1
                       , j
                       , i
                       );
        ++j;
      }
    }
  } else {
    g_leaves.push_back( impl );
  }

  std::sort( impl->m_partitions, impl->m_partitions + impl->m_num_partitions
           , []( ThreadExecutionResource::Impl const & a, ThreadExecutionResource::Impl const & b )
           {
             return a.m_obj->logical_index < b.m_obj->logical_index;
           });

  for (int i=0; i < impl->m_num_partitions; ++i) {
    impl->m_partitions[i].m_member_id  = i;
  }
}

void destroy_tree( ThreadExecutionResource::Impl * impl ) noexcept
{
  hwloc_bitmap_free( impl->m_cpuset );

  if (impl->m_num_partitions > 0) {
    for (int i=0; i < impl->m_num_partitions; ++i) {
      destroy_tree( impl->m_partitions + i );
    }
    delete [] impl->m_partitions;
  }
}

void initialize() noexcept
{
  if ( !g_topology ) {
    hwloc_topology_init( & g_topology );
    hwloc_topology_load( g_topology );
    g_process = hwloc_bitmap_alloc();
    #if !defined( _OPENMP )
      const int err = hwloc_get_cpubind( g_topology, g_process, HWLOC_CPUBIND_PROCESS );
    #else
      int err = 0;
      int num_procs = omp_get_num_procs();
      #pragma omp parallel num_threads(num_procs)
      {

        hwloc_cpuset_t tmp = hwloc_bitmap_alloc();
        const int tmp_err = hwloc_get_cpubind( g_topology, tmp, HWLOC_CPUBIND_PROCESS );

        #pragma omp atomic
        err |= tmp_err;

        #pragma omp critical
        {
          hwloc_bitmap_or( g_process, g_process, tmp );
        }

        hwloc_bitmap_free( tmp );
      }
    #endif


    if (err) {
      // assume entire node if unable to detect process binding
      const int num_pu = hwloc_get_nbobjs_by_type( g_topology, HWLOC_OBJ_PU );
      hwloc_bitmap_set_range( g_process, 0, num_pu -1 );
    }

    int num_leaves = hwloc_bitmap_weight( g_process );

    if ( num_leaves == 0 ) {
      num_leaves = hwloc_get_nbobjs_by_type( g_topology, HWLOC_OBJ_PU );
      hwloc_bitmap_set_range( g_process, 0, num_leaves-1);
    }

    hwloc_cpuset_t scratch = hwloc_bitmap_alloc();

    g_root = new (std::nothrow) ThreadExecutionResource::Impl;

    initialize_tree( g_root
                   , nullptr
                   , hwloc_get_root_obj( g_topology )
                   , scratch
                   , 0
                   , 0
                   , 0
                   );

    std::sort( g_leaves.begin(), g_leaves.end()
             , []( ThreadExecutionResource::Impl const * a, ThreadExecutionResource::Impl const * b ) {
                return a->m_obj->logical_index < b->m_obj->logical_index;
             });
  }
}

void finalize() noexcept
{
  if (g_topology) {
    // rebind master thread to entire process
    hwloc_set_cpubind( g_topology
                     , g_process
                     , HWLOC_CPUBIND_THREAD | HWLOC_CPUBIND_STRICT
                     );

    hwloc_topology_destroy( g_topology );
    hwloc_bitmap_free( g_process );
    g_topology = nullptr;
    g_process = nullptr;

    destroy_tree( g_root );
    delete g_root;
    g_root = nullptr;
    g_leaves.clear();
  }
}

class Sentinal
{
public:
  Sentinal() noexcept
  {
    initialize();

  }

  ~Sentinal() noexcept
  {
    finalize();
  }
};

void sentinal() noexcept
{
  static const Sentinal s;
}

//------------------------------------------------------------------------------
} // namespace

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------

ThreadExecutionResource ThreadExecutionResource::root() noexcept
{
  sentinal();

  return ThreadExecutionResource{g_root};
}

ThreadExecutionResource ThreadExecutionResource::leaf( int i ) noexcept
{
  sentinal();

  return ThreadExecutionResource{ g_leaves[i] };
}

int ThreadExecutionResource::num_leaves() noexcept
{
  sentinal();

  return static_cast<int>(g_leaves.size());
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------


ThreadExecutionResource ThreadExecutionResource::member_of() const noexcept
{
  sentinal();

  return ThreadExecutionResource{ m_pimpl->m_member_of };
}

int ThreadExecutionResource::global_id() const noexcept
{
  sentinal();

  return m_pimpl->m_global_id;
}

int ThreadExecutionResource::concurrency() const noexcept
{
  sentinal();

  return m_pimpl->m_concurrency;
}

int ThreadExecutionResource::num_partitions() const noexcept
{
  sentinal();

  return m_pimpl->m_num_partitions;
}

ThreadExecutionResource ThreadExecutionResource::partition( int i ) const noexcept
{
  sentinal();

  return ThreadExecutionResource{ m_pimpl->m_partitions + i };
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------


namespace {

const ThreadExecutionResource::Impl * get_impl( ThreadExecutionResource::Impl const * impl
                                        , hwloc_cpuset_t cpuset
                                        ) noexcept
{
  if( hwloc_bitmap_isequal(impl->m_cpuset, cpuset) ) {
    // traverse to the lowest level
    while (impl->m_num_partitions == 1) {
      impl = impl->m_partitions;
    }
    return impl;
  }
  else if ( impl->m_num_partitions > 0 ) {
    for (int i=0; i < impl->m_num_partitions; ++i) {
      if ( hwloc_bitmap_isincluded( cpuset, impl->m_partitions[i].m_cpuset) ) {
        return get_impl( impl->m_partitions + i, cpuset );
      }
    }
  }

  return nullptr;
}

} // namespace

bool this_thread_set_binding( const ThreadExecutionResource res ) noexcept
{

  sentinal();

  #if !defined( _OPENMP )
    const int err = hwloc_set_cpubind( g_topology
                                     , res.get_impl()->m_cpuset
                                     , HWLOC_CPUBIND_THREAD | HWLOC_CPUBIND_STRICT
                                     );
    return !err;
  #else
    return false;
  #endif
}

ThreadExecutionResource this_thread_get_bind() noexcept
{
  sentinal();

  hwloc_cpuset_t scratch = hwloc_bitmap_alloc();
  const int err = hwloc_get_cpubind( g_topology
                                   , scratch
                                   , HWLOC_CPUBIND_THREAD
                                   );

  const ThreadExecutionResource::Impl * result = nullptr;

  if (!err) {
    result = get_impl( g_root, scratch );
  }

  if (!result ) { result = g_root; }

  hwloc_bitmap_free( scratch );

  return ThreadExecutionResource{ result };
}

ThreadExecutionResource this_thread_get_resource() noexcept
{
  sentinal();

  hwloc_cpuset_t scratch = hwloc_bitmap_alloc();

  const int err =hwloc_get_last_cpu_location( g_topology
                                            , scratch
                                            , HWLOC_CPUBIND_THREAD
                                            );

  const ThreadExecutionResource::Impl * result = nullptr;

  if (!err) {
    result = get_impl( g_root, scratch );
  }

  if (!result ) { result = g_root; }

  hwloc_bitmap_free( scratch );

  return ThreadExecutionResource{ result };
}

namespace {

std::ostream & operator<<( std::ostream & out, const ThreadExecutionResource::Impl * res )
{
  if (res) {
    if(res->m_level == 0) {
      out << "root";
    } else {
      out << res->m_member_of << "." << res->m_global_id;
    }
  }
  else {
    out << "null";
  }

  return out;
}


} // namespace

std::ostream & operator<<( std::ostream & out, const ThreadExecutionResource res )
{
  sentinal();

  if (res) {
    out << "{ " << res.get_impl() << " : " << res.concurrency() << " }";
  }
  else {
    out << "{ null : 0 }";
  }
  return out;
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------

}} // namespace Kokkos::Impl

#elif defined( _GNU_SOURCE ) && !defined( __APPLE__ )

#include <iostream>
#include <vector>
#include <algorithm>

#include <sched.h>
#include <unistd.h>
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
namespace Kokkos { namespace Impl {

struct ThreadExecutionResource::Impl
{
  int    m_level          {0};
  int    m_global_id      {0};
};

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------

namespace {

cpu_set_t g_process     {};
int       g_size        {0};
int       g_concurrency {0};

std::vector< ThreadExecutionResource::Impl > g_leaves;

ThreadExecutionResource::Impl g_root {};

void initialize() noexcept
{
  if (g_size == 0) {
    CPU_ZERO( &g_process );

    g_size = sysconf( _SC_NPROCESSORS_ONLN );

    if (g_size <= 0) {
      g_size = 0;
    }
    else {
      #if defined( _OPENMP )
        int num_procs = omp_get_num_procs();
        #pragma omp parallel num_threads(num_procs)
        {
          cpu_set_t tmp;
          sched_getaffinity( 0
                           , sizeof(cpu_set_t)
                           , &tmp
                           );
          #pragma omp critical
          {
            CPU_OR( &g_process, &g_process, &tmp );
          }
        }
      #else
        sched_getaffinity( 0
                         , sizeof(cpu_set_t)
                         , &g_process
                         );
      #endif
    }

    g_concurrency = CPU_COUNT( &g_process );

    g_leaves.resize( g_concurrency );

    for (int i=0; i<g_size; ++i) {
      if( CPU_ISSET( i , &g_process ) ) {
        g_leaves[i].m_level     = 1;
        g_leaves[i].m_global_id = i;
      }
    }
  }
}

void finalize() noexcept
{
  if (g_size > 0) {
    g_size = 0;
    g_concurrency = 0;
    g_leaves.clear();
  }
}

class Sentinal
{
public:
  Sentinal() noexcept
  {
    initialize();

  }

  ~Sentinal() noexcept
  {
    finalize();
  }
};

void sentinal() noexcept
{
  static const Sentinal s;
}

} // namespace

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------

ThreadExecutionResource ThreadExecutionResource::root() noexcept
{
  sentinal();

  return ThreadExecutionResource{&g_root};
}

ThreadExecutionResource ThreadExecutionResource::leaf( int i ) noexcept
{
  sentinal();

  return ThreadExecutionResource{ &g_leaves[i] };
}

int ThreadExecutionResource::num_leaves() noexcept
{
  sentinal();

  return g_concurrency;
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------


ThreadExecutionResource ThreadExecutionResource::member_of() const noexcept
{
  sentinal();

  return m_pimpl->m_level == 0
       ? ThreadExecutionResource{ nullptr }
       : ThreadExecutionResource{ &g_root }
       ;
}

int ThreadExecutionResource::global_id() const noexcept
{
  return m_pimpl->m_global_id;
}

int ThreadExecutionResource::concurrency() const noexcept
{
  sentinal();

  return m_pimpl->m_level == 0
       ? g_concurrency
       : 1
       ;
}

int ThreadExecutionResource::num_partitions() const noexcept
{
  return m_pimpl->m_level == 0
       ? g_concurrency
       : 0
       ;
}

ThreadExecutionResource ThreadExecutionResource::partition( int i ) const noexcept
{
  sentinal();

  return m_pimpl->m_level == 0
       ? ThreadExecutionResource{ &g_leaves[i] }
       : ThreadExecutionResource{ nullptr }
       ;
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------


bool this_thread_set_binding( const ThreadExecutionResource res ) noexcept
{
  sentinal();

  #if !defined( _OPENMP )
    int err = 0;
    if (res.get_impl()->m_level == 0) {
      err = sched_setaffinity( 0
                             , sizeof(cpu_set_t)
                             , &g_process
                             );
    } else {
      cpu_set_t tmp;
      CPU_ZERO( &tmp );
      CPU_SET( res.get_impl()->m_global_id, &tmp );
      err = sched_setaffinity( 0
                             , sizeof(cpu_set_t)
                             , &tmp
                             );
    }
    return !err;
  #else
    return false;
  #endif
}

ThreadExecutionResource this_thread_get_bind() noexcept
{
  sentinal();

  cpu_set_t tmp;
  CPU_ZERO( &tmp );
  const int err = sched_getaffinity( 0
                                   , sizeof( cpu_set_t )
                                   , & tmp
                                   );

  const int count = CPU_COUNT( &tmp );
  if ( err || count > 1 || count <= 0 ) {
    return ThreadExecutionResource{ &g_root };
  }

  int gid = 0;
  for (; gid<g_size; ++gid) {
    if ( CPU_ISSET( gid, &tmp ) ) break;
  }

  int i = 0;
  for (; i<g_concurrency; ++i) {
    if ( gid == g_leaves[i].m_global_id ) break;
  }

  return ThreadExecutionResource{ &g_leaves[i] };
}

ThreadExecutionResource this_thread_get_resource() noexcept
{
  sentinal();

  const int gid = sched_getcpu();

  if (gid < 0 || gid > g_size ) {
    return ThreadExecutionResource{ &g_root };
  }

  int i = 0;
  for (; i<g_concurrency; ++i) {
    if ( gid == g_leaves[i].m_global_id ) break;
  }

  return ThreadExecutionResource{ &g_leaves[i] };
}

std::ostream & operator<<( std::ostream & out, const ThreadExecutionResource res )
{
  sentinal();

  if ( res ) {
    if (res.get_impl()->m_level == 0) {
      out << "{ root : " << g_concurrency << " }";
    }
    else {
      out << "{ root." << res.get_impl()->m_global_id << " : 1 }";
    }
  }
  else {
    out << "{ null : 0 }";
  }
  return out;
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------

}} // namespace Kokkos::Impl


#else

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------

#include <iostream>

#if defined( _GNU_SOURCE )
#include <unistd.h>
#endif

namespace Kokkos { namespace Impl {

struct ThreadExecutionResource::Impl
{
  int m_concurrency{0};
};

namespace {
ThreadExecutionResource::Impl g_root;

void initialize() noexcept
{
  if ( g_root.m_concurrency == 0 ) {
  #if defined( _OPENMP )
    int num_procs = omp_get_num_procs();
    #pragma omp parallel num_threads(num_procs)
    {
      #pragma omp atomic
      ++g_root.m_concurrency;
    }
  #elif defined( _GNU_SOURCE )
    g_root.m_concurrency = sysconf( _SC_NPROCESSORS_ONLN );
  #else
    g_root.m_concurrency = 1;
  #endif
  }
}

void finalize() noexcept
{
  g_root.m_concurrency = 0;
}

class Sentinal
{
public:
  Sentinal() noexcept
  {
    initialize();

  }

  ~Sentinal() noexcept
  {
    finalize();
  }
};

void sentinal() noexcept
{
  static const Sentinal s;
}

} // namespace

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------

ThreadExecutionResource ThreadExecutionResource::root() noexcept
{
  sentinal();
  return ThreadExecutionResource{&g_root};
}

ThreadExecutionResource ThreadExecutionResource::leaf( int ) noexcept
{
  sentinal();
  return ThreadExecutionResource{ &g_root };
}

int ThreadExecutionResource::num_leaves() noexcept
{
  sentinal();
  return 1;
}

//------------------------------------------------------------------------------

ThreadExecutionResource ThreadExecutionResource::member_of() const noexcept
{
  sentinal();
  return ThreadExecutionResource{ nullptr };
}

int ThreadExecutionResource::global_id() const noexcept
{
  sentinal();
  return 0;
}

int ThreadExecutionResource::concurrency() const noexcept
{
  sentinal();
  return g_root.m_concurrency;
}

int ThreadExecutionResource::num_partitions() const noexcept
{
  sentinal();
  return 0;
}

ThreadExecutionResource ThreadExecutionResource::partition( int i ) const noexcept
{
  sentinal();
  return ThreadExecutionResource{ nullptr };
}

//------------------------------------------------------------------------------

bool this_thread_set_binding( const ThreadExecutionResource res ) noexcept
{ return false; }

ThreadExecutionResource this_thread_get_bind() noexcept
{
  sentinal();
  return ThreadExecutionResource{&g_root};
}

ThreadExecutionResource this_thread_get_resource() noexcept
{
  sentinal();
  return ThreadExecutionResource{&g_root};
}

std::ostream & operator<<( std::ostream & out, const ThreadExecutionResource res )
{
  sentinal();
  out << "{ root : " << g_root.m_concurrency << " }";
  return out;
}

}} // namespace Kokkos::Impl

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------

#endif

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------

// Deprecated APIs

// These functions assume symmetry and will return incorrect results for
// non-symmetric process bindings
namespace Kokkos { namespace hwloc {

#if defined( KOKKOS_ENABLE_HWLOC )

unsigned get_available_numa_count() noexcept
{
  Kokkos::Impl::sentinal();
  const int num_numa =
    hwloc_get_nbobjs_inside_cpuset_by_type( Kokkos::Impl::g_topology
                                          , Kokkos::Impl::g_process
                                          , HWLOC_OBJ_NUMANODE
                                          );

  return num_numa;
}

unsigned get_available_cores_per_numa() noexcept
{
  Kokkos::Impl::sentinal();
  const int num_cores =
    hwloc_get_nbobjs_inside_cpuset_by_type( Kokkos::Impl::g_topology
                                          , Kokkos::Impl::g_process
                                          , HWLOC_OBJ_CORE
                                          );

  return num_cores / get_available_numa_count();
}

unsigned get_available_threads_per_core() noexcept
{
  Kokkos::Impl::sentinal();
  const int num_cores =
    hwloc_get_nbobjs_inside_cpuset_by_type( Kokkos::Impl::g_topology
                                          , Kokkos::Impl::g_process
                                          , HWLOC_OBJ_CORE
                                          );

  const int num_pus =
    hwloc_get_nbobjs_inside_cpuset_by_type( Kokkos::Impl::g_topology
                                          , Kokkos::Impl::g_process
                                          , HWLOC_OBJ_PU
                                          );

  return num_pus / num_cores;
}

#else

unsigned get_available_numa_count() noexcept
{
  Kokkos::Impl::sentinal();
  return 1;
}

unsigned get_available_cores_per_numa() noexcept
{
  Kokkos::Impl::sentinal();
  return Kokkos::Impl::ThreadExecutionResource::root().concurrency();
}

unsigned get_available_threads_per_core() noexcept
{
  Kokkos::Impl::sentinal();
  return 1;
}

#endif

}} // namespace Kokkos::hwloc

