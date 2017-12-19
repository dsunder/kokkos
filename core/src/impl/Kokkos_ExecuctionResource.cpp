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

#include <Kokkos_ExecutionResource.hpp>

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <new>

#if defined(_OPENMP)
#include <omp.h>
#endif


//------------------------------------------------------------------------------
// hwloc implementation
//------------------------------------------------------------------------------
#if defined(KOKKOS_ENABLE_HWLOC)

#include <hwloc.h>

namespace Kokkos {

class ExecutionResource::Pimpl
{
public:

  static void initialize() noexcept
  {
    if ( is_initialized() ) return;

    hwloc_topology_init( & s_topology );
    hwloc_topology_load( s_topology );

    s_process = hwloc_bitmap_alloc();
    s_scratch = hwloc_bitmap_alloc();

    int err = 0;

    // get the process binding
    #if !defined( _OPENMP )
      err = hwloc_get_cpubind( s_topology, s_process, HWLOC_CPUBIND_PROCESS );
    #else
      if ( !omp_in_parallel() ) {

        int num_procs = omp_get_num_procs();
        #if KOKKOS_OPENMP_VERSION >= 40
        #pragma omp parallel num_threads(num_procs) proc_bind(spread)
        #else
        #pragma omp parallel num_threads(num_procs)
        #endif
        {
          hwloc_cpuset_t tmp = hwloc_bitmap_alloc();
          const int tmp_err = hwloc_get_cpubind( s_topology, tmp, HWLOC_CPUBIND_PROCESS );

          #pragma omp atomic update
          err |= tmp_err;

          #pragma omp critical
          {
            hwloc_bitmap_or( s_process, s_process, tmp );
          }

          hwloc_bitmap_free( tmp );
        }
      } else {
        std::cerr << "Error: Cannot call initialize_execution_resources() in parallel" << std::endl;
        std::abort();
      }
    #endif

    if (err) {
      // assume entire node if unable to detect process binding
      const int num_pu = hwloc_get_nbobjs_by_type( s_topology, HWLOC_OBJ_PU );
      hwloc_bitmap_set_range( s_process, 0, num_pu -1 );
    }

    s_root = new (std::nothrow) Pimpl{};

    initialize( s_root                           // Pimpl
              , nullptr                          // parent
              , hwloc_get_root_obj( s_topology ) // hwloc_obj_t
              , 0                                // depth
              , 0                                // index in parent's array
              );
  }

  static void finalize() noexcept
  {
    if ( !is_initialized() ) return;

    finalize(s_root);
    delete s_root;
    s_root = nullptr;

    hwloc_topology_destroy( s_topology );
    s_topology = nullptr;

    hwloc_bitmap_free( s_process );
    s_process = nullptr;

    hwloc_bitmap_free( s_scratch );
    s_scratch = nullptr;
  }

  static bool is_initialized() noexcept { return s_topology != nullptr; }

  static hwloc_topology_t topology() noexcept { return s_topology; }

  static ExecutionResource process() noexcept { return ExecutionResource{ s_root }; }

  int  id()           const noexcept { return m_obj->logical_index; }
  int  depth()        const noexcept { return m_depth;              }
  int  index()        const noexcept { return m_index;              }
  int  num_children() const noexcept { return m_num_children;       }
  int  num_leaves()   const noexcept { return m_num_leaves;         }
  int  concurrency()  const noexcept { return m_concurrency;        }
  bool is_symmetric() const noexcept { return m_symmetric == 1;     }

  Pimpl const * parent()     const noexcept { return m_parent;       }
  Pimpl const * child(int i) const noexcept { return m_children + i; }
  Pimpl const * leaf(int i)  const noexcept { return m_leaves[i];    }

  hwloc_const_cpuset_t get_cpuset() const noexcept { return m_cpuset; }
  hwloc_obj_t          get_obj()    const noexcept { return m_obj;    }

  Pimpl()                           noexcept = default;
  Pimpl( Pimpl const &)             noexcept = default;
  Pimpl( Pimpl &&)                  noexcept = default;
  Pimpl & operator=( Pimpl const &) noexcept = default;
  Pimpl & operator=( Pimpl &&)      noexcept = default;

private:
  static bool intersects_process( hwloc_obj_t obj ) noexcept
  {
    hwloc_bitmap_and( s_scratch, s_process, obj->allowed_cpuset );
    return !hwloc_bitmap_iszero( s_scratch );
  }

  static int num_children( hwloc_obj_t obj ) noexcept
  {
    int result = 0;
    for (int i=0; i < obj->arity; ++i) {
      if ( intersects_process( obj->children[i] ) ) {
        ++result;
      }
    }
    return result;
  }

  static hwloc_cpuset_t get_cpuset( hwloc_obj_t obj ) noexcept
  {
    hwloc_cpuset_t result = hwloc_bitmap_alloc();
    hwloc_bitmap_and( result, obj->allowed_cpuset, s_process );
    return result;
  }


  // return max depth
  static int initialize( Pimpl *     pimpl
                       , Pimpl *     arg_parent
                       , hwloc_obj_t arg_obj
                       , const int   arg_depth
                       , const int   arg_index
                       ) noexcept
  {
    // flatten out superfluous levels
    while ( arg_obj->arity == 1 ) {
      arg_obj = arg_obj->children[0];
    }

    pimpl->m_depth       = arg_depth;
    pimpl->m_index       = arg_index;
    pimpl->m_parent      = arg_parent;
    pimpl->m_obj         = arg_obj;
    pimpl->m_cpuset      = get_cpuset(arg_obj);
    pimpl->m_concurrency = hwloc_bitmap_weight( pimpl->m_cpuset );

    pimpl->m_num_children = num_children( arg_obj );

    // interior node
    int tmp_child_max_depth   = 0;
    if ( pimpl->m_num_children > 0 ) {
      pimpl->m_children = new (std::nothrow) Pimpl[pimpl->m_num_children]{};

      const int arity = arg_obj->arity;
      int tmp_child_concurrency  = 0;
      int tmp_child_num_children = 0;
      bool tmp_is_symmetric = true;
      for (int i=0, j=0; i < arity; ++i) {
        if (intersects_process( arg_obj->children[i]) ) {
          const int child_max_depth = initialize( pimpl->m_children + j  // pimpl
                                                , pimpl                  // parent
                                                , arg_obj->children[i]  // hwloc_obj_t
                                                , arg_depth + 1         // depth
                                                , j                     // index
                                                );
          // determine if symmetric
          if ( j != 0 && tmp_is_symmetric ) {
            tmp_is_symmetric =  pimpl->child(j)->is_symmetric()
                             && child_max_depth == tmp_child_max_depth
                             && pimpl->child(j)->num_children() == tmp_child_num_children
                             && tmp_child_concurrency == pimpl->child(j)->concurrency()
                             ;
          }
          else if ( j == 0 ) {
            tmp_child_max_depth = child_max_depth;
            tmp_child_concurrency = pimpl->child(0)->concurrency();
            tmp_child_num_children = pimpl->child(0)->num_children();
          }

          if ( !tmp_is_symmetric) {
            tmp_child_max_depth = child_max_depth < tmp_child_max_depth
                                ? tmp_child_max_depth
                                : child_max_depth
                                ;
          }

          // advance child index
          ++j;
        }
      }
      pimpl->m_symmetric = tmp_is_symmetric;

      std::sort( pimpl->m_children, pimpl->m_children + pimpl->m_num_children
               , []( const Pimpl & a, const Pimpl & b ) noexcept {
                   return a.id() < b.id();
                 }
               );

      // count the number of leaves

      pimpl->m_num_leaves = 0;
      for (int i=0; i<pimpl->m_num_children; ++i) {
        pimpl->m_num_leaves += pimpl->child(i)->num_leaves();
      }

      pimpl->m_leaves = new (std::nothrow) Pimpl*[pimpl->m_num_leaves]{};

      // copy leaves and sort leaves from children
      for (int i=0, offset=0; i<pimpl->m_num_children; ++i) {
        const int n = pimpl->child(i)->num_leaves();
        std::copy( pimpl->child(i)->m_leaves
                 , pimpl->child(i)->m_leaves + n
                 , pimpl->m_leaves + offset
                 );
        offset += n;
      }

      std::sort( pimpl->m_leaves, pimpl->m_leaves + pimpl->m_num_leaves
               , []( const Pimpl * a, const Pimpl * b ) noexcept {
                   return a->id() < b->id();
                 }
               );
    }
    // leaf node
    else {
      pimpl->m_symmetric  = 1;
      pimpl->m_children   = nullptr;
      pimpl->m_num_leaves = 1;
      pimpl->m_leaves = new (std::nothrow) Pimpl*[1]{};
      pimpl->m_leaves[0] = pimpl;

      tmp_child_max_depth = 0;
    }

    return tmp_child_max_depth;
  }

  static void finalize( Pimpl * pimpl ) noexcept
  {
    const int n = pimpl->m_num_children;
    for (int i=0; i < n; ++i) {
      finalize( pimpl->m_children + i );
    }

    if (n > 0) {
      delete [] pimpl->m_children;
    }

    delete [] pimpl->m_leaves;
  }

private:
  int            m_depth          {0};
  int            m_index          {0};
  int            m_num_children   {0};
  int            m_num_leaves     {0};
  int            m_concurrency    {0};
  int            m_symmetric      {0};
  Pimpl *        m_parent         {nullptr}; // member_of
  hwloc_obj_t    m_obj            {nullptr};
  hwloc_cpuset_t m_cpuset         {nullptr};
  Pimpl *        m_children       {nullptr}; // partitions
  Pimpl **       m_leaves         {nullptr};

  static ExecutionResource::Pimpl * s_root;
  static hwloc_topology_t           s_topology;
  static hwloc_cpuset_t             s_process;
  static hwloc_cpuset_t             s_scratch;
};

ExecutionResource::Pimpl * ExecutionResource::Pimpl::s_root     {nullptr};
hwloc_topology_t           ExecutionResource::Pimpl::s_topology {nullptr};
hwloc_cpuset_t             ExecutionResource::Pimpl::s_process  {nullptr};
hwloc_cpuset_t             ExecutionResource::Pimpl::s_scratch  {nullptr};

//------------------------------------------------------------------------------

namespace {

const ExecutionResource::Pimpl * find_impl( ExecutionResource::Pimpl const * pimpl
                                         , hwloc_const_cpuset_t cpuset
                                         ) noexcept
{
  if( hwloc_bitmap_isequal(pimpl->get_cpuset(), cpuset) ) {
    // traverse to the lowest level
    while (pimpl->num_children() == 1) {
      pimpl = pimpl->child(0);
    }
    return pimpl;
  }
  else if ( pimpl->num_children() > 0 ) {
    for (int i=0; i < pimpl->num_children(); ++i) {
      if ( hwloc_bitmap_isincluded( cpuset, pimpl->child(i)->get_cpuset()) ) {
        return find_impl( pimpl->child(i), cpuset );
      }
    }
  }

  return nullptr;
}

std::ostream & operator<<( std::ostream & out, const ExecutionResource::Pimpl * res )
{
  if (res) {
    if (res->depth() == 0) {
      out << "root";
    } else {
      out << res->parent() << "." << res->id();
    }
  }
  else {
    out << "null";
  }
  return out;
}

} // namespace

std::ostream & operator<<( std::ostream & out, const ExecutionResource res )
{
  if (res) {
    out << "{ " << ExecutionResource::impl_get_pimpl(res) << " : " << res.concurrency() << " }";
  }
  else {
    out << "{ null : 0 }";
  }
  return out;
}

ExecutionResource this_thread_get_bind() noexcept
{
  hwloc_cpuset_t scratch = hwloc_bitmap_alloc();
  const int err = hwloc_get_cpubind( ExecutionResource::Pimpl::topology()
                                   , scratch
                                   , HWLOC_CPUBIND_THREAD
                                   );

  const ExecutionResource::Pimpl * result = nullptr;

  const ExecutionResource::Pimpl * root = ExecutionResource::impl_get_pimpl( ExecutionResource::process() );

  if (!err) {
    result = find_impl( root , scratch );
  }

  if (!result ) { result = root; }

  hwloc_bitmap_free( scratch );

  return ExecutionResource{ result };
}

ExecutionResource this_thread_get_resource() noexcept
{
  hwloc_cpuset_t scratch = hwloc_bitmap_alloc();
  const int err = hwloc_get_last_cpu_location( ExecutionResource::Pimpl::topology()
                                             , scratch
                                             , HWLOC_CPUBIND_THREAD
                                             );

  const ExecutionResource::Pimpl * result = nullptr;

  const ExecutionResource::Pimpl * root = ExecutionResource::impl_get_pimpl( ExecutionResource::process() );

  if (!err) {
    result = find_impl( root, scratch );
  }

  if (!result ) { result = root; }

  hwloc_bitmap_free( scratch );

  return ExecutionResource{ result };
}

} // namespace Kokkos


namespace Kokkos { namespace Impl {

bool this_thread_set_binding( const ExecutionResource res ) noexcept
{
  if (  this_thread_can_bind() ) {
    const int err = hwloc_set_cpubind( ExecutionResource::Pimpl::topology()
                                     , ExecutionResource::impl_get_pimpl(res)->get_cpuset()
                                     , HWLOC_CPUBIND_THREAD | HWLOC_CPUBIND_STRICT
                                     );
    return !err;
  }
  return false;
}

}} // namespace Kokkos::Impl

//------------------------------------------------------------------------------
// sched implementation
//------------------------------------------------------------------------------
#elif defined(_GNU_SOURCE) && !defined( __APPLE__ )

#include <sched.h>
#include <unistd.h>

namespace Kokkos {

class ExecutionResource::Pimpl
{
public:

  static void initialize() noexcept
  {
    if ( is_initialized() ) return;

    CPU_ZERO( &s_cpuset );

    s_total_size = sysconf( _SC_NPROCESSORS_ONLN );

    if (s_total_size <= 0) {
      s_total_size = 1;
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
            CPU_OR( &s_cpuset, &s_cpuset, &tmp );
          }
        }
      #else
        sched_getaffinity( 0
                         , sizeof(cpu_set_t)
                         , &s_cpuset
                         );
      #endif
    }


    s_concurrency = CPU_COUNT( &s_cpuset );

    s_pimpl = new (std::nothrow) int[s_concurrency + 1]{};

    s_pimpl[0] = Pimpl{-1};
    for (int i=0, j=1; i<s_total_size; ++i) {
      if( CPU_ISSET( i , &s_cpuset ) ) {
        s_pimpl[j] = Pimpl{i+1};
        ++j;
      }
    }
  }

  static void finalize() noexcept
  {
    if ( !is_initialized() ) return;

    if ( s_total_size ==  1 ) {
      s_total_size = 0;
      return;
    }

    s_total_size = 0;

    s_concurrency = 0;

    CPU_ZERO( &s_cpuset );

    delete [] s_pimpl;
    s_pimpl = nullptr;

  }

  static bool is_initialized() noexcept { return s_pimpl == nullptr }

  static ExecutionResource process() noexcept { return ExecutionResource{ s_pimpl }; }

  int  id()           const noexcept { return m_id }
  int  depth()        const noexcept { return m_id < 0 ? 0 : 1;  }
  int  index()        const noexcept { return m_id;              }
  int  num_children() const noexcept { return m_id < 0 ? s_concurrency : 0; }
  int  num_leaves()   const noexcept { return m_id < 0 ? s_concurrency : 1; }
  int  concurrency()  const noexcept { return m_id < 0 ? s_concurrency : 1; }
  bool is_symmetric() const noexcept { return true;     }

  Pimpl const * parent()     const noexcept { return m_id < 0 ? nullptr : s_pimpl; }
  Pimpl const * child(int i) const noexcept { return m_id < 0 ? s_pimpl +i +1 : nullptr; }
  Pimpl const * leaf(int i)  const noexcept { return s_pimpl +i +1; }

  Pimpl()                           noexcept = default;
  Pimpl( Pimpl const &)             noexcept = default;
  Pimpl( Pimpl &&)                  noexcept = default;
  Pimpl & operator=( Pimpl const &) noexcept = default;
  Pimpl & operator=( Pimpl &&)      noexcept = default;

  Pimpl( int i ) noexcept
    : m_id{i}
  {}

  static int       s_total_size;
  static int       s_concurrency;
  static cpu_set_t s_cpuset;
  static Pimpl *   s_pimpl;

private:
  int m_id {0};  // root is any index < 0

};

int       ExecutionResource::Pimpl::s_total_size     {0};
int       ExecutionResource::Pimpl::s_concurrency    {0};
cpu_set_t ExecutionResource::Pimpl::s_cpuset         {};

ExecutionResource::Pimpl * ExecutionResource::Pimpl::s_pimpl {nullptr};

//------------------------------------------------------------------------------

std::ostream & operator<<( std::ostream & out, const ExecutionResource res )
{

  if (res) {
    if ( res.id() >= 0 ) {
      out << "{ " << res.id() << " : " << res.concurrency() << " }";
    }
    else {
      out << "{ root : " << res.concurrency() << " }";
    }
  }
  else {
    out << "{ null : 0 }";
  }
  return out;
}

ExecutionResource this_thread_get_bind() noexcept
{
  cpu_set_t tmp;
  CPU_ZERO( &tmp );
  const int err = sched_getaffinity( 0
                                   , sizeof( cpu_set_t )
                                   , & tmp
                                   );

  ExecutionResource::Pimpl const * pimpl = ExecutionResource::Pimpl::s_pimpl;

  const int count = CPU_COUNT( &tmp );
  if ( err || count > 1 || count <= 0 ) {
    return ExecutionResource{ pimpl };
  }

  int gid = 0;
  for (; gid<ExecutionResource::Pimpl::s_total_size; ++gid) {
    if ( CPU_ISSET( gid, &tmp ) ) break;
  }

  int i = 0;
  for (; i<ExecutionResource::Pimpl::s_concurrency; ++i) {
    if ( gid == pimpl[i+1].id() ) break;
  }

  return ExecutionResource{ pimpl +i +1 };
}


ExecutionResource this_thread_get_resource() noexcept
{
  ExecutionResource::Pimpl const * pimpl = ExecutionResource::Pimpl::s_pimpl;

  const int gid = sched_getcpu();

  int i = 0;
  for (; i<ExecutionResource::Pimpl::s_concurrency; ++i) {
    if ( gid == pimpl[i+1].id() ) break;
  }

  return ExecutionResource{ pimpl +i +1 };
}

} // namespace Kokkos

namespace Kokkos { namespace Impl {

bool this_thread_set_binding( const ExecutionResource res ) noexcept
{
  if (  this_thread_can_bind() ) {
    int err = 0;
    if (res.id() == -1) {
      err = sched_setaffinity( 0
                             , sizeof(cpu_set_t)
                             , &ExecutionResource::Pimpl::s_cpuset
                             );
    } else {
      cpu_set_t tmp;
      CPU_ZERO( &tmp );
      CPU_SET( res.id(), &tmp );
      err = sched_setaffinity( 0
                             , sizeof(cpu_set_t)
                             , &tmp
                             );
    }
    return !err;
  }
  return false;
}

}} // namespace Kokkos::Impl

//------------------------------------------------------------------------------
// all other implementation
//------------------------------------------------------------------------------
#else

#if defined(_GNU_SOURCE)
#include <unistd.h>
#endif

namespace Kokkos {

class ExecutionResource::Pimpl
{
public:

  static void initialize() noexcept
  {
    if (s_root) return;

    #if defined(_OPENMP)
      const int n = omp_get_num_procs();
    #elif defined(_GNU_SOURCE)
      const int n = sysconf( _SC_NPROCESSORS_ONLN );
    #else
      const int n = 1;
    #endif

    s_root = new Pimpl{n};
  }

  static void finalize() noexcept
  {
    if (s_root) {
      delete s_root;
      s_root = nullptr;
    }
  }

  static bool is_initialized() noexcept { return s_root; }

  static ExecutionResource process() noexcept { return ExecutionResource{ s_root }; }

  int  id()           const noexcept { return 0              }
  int  depth()        const noexcept { return 0;             }
  int  index()        const noexcept { return 0;             }
  int  num_children() const noexcept { return 0;             }
  int  num_leaves()   const noexcept { return 1;             }
  int  concurrency()  const noexcept { return m_concurrency; }
  bool is_symmetric() const noexcept { return true;          }

  Pimpl const * parent()   const noexcept { return nullptr; }
  Pimpl const * child(int) const noexcept { return nullptr; }
  Pimpl const * leaf(int)  const noexcept { return s_root;  }

  Pimpl()                           noexcept = default;
  Pimpl( Pimpl const &)             noexcept = default;
  Pimpl( Pimpl &&)                  noexcept = default;
  Pimpl & operator=( Pimpl const &) noexcept = default;
  Pimpl & operator=( Pimpl &&)      noexcept = default;

private:
  int m_concurrency {0};

  static Pimpl * s_root;
};

ExecutionResource::Pimpl * ExecutionResource::Pimpl::s_root {nullptr};

//------------------------------------------------------------------------------

std::ostream & operator<<( std::ostream & out, const ExecutionResource res )
{

  if (res) {
    out << "{ root : " << res.concurrency() << " }";
  }
  else {
    out << "{ null : 0 }";
  }
  return out;
}

ExecutionResource this_thread_get_bind() noexcept
{
  return ExecutionResource::process();
}


ExecutionResource this_thread_get_resource() noexcept
{
  return ExecutionResource::process();
}

} // namespace Kokkos

namespace Kokkos { namespace Impl {

bool this_thread_set_binding( const ExecutionResource res ) noexcept
{
  return false;
}

}} // namespace Kokkos::Impl

#endif


//------------------------------------------------------------------------------
// Common behavior
//------------------------------------------------------------------------------

namespace Kokkos {

bool ExecutionResource::is_initialized() noexcept
{ return Pimpl::is_initialized(); }

ExecutionResource ExecutionResource::process() noexcept
{ return Pimpl::process(); }

int ExecutionResource::depth() const noexcept
{ return m_pimpl->depth(); }

int ExecutionResource::id() const noexcept
{ return m_pimpl->id(); }

int ExecutionResource::concurrency() const noexcept
{ return m_pimpl->concurrency(); }

int ExecutionResource::num_leaves() const noexcept
{ return m_pimpl->num_leaves(); }

ExecutionResource ExecutionResource::leaf(int i) const noexcept
{ return ExecutionResource{ m_pimpl->leaf(i) }; }

ExecutionResource ExecutionResource::member_of() const noexcept
{ return ExecutionResource{ m_pimpl->parent() }; }

int ExecutionResource::num_partitions() const noexcept
{ return m_pimpl->num_children(); }

ExecutionResource ExecutionResource::partition(int i) const noexcept
{ return ExecutionResource{ m_pimpl->child(i) }; }

bool ExecutionResource::is_symmetric() const noexcept
{ return m_pimpl->is_symmetric(); }

} // namespace Kokkos

//------------------------------------------------------------------------------

namespace Kokkos { namespace Impl {

// called by Kokkos::initialize()
void initialize_execution_resources() noexcept
{ ExecutionResource::Pimpl::initialize(); }

// called by Kokkos::finalize()
void finalize_execution_resources() noexcept
{ ExecutionResource::Pimpl::finalize(); }

}} // namespace Kokkos::Impl


