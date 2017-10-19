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
  int            m_num_partitions {0};
  int            m_concurrency    {0};
  int            m_level          {0};
  int            m_partition_id   {0};
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

int initialize_tree( ThreadExecutionResource::Impl * impl
                   , ThreadExecutionResource::Impl * member_of
                   , hwloc_obj_t obj
                   , hwloc_cpuset_t scratch
                   , int level
                   , int partition_id
                   ) noexcept
{
  impl->m_cpuset         = get_cpuset( obj );
  impl->m_member_of      = member_of;
  impl->m_level          = level;
  impl->m_partition_id   = partition_id;

  int npartitions = get_num_partitions( obj, scratch );
  hwloc_obj_t tmp_obj = obj;

  while ( npartitions == 1 ) {
    for (int i=0; i < tmp_obj->arity; ++i) {
      if ( intersect_process( tmp_obj->children[i], scratch) ) {
        tmp_obj = tmp_obj->children[i];
        break;
      }
    }
    npartitions = get_num_partitions( tmp_obj, scratch );
  }

  impl->m_obj            = tmp_obj;
  impl->m_num_partitions = npartitions;

  if ( npartitions > 0 ) {
    impl->m_partitions = new (std::nothrow) ThreadExecutionResource::Impl[impl->m_num_partitions]{};

    impl->m_concurrency = 0;
    for (int i=0, j=0; i < tmp_obj->arity; ++i) {
      if ( intersect_process( tmp_obj->children[i], scratch) ) {
        impl->m_concurrency += initialize_tree( impl->m_partitions + j
                                              , impl
                                              , tmp_obj->children[i]
                                              , scratch
                                              , level + 1
                                              , j
                                              );
        ++j;
      }
    }
  } else {
    impl->m_concurrency = hwloc_bitmap_weight( impl->m_cpuset );
    g_leaves.push_back( impl );
  }

  std::sort( impl->m_partitions, impl->m_partitions + impl->m_num_partitions
           , []( ThreadExecutionResource::Impl const & a, ThreadExecutionResource::Impl const & b )
           {
             return a.m_obj->logical_index < b.m_obj->logical_index;
           });

  for (int i=0; i < impl->m_num_partitions; ++i) {
    impl->m_partitions[i].m_partition_id = i;
  }

  return impl->m_concurrency;
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

//------------------------------------------------------------------------------
} // namespace

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------

ThreadExecutionResource ThreadExecutionResource::root() noexcept
{
  return ThreadExecutionResource{g_root};
}

ThreadExecutionResource ThreadExecutionResource::leaf( int i ) noexcept
{
  return ThreadExecutionResource{ g_leaves[i] };
}

int ThreadExecutionResource::num_leaves() noexcept
{
  return static_cast<int>(g_leaves.size());
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------

void ThreadExecutionResource::initialize() noexcept
{
  hwloc_topology_init( & g_topology );
  hwloc_topology_load( g_topology );
  g_process = hwloc_bitmap_alloc();
  hwloc_get_cpubind( g_topology, g_process, HWLOC_CPUBIND_PROCESS );

  int num_leaves = hwloc_bitmap_weight( g_process );

  if ( num_leaves == 0 ) {
    num_leaves = hwloc_get_nbobjs_by_type( g_topology, HWLOC_OBJ_PU );
    hwloc_bitmap_set_range( g_process, 0, num_leaves-1);
  }

  hwloc_cpuset_t scratch = hwloc_bitmap_alloc();

  g_root = new (std::nothrow) Impl;

  initialize_tree( g_root
                 , nullptr
                 , hwloc_get_root_obj( g_topology )
                 , scratch
                 , 0
                 , 0
                 );

  std::sort( g_leaves.begin(), g_leaves.end()
           , []( Impl const * a, Impl const * b ) {
              return a->m_obj->logical_index < b->m_obj->logical_index;
           });
}

void ThreadExecutionResource::finalize() noexcept
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

ThreadExecutionResource ThreadExecutionResource::member_of() const noexcept
{
  return ThreadExecutionResource{ m_pimpl->m_member_of };
}

int ThreadExecutionResource::concurrency() const noexcept
{
  return m_pimpl->m_concurrency;
}

int ThreadExecutionResource::num_partitions() const noexcept
{
  return m_pimpl->m_num_partitions;
}

ThreadExecutionResource ThreadExecutionResource::partition( int i ) const noexcept
{
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

ThreadExecutionResource this_thread_last_resource() noexcept
{
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

std::ostream & operator<<( std::ostream & out, const ThreadExecutionResource res )
{
  if (res) {
    if(res.get_impl()->m_level == 0) {
      out << "root";
    } else {
      out << res.member_of() << "." << res.get_impl()->m_partition_id;
    }
  }
  else {
    out << "null";
  }

  return out;
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------

}} // namespace Kokkos::Impl

#else

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------

#if defined( _GNU_SOURCE ) && !defined( _OPENMP )
  #include <unistd.h>
#endif

#include <iostream>

namespace Kokkos { namespace Impl {

struct ThreadExecutionResource::Impl
{
  int m_concurrency{0};
};

namespace {
ThreadExecutionResource::Impl g_root;
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------

ThreadExecutionResource ThreadExecutionResource::root() noexcept
{
  return ThreadExecutionResource{&g_root};
}

ThreadExecutionResource ThreadExecutionResource::leaf( int ) noexcept
{
  return ThreadExecutionResource{ &g_root };
}

int ThreadExecutionResource::num_leaves() noexcept
{
  return 1;
}

//------------------------------------------------------------------------------

void ThreadExecutionResource::initialize() noexcept
{
  #if defined( _OPENMP )
    g_root.m_concurrency = 0;
    #pragma omp parallel
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

void ThreadExecutionResource::finalize() noexcept
{
  g_root.m_concurrency = 0;
}

ThreadExecutionResource ThreadExecutionResource::member_of() const noexcept
{
  return ThreadExecutionResource{ nullptr };
}

int ThreadExecutionResource::concurrency() const noexcept
{
  return g_root.m_concurrency;
}

int ThreadExecutionResource::num_partitions() const noexcept
{
  return 0;
}

ThreadExecutionResource ThreadExecutionResource::partition( int i ) const noexcept
{
  return ThreadExecutionResource{ nullptr };
}

//------------------------------------------------------------------------------

bool this_thread_set_binding( const ThreadExecutionResource res ) noexcept
{ return false; }

ThreadExecutionResource this_thread_get_bind() noexcept
{ return ThreadExecutionResource{&g_root}; }

ThreadExecutionResource this_thread_last_resource() noexcept
{ return ThreadExecutionResource{&g_root}; }

std::ostream & operator<<( std::ostream & out, const ThreadExecutionResource res )
{
  out << "root";
  return out;
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------

}} // namespace Kokkos::Impl

#endif
