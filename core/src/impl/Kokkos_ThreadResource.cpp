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

#include <impl/Kokkos_ThreadResource.hpp>

#include <algorithm>
#include <cstdlib>
#include <iostream>

#if defined(_OPENMP)
#include <omp.h>
#endif

#if defined(KOKKOS_ENABLE_HWLOC)

#include <hwloc.h>

namespace Kokkos { namespace Impl {

class ThreadResource::Pimpl
{
public:

  static void initialize()
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
        std::cerr << "Error: Cannot call ThreadResource::initialize() in parallel" << std::endl;
        std::abort();
      }
    #endif

    if (err) {
      // assume entire node if unable to detect process binding
      const int num_pu = hwloc_get_nbobjs_by_type( s_topology, HWLOC_OBJ_PU );
      hwloc_bitmap_set_range( s_process, 0, num_pu -1 );
    }

    s_root = new Pimpl{};

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

  static bool is_initialized()       noexcept { return s_topology != nullptr; }

  static hwloc_topology_t topology() noexcept { return s_topology; }

  static ThreadResource process()    noexcept { return ThreadResource{ s_root }; }

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
                       )
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

      pimpl->m_leaves = new Pimpl*[pimpl->m_num_leaves]{};

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
      pimpl->m_leaves = new Pimpl*[1]{};
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

  static ThreadResource::Pimpl * s_root;
  static hwloc_topology_t        s_topology;
  static hwloc_cpuset_t          s_process;
  static hwloc_cpuset_t          s_scratch;
};

ThreadResource::Pimpl * ThreadResource::Pimpl::s_root     {nullptr};
hwloc_topology_t        ThreadResource::Pimpl::s_topology {nullptr};
hwloc_cpuset_t          ThreadResource::Pimpl::s_process  {nullptr};
hwloc_cpuset_t          ThreadResource::Pimpl::s_scratch  {nullptr};

}} // namespace Kokkos::Impl


#endif // KOKKOS_ENABLE_HWLOC

