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

#ifndef KOKKOS_OPENMPEXEC_HPP
#define KOKKOS_OPENMPEXEC_HPP

#include <Kokkos_Macros.hpp>
#if defined( KOKKOS_ENABLE_OPENMP )

#if !defined(_OPENMP)
#error "You enabled Kokkos OpenMP support without enabling OpenMP in the compiler!"
#endif

#include <Kokkos_OpenMP.hpp>

#include <impl/Kokkos_Traits.hpp>
#include <impl/Kokkos_HostThreadTeam.hpp>
#include <impl/Kokkos_ThreadResource.hpp>

#include <impl/Kokkos_CacheBlockedArray.hpp>

#include <Kokkos_Atomic.hpp>

#include <Kokkos_UniqueToken.hpp>

#include <iostream>
#include <sstream>
#include <fstream>

#include <omp.h>

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos { namespace Impl {

class OpenMPExec;

//----------------------------------------------------------------------------
/** \brief  Data for OpenMP thread execution */

class OpenMPExec
{
public:

  friend class Kokkos::OpenMP ;

  void clear_thread_data();

  static void validate_partition( const int nthreads
                                , int & num_partitions
                                , int & partition_size
                                );

private:
  OpenMPExec( int arg_master_tid )
    : m_master_tid{ arg_master_tid }
    , m_pool( concurrency() )
  {}

  ~OpenMPExec()
  {
    clear_thread_data();
  }

  int m_master_tid;

  CacheBlockedArray< HostThreadTeamData * > m_pool;

  static bool s_is_partitioned;
  static bool s_is_initialized;

  static int s_num_partitions;
  static int s_partition_size;

  static OpenMPExec * s_instance;
  static CacheBlockedArray< OpenMPExec * > s_partition_instances;

public:

  static OpenMPExec * instance() noexcept
  {
    return !s_is_partitioned
         ? s_instance
         : s_partition_instances[tid()]
         ;
  }

  static void set_instance( OpenMPExec * ptr ) noexcept
  {
    if (!s_is_partitioned) {
      s_instance = ptr;
    }
    else {
      s_partition_instances[tid()] = ptr;
    }
  }

  static int max_concurrency() noexcept
  {
    return  Impl::ThreadResource::process().concurrency();
  }

  static int tid() noexcept
  {
    return !s_is_partitioned
         ? omp_get_thread_num()
         : omp_get_ancestor_thread_num(1)*s_partition_size + omp_get_ancestor_thread_num(2);
         ;
  }

  static bool is_partitioned() noexcept { return s_is_partitioned; }

  static void verify_is_master( const char * const );

  static int concurrency() noexcept
  {
    return !in_parallel()
         ? omp_get_max_threads()
         : omp_get_num_threads()
         ;
  }

  static int pool_rank()   noexcept
  {
    return !Impl::OpenMPExec::is_partitioned() || omp_get_level() >= 2
         ?  omp_get_thread_num()
         :  0
         ;
  }

  static bool in_parallel() noexcept
  {
    return omp_in_parallel() && ( !s_is_partitioned || omp_get_level() > 2 );
  }

  void resize_thread_data( size_t pool_reduce_bytes
                         , size_t team_reduce_bytes
                         , size_t team_shared_bytes
                         , size_t thread_local_bytes );

  inline
  HostThreadTeamData * get_thread_data() const noexcept
  { return m_pool[pool_rank()]; }

  inline
  HostThreadTeamData * get_thread_data( int i ) const noexcept
  { return m_pool[i]; }
};


}} // namespace Kokkos::Impl

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {

inline
bool OpenMP::is_initialized() noexcept
{ return Impl::OpenMPExec::s_is_initialized; }

inline
bool OpenMP::in_parallel( OpenMP const& ) noexcept
{
  Impl::OpenMPExec::in_parallel();
}

inline
int OpenMP::concurrency() noexcept
{
  return Impl::OpenMPExec::concurrency();
}

template <typename F>
void OpenMP::partition_master( F const& f
                             , int num_partitions
                             , int partition_size
                             )
{
  using Exec = Impl::OpenMPExec;

  if (omp_get_nested() && !Impl::OpenMPExec::is_partitioned() ) {

    Exec::s_is_partitioned = true;

    // setup partitions
    if (Exec::s_num_partitions != num_partitions || Exec::s_partition_size != partition_size) {

      OpenMP::memory_space space;

      // delete the old partitions
      if ( Exec::s_num_partitions > 0 ) {
        #pragma omp parallel num_threads(Exec::s_num_partitions) proc_bind(spread)
        {
          auto tmp = Exec::instance();
          tmp->~Exec();

          space.deallocate( tmp, sizeof(Exec) );
        }
      }

      Exec::validate_partition( Exec::max_concurrency(), num_partitions, partition_size );

      Exec::s_num_partitions = num_partitions;
      Exec::s_partition_size = partition_size;

      #pragma omp parallel num_threads(Exec::s_num_partitions) proc_bind(spread)
      {
        void * const ptr = space.allocate( sizeof(Exec) );
        Exec * tmp = new (ptr) Exec( Exec::tid() );

        omp_set_num_threads(Exec::s_partition_size);

        size_t pool_reduce_bytes  =   32 * partition_size ;
        size_t team_reduce_bytes  =   32 * partition_size ;
        size_t team_shared_bytes  = 1024 * partition_size ;
        size_t thread_local_bytes = 1024 ;

        tmp->resize_thread_data( pool_reduce_bytes
                               , team_reduce_bytes
                               , team_shared_bytes
                               , thread_local_bytes
                               );

        // set threads in team to point to same instance
        #pragma omp parallel num_threads(Exec::s_partition_size) proc_bind(spread)
        {
          Exec::set_instance(tmp);
        }
      }
    }


    #pragma omp parallel num_threads(Exec::s_num_partitions) proc_bind(spread)
    {
      omp_set_num_threads(Exec::s_partition_size);
      f( omp_get_thread_num(), omp_get_num_threads() );
    }

    Exec::s_is_partitioned = false;
  }
  else {
    // nested openmp not enabled
    f(0,1);
  }
}


namespace Experimental {

template<>
class MasterLock<OpenMP>
{
public:
  void lock()     { omp_set_lock( &m_lock );   }
  void unlock()   { omp_unset_lock( &m_lock ); }
  bool try_lock() { return static_cast<bool>(omp_test_lock( &m_lock )); }

  MasterLock()  { omp_init_lock( &m_lock ); }
  ~MasterLock() { omp_destroy_lock( &m_lock ); }

  MasterLock( MasterLock const& ) = delete;
  MasterLock( MasterLock && )     = delete;
  MasterLock & operator=( MasterLock const& ) = delete;
  MasterLock & operator=( MasterLock && )     = delete;

private:
  omp_lock_t m_lock;

};

template<>
class UniqueToken< OpenMP, UniqueTokenScope::Instance>
{
public:
  using execution_space = OpenMP;
  using size_type       = int;

  /// \brief create object size for concurrency on the given instance
  ///
  /// This object should not be shared between instances
  UniqueToken( execution_space const& = execution_space() ) noexcept {}

  /// \brief upper bound for acquired values, i.e. 0 <= value < size()
  KOKKOS_INLINE_FUNCTION
  int size() const noexcept
    {
      #if defined( KOKKOS_ACTIVE_EXECUTION_MEMORY_SPACE_HOST )
      return Impl::OpenMPExec::concurrency();
      #else
      return 0 ;
      #endif
    }

  /// \brief acquire value such that 0 <= value < size()
  KOKKOS_INLINE_FUNCTION
  int acquire() const  noexcept
    {
      #if defined( KOKKOS_ACTIVE_EXECUTION_MEMORY_SPACE_HOST )
      return Impl::OpenMPExec::pool_rank();
      #else
      return 0 ;
      #endif
    }

  /// \brief release a value acquired by generate
  KOKKOS_INLINE_FUNCTION
  void release( int ) const noexcept {}
};

template<>
class UniqueToken< OpenMP, UniqueTokenScope::Global>
{
public:
  using execution_space = OpenMP;
  using size_type       = int;

  /// \brief create object size for concurrency on the given instance
  ///
  /// This object should not be shared between instances
  UniqueToken( execution_space const& = execution_space() ) noexcept {}

  /// \brief upper bound for acquired values, i.e. 0 <= value < size()
  KOKKOS_INLINE_FUNCTION
  int size() const noexcept
    {
      #if defined( KOKKOS_ACTIVE_EXECUTION_MEMORY_SPACE_HOST )
      return Impl::ThreadResource::process().concurrency();
      #else
      return 0 ;
      #endif
    }

  /// \brief acquire value such that 0 <= value < size()
  KOKKOS_INLINE_FUNCTION
  int acquire() const noexcept
    {
      #if defined( KOKKOS_ACTIVE_EXECUTION_MEMORY_SPACE_HOST )
      return Impl::OpenMPExec::tid();
      #else
      return 0 ;
      #endif
    }

  /// \brief release a value acquired by generate
  KOKKOS_INLINE_FUNCTION
  void release( int ) const noexcept {}
};

} // namespace Experimental


#ifdef KOKKOS_ENABLE_DEPRECATED_CODE

inline
int OpenMP::thread_pool_size() noexcept
{
  return   OpenMP::in_parallel()
         ? omp_get_num_threads()
         : omp_get_max_threads()
         ;
}

KOKKOS_INLINE_FUNCTION
int OpenMP::thread_pool_rank() noexcept
{
#if defined( KOKKOS_ACTIVE_EXECUTION_MEMORY_SPACE_HOST )
  return Impl::OpenMPExec::pool_rank();
#else
  return -1 ;
#endif
}

inline
int OpenMP::thread_pool_size( int depth )
{
  return depth < 2
         ? thread_pool_size()
         : 1;
}

inline
int OpenMP::hardware_thread_id() noexcept
{
  return Impl::OpenMPExec::tid();
}

inline
int OpenMP::max_hardware_threads() noexcept
{
  return  Impl::OpenMPExec::max_concurrency();
}


#endif

} // namespace Kokkos

#endif
#endif /* #ifndef KOKKOS_OPENMPEXEC_HPP */

