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

void validate_partition( const int nthreads
                       , int & num_partitions
                       , int & partition_size
                       );

void verify_is_master( const char * const );

void resize_thread_data( size_t pool_reduce_bytes
                       , size_t team_reduce_bytes
                       , size_t team_shared_bytes
                       , size_t thread_local_bytes
                       );

}} // namespace Kokkos::Impl

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {

inline bool OpenMP::in_parallel( OpenMP const& ) noexcept
{
  return Impl::host_in_parallel();
}

inline int OpenMP::concurrency() noexcept
{
  return Impl::host_pool_concurrency();
}

inline int OpenMP::num_threads() noexcept
{
  return Impl::host_pool_size();
}

inline int OpenMP::thread_rank() noexcept
{
  return Impl::host_pool_rank();
}

template <typename F>
void OpenMP::partition_master( F const& f
                             , int num_partitions
                             , int partition_size
                             )
{
  Exec::validate_partition( Impl::host_pool_concurrency(), num_partitions, partition_size );

  if (omp_get_nested() && num_partitions > 1 ) {
    using Exec = Impl::OpenMPExec;

    Impl::HostThreadLocal prev = Impl::t_host_local;


    OpenMP::memory_space space;

    #if KOKKOS_OPENMP_VERSION >= 40
    #pragma omp parallel num_threads( num_partitions ) proc_bind(spread)
    #else
    #pragma omp parallel num_threads( num_partitions )
    #endif
    {
      omp_set_num_threads(partition_size);

      const size_t partition_data_size = sizeof(HostThreadTeamData*) * partition_size;

      Impl::HostThreadTeamData ** partition_data = (Impl::HostThreadTeamData**)space.allocate( partition_data_size );

      Impl::set_host_partition_data( partition_data );

      // TODO merge into resize_thread_data
      #if KOKKOS_OPENMP_VERSION >= 40
      #pragma omp parallel num_threads( partition_size ) proc_bind(spread)
      #else
      #pragma omp parallel num_threads( partition_size )
      #endif
      {
        partition_data[ Impl::host_pool_rank() ] = Impl::host_thread_data();
      }

      size_t pool_reduce_bytes  =   32 * partition_size ;
      size_t team_reduce_bytes  =   32 * partition_size ;
      size_t team_shared_bytes  = 1024 * partition_size ;
      size_t thread_local_bytes = 1024 ;

      Impl::resize_thread_data( pool_reduce_bytes
                              , team_reduce_bytes
                              , team_shared_bytes
                              , thread_local_bytes
                              );

      f( omp_get_thread_num(), omp_get_num_threads() );

      space.deallocate( partition_data, partition_data_size );
    }

    // restore thread locals
    Impl::set_host_partition_data( prev.partition_data );
    Impl::set_host_thread_data( prev.thread_data );
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
    return Impl::host_in_parallel()
         ? Impl::host_pool_size()
         : Impl::host_pool_concurrency()
         ;
    #else
    return 0 ;
    #endif
  }

  /// \brief acquire value such that 0 <= value < size()
  KOKKOS_INLINE_FUNCTION
  int acquire() const  noexcept
  {
    #if defined( KOKKOS_ACTIVE_EXECUTION_MEMORY_SPACE_HOST )
    return Impl::host_pool_rank();
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
      return Kokkos::Impl::host_thread_limit();
      #else
      return 0 ;
      #endif
    }

  /// \brief acquire value such that 0 <= value < size()
  KOKKOS_INLINE_FUNCTION
  int acquire() const noexcept
    {
      #if defined( KOKKOS_ACTIVE_EXECUTION_MEMORY_SPACE_HOST )
      return Kokkos::Impl::host_tid();
      #else
      return 0;
      #endif
    }

  /// \brief release a value acquired by generate
  KOKKOS_INLINE_FUNCTION
  void release( int ) const noexcept {}
};

} // namespace Experimental


#if !defined( KOKKOS_DISABLE_DEPRECATED )

inline
int OpenMP::thread_pool_size( int depth ) noexcept
{
  if ( depth > 0 ) return 1;
  return OpenMP::in_parallel()
       ? Impl::host_pool_size()
       : Impl::host_pool_concurrency();
       ;
}

KOKKOS_INLINE_FUNCTION
int OpenMP::thread_pool_rank() noexcept
{
#if defined( KOKKOS_ACTIVE_EXECUTION_MEMORY_SPACE_HOST )
  return Impl::host_pool_rank();
#else
  return -1 ;
#endif
}

KOKKOS_INLINE_FUNCTION
int OpenMP::hardware_thread_id() noexcept
{
#if defined( KOKKOS_ACTIVE_EXECUTION_MEMORY_SPACE_HOST )
  return Impl::host_tid();
#else
  return -1 ;
#endif
}

inline
int OpenMP::max_hardware_threads() noexcept
{
  return Impl::host_thread_limit();
}

#endif // KOKKOS_DISABLE_DEPRECATED

} // namespace Kokkos

#endif
#endif /* #ifndef KOKKOS_OPENMPEXEC_HPP */

