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

#ifndef KOKKOS_OPENMP_HPP
#define KOKKOS_OPENMP_HPP

#include <Kokkos_Macros.hpp>
#if defined( KOKKOS_ENABLE_OPENMP)

#include <Kokkos_Core_fwd.hpp>

#include <cstddef>
#include <iosfwd>
#include <Kokkos_HostSpace.hpp>

#ifdef KOKKOS_ENABLE_HBWSPACE
#include <Kokkos_HBWSpace.hpp>
#endif

#include <Kokkos_ScratchSpace.hpp>
#include <Kokkos_Parallel.hpp>
#include <Kokkos_TaskScheduler.hpp>
#include <Kokkos_Layout.hpp>
#include <impl/Kokkos_Tags.hpp>

/*--------------------------------------------------------------------------*/

namespace Kokkos {

namespace Impl {
class OpenMPExec;
}

/// \class OpenMP
/// \brief Kokkos device for multicore processors in the host memory space.
class OpenMP {
public:
  //! Tag this class as a kokkos execution space
  using execution_space = OpenMP;

  using memory_space =
  #ifdef KOKKOS_ENABLE_HBWSPACE
    Experimental::HBWSpace;
  #else
    HostSpace;
  #endif

  //! This execution space preferred device_type
  using device_type          = Kokkos::Device< execution_space, memory_space >;
  using array_layout         = LayoutRight;
  using size_type            = memory_space::size_type;
  using scratch_memory_space = ScratchMemorySpace< OpenMP >;

  static constexpr const char* name() noexcept { return "OpenMP"; }

  OpenMP() noexcept = default;

  /// \brief Initialize the default execution space
  ///
  /// if ( thread_count <= 0 )
  ///   use omp_get_max_threads()
  /// if ( 0 < thread_count < omp_get_num_procs() )
  ///   use the given number of threads
  /// if ( thread_count > omp_get_num_procs()
  ///   use omp_get_num_procs threads
  static void initialize( int thread_count = -1 );

  /// \brief Free any resources being consumed by the default execution space
  static void finalize();

  /// \brief is the default execution space initialized for current 'master' thread
  static bool is_initialized() noexcept;

  /// \brief Print configuration information to the given output stream.
  static void print_configuration( std::ostream & , const bool verbose = false );

  /// \brief is the instance asynchronous
  static constexpr bool is_asynchronous( OpenMP const& = OpenMP() ) noexcept { return false };

  /// \brief Wait until all dispatched functors complete on the given instance
  inline static void fence( OpenMP const& = OpenMP() ) noexcept {};

  /// \brief is the instance running a parallel algorithm
  inline static bool in_parallel( OpenMP const& = OpenMP() ) noexcept;

  /// \brief total number of threads available to the current pool
  inline static int concurrency() noexcept;

  /// \brief total number of threads in the current running parallel algorithm
  inline static int num_threads() noexcept;

  /// \brief rank of the thread in the current running parallel algorithm
  inline static int thread_rank() noexcept;

  /// \brief Partition the default instance and call 'f' on each new 'master' thread
  ///
  /// Func is a functor with the following signiture
  ///   void( int partition_id, int num_partitions )
  template <typename F>
  static void partition_master( F const& f
                              , int requested_num_partitions = 0
                              , int requested_partition_size = 0
                              );

#if !defined( KOKKOS_DISABLE_DEPRECATED )
  /// \brief Initialize the default execution space
  static void initialize( int thread_count,
                          int use_numa_count,
                          int use_cores_per_numa = 0);

  inline
  static int thread_pool_size( int depth = 0 );

  /// \brief  The rank of the executing thread in this thread pool
  KOKKOS_INLINE_FUNCTION
  static int thread_pool_rank() noexcept;


  static void sleep() {};
  static void wake() {};

  // use UniqueToken
  inline
  static int max_hardware_threads() noexcept;

  // use UniqueToken
  KOKKOS_INLINE_FUNCTION
  static int hardware_thread_id() noexcept;
#endif

};

} // namespace Kokkos

/*--------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------*/

namespace Kokkos {
namespace Impl {

template<>
struct MemorySpaceAccess
  < Kokkos::OpenMP::memory_space
  , Kokkos::OpenMP::scratch_memory_space
  >
{
  enum { assignable = false };
  enum { accessible = true };
  enum { deepcopy   = false };
};

template<>
struct VerifyExecutionCanAccessMemorySpace
  < Kokkos::OpenMP::memory_space
  , Kokkos::OpenMP::scratch_memory_space
  >
{
  enum { value = true };
  inline static void verify( void ) { }
  inline static void verify( const void * ) { }
};

} // namespace Impl
} // namespace Kokkos

/*--------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------*/

#include <OpenMP/Kokkos_OpenMP_Exec.hpp>
#include <OpenMP/Kokkos_OpenMP_Team.hpp>
#include <OpenMP/Kokkos_OpenMP_Parallel.hpp>
#include <OpenMP/Kokkos_OpenMP_Task.hpp>

#include <KokkosExp_MDRangePolicy.hpp>
/*--------------------------------------------------------------------------*/

#endif /* #if defined( KOKKOS_ENABLE_OPENMP ) && defined( _OPENMP ) */
#endif /* #ifndef KOKKOS_OPENMP_HPP */

