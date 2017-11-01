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

#ifndef KOKKOS_IMPL_THREAD_PRIVATE_HPP
#define KOKKOS_IMPL_THREAD_PRIVATE_HPP

#include <Kokkos_Macros.hpp>

#include <impl/Kokkos_HostThreadTeam.hpp>

#if defined( KOKKOS_ENABLE_OPENMP )
#include <omp.h>
#endif

namespace Kokkos { namespace Impl {

  enum { MAX_THREAD_COUNT = 512 };

  int host_thread_limit()      noexcept;
  void set_host_thread_limit() noexcept;

#if defined( KOKKOS_ENABLE_OPENMP )

  struct HostThreadLocal
  {
    HostThreadTeamData ** partition_data;
    HostThreadTeamData * thread_data;
    int  tid;
    bool in_parallel;
    bool reference_tracking;
  };

  extern HostThreadLocal t_host_local;
  #pragma omp threadprivate( t_host_local )

  inline bool host_in_parallel()         noexcept { return t_host_local.in_parallel; }
  inline int  host_pool_concurrency()    noexcept { return !host_in_parallel() ? omp_get_max_threads() : 1; }
  inline int  host_pool_size()           noexcept { return  host_in_parallel() ? omp_get_num_threads() : 1; }
  inline int  host_pool_rank()           noexcept { return  host_in_parallel() ? omp_get_thread_num()  : 0; }
  inline int  host_pool_level()          noexcept { return  omp_get_level(); }
  inline bool host_reference_tracking()  noexcept { return t_host_local.reference_tracking; }
  inline int  host_tid()                 noexcept { return t_host_local.tid; }

  inline void set_host_tid(int t)                 noexcept { t_host_local.tid = t; }
  inline void set_host_reference_tracking(bool b) noexcept { t_host_local.reference_tracking=b; }
  inline void set_host_in_parallel(bool b)        noexcept { t_host_local.in_parallel=b; }

  inline HostThreadTeamData * host_thread_data()     noexcept { return t_host_local.thread_data; }
  inline HostThreadTeamData ** host_partition_data() noexcept { return t_host_local.partition_data; }

  inline void set_host_thread_data( HostThreadTeamData * d )     noexcept { t_host_local.thread_data=d; }
  inline void set_host_partition_data( HostThreadTeamData ** p ) noexcept { t_host_local.partition_data=p; }

#elif defined( KOKKOS_ENABLE_THREADS ) || defined( KOKKOS_ENABLE_STDTHREADS )

  struct HostThreadLocal
  {
    HostThreadTeamData ** partition_data;
    HostThreadTeamData * thread_data;
    int  pool_size;
    int  pool_rank;
    int  pool_concurrency;
    int  pool_level;
    int  tid;
    bool in_parallel;
    bool reference_tracking;
  };

  #if defined( KOKKOS_ENABLE_STDTHREADS )
    extern thread_local HostThreadLocal t_host_local;
  #else
    extern __thread HostThreadLocal t_host_local;
  #endif

  inline bool host_in_parallel()        noexcept { return t_host_local.in_parallel; }
  inline int  host_pool_rank()          noexcept { return t_host_local.pool_rank;        }
  inline int  host_pool_size()          noexcept { return t_host_local.pool_size;        }
  inline int  host_pool_concurrency()   noexcept { return t_host_local.pool_concurrency; }
  inline int  host_pool_level()         noexcept { return t_host_local.pool_level;       }
  inline int  host_tid()                noexcept { return t_host_local.tid; }
  inline bool host_reference_tracking() noexcept { return t_host_local.reference_tracking; }

  inline void set_host_pool_size( int n )         noexcept { t_host_local.pool_size        = n; }
  inline void set_host_pool_rank( int n )         noexcept { t_host_local.pool_rank        = n; }
  inline void set_host_pool_level( int n )        noexcept { t_host_local.pool_level       = n; }
  inline void set_host_pool_concurrency( int n )  noexcept { t_host_local.pool_concurrency = n; }
  inline void set_host_tid(int t)                 noexcept { t_host_local.tid = t; }
  inline void set_host_in_parallel(bool b)        noexcept { t_host_local.in_parallel=b; }
  inline void set_host_reference_tracking(bool b) noexcept { t_host_local.reference_tracking=b; }

  inline HostThreadTeamData * host_thread_data()     noexcept { return t_host_local.thread_data; }
  inline HostThreadTeamData ** host_partition_data() noexcept { return t_host_local.partition_data; }

  inline void set_host_thread_data( HostThreadTeamData * d )     noexcept { t_host_local.thread_data=d; }
  inline void set_host_partition_data( HostThreadTeamData ** p ) noexcept { t_host_local.partition_data=p; }

#elif defined( KOKKOS_ENABLE_SERIAL )

  struct HostThreadLocal
  {
    HostThreadTeamData * thread_data;
    bool                 reference_tracking;
  };

  extern HostThreadLocal g_host_local;

  inline constexpr bool host_in_parallel()      noexcept { return false; }
  inline constexpr int  host_pool_size()        noexcept { return 1; }
  inline constexpr int  host_pool_rank()        noexcept { return 0; }
  inline constexpr int  host_pool_concurrency() noexcept { return 1; }
  inline constexpr int  host_pool_level()       noexcept { return 0; }
  inline constexpr int  host_tid()              noexcept { return 0; }

  inline bool host_reference_tracking()           noexcept { return g_host_local.reference_tracking; }
  inline void set_host_reference_tracking(bool b) noexcept { g_host_local.reference_tracking=b; }

  inline HostThreadTeamData ** host_partition_data()      noexcept { return &t_host_local.thread_data; }
  inline HostThreadTeamData * host_thread_data()          noexcept { return g_host_local.thread_data; }
  inline void set_host_thread_data( HostThreadLocal * d ) noexcept { g_host_local.thread_data=d; }

#endif

}} // namespace Kokkos::Impl

#endif // KOKKOS_IMPL_THREAD_PRIVATE_HPP
