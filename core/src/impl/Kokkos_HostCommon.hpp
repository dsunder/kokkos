
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

#ifndef KOKKOS_IMPL_HOSTCOMMON_HPP
#define KOKKOS_IMPL_HOSTCOMMON_HPP

#include <Kokkos_Macros.hpp>
#include <impl/Kokkos_ThreadExecutionResource.hpp>

#if defined( KOKKOS_ENABLE_OPENMP )
#include <omp.h>
#elif defined( KOKKOS_ENABLE_STDTHREADS )
#include <impl/Kokkos_STDThreadPool.hpp>
#endif

namespace Kokkos { namespace Impl { namespace Host {

bool initialize( int max_threads = 0 ) noexcept;
bool finialize() noexcept;

int  thread_limit() noexcept;

#if   defined( KOKKOS_ENABLE_OPENMP )

struct Local
{
  void ** partition_data     ; // only set on master threads
  void *  thread_data        ; // set on all threads
  int     tid                ; // unique id:  0 <= tid < thread_limit()
  bool    in_parallel        ; // only set on the master thread
  bool    reference_tracking ; // is Kokkos::View reference counting enabled
};

extern Local t_host;
#pragma omp threadprivate( t_host )

inline bool in_parallel() noexcept
{
  return t_host.partition_data==nullptr || t_host.in_parallel;
}

inline bool is_master() noexcept
{
  return t_host.partition_data != nullptr;
}

inline int tid() noexcept
{
  return t_host.tid;
}

inline int concurrency() noexcept
{
  return !in_parallel() ? omp_get_max_threads() : 1;
}

inline int pool_size() noexcept
{
  return in_parallel() ? omp_get_num_threads() : 1;
}

inline int pool_rank() noexcept
{
  return in_parallel() ? omp_get_thread_num() : 0;
}



inline bool reference_tracking() noexcept
{
  return t_host.reference_tracking;
}

inline void reference_tracking( const bool b ) noexcept
{
  t_host.reference_tracking = b;
}

#elif defined( KOKKOS_ENABLE_STDTHREADS )

#elif defined( KOKKOS_ENABLE_SERIAL )

#else
#error "Error: No host execution space detected."
#endif






}}} // namespace Kokkos::Impl::Host


#endif // KOKKOS_IMPL_HOSTCOMMON_HPP
