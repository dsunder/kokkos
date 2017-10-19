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

#if defined( KOKKOS_ENABLE_OPENMP )
#include <omp.h>
#endif

namespace Kokkos { namespace Impl {

#if defined( KOKKOS_ENABLE_OPENMP )

inline int thread_pool_rank()        noexcept { return omp_get_thread_num();  }
inline int thread_pool_size()        noexcept { return omp_get_num_threads(); }
inline int thread_pool_level()       noexcept { return omp_get_level();       }
inline int thread_pool_concurrency() noexcept { return omp_get_max_threads(); }

#elif defined( KOKKOS_ENABLE_THREADS )

  #if defined( KOKKOS_ENABLE_STDTHREADS )
    extern thread_local int t_pool_rank{0};
    extern thread_local int t_pool_size{0};
    extern thread_local int t_pool_level{0};
    extern thread_local int t_pool_concurrency{0};
  #else
    extern __thread int t_pool_rank{0};
    extern __thread int t_pool_size{0};
    extern __thread int t_pool_level{0};
    extern __thread int t_pool_concurrency{0};
  #endif

inline int thread_pool_rank()        noexcept { return t_pool_rank;        }
inline int thread_pool_size()        noexcept { return t_pool_size;        }
inline int thread_pool_level()       noexcept { return t_pool_level;       }
inline int thread_pool_concurrency() noexcept { return t_pool_concurrency; }

inline void set_thread_pool_rank( int n )        noexcept { t_pool_rank        = n; }
inline void set_thread_pool_size( int n )        noexcept { t_pool_size        = n; }
inline void set_thread_pool_level( int n )       noexcept { t_pool_level       = n; }
inline void set_thread_pool_concurrency( int n ) noexcept { t_pool_concurrency = n; }


#else // Serial only

inline int thread_pool_rank()        noexcept { return 0; }
inline int thread_pool_size()        noexcept { return 1; }
inline int thread_pool_level()       noexcept { return 0; }
inline int thread_pool_concurrency() noexcept { return 1; }

#endif

}} // namespace Kokkos::Impl

#endif // KOKKOS_IMPL_THREAD_PRIVATE_HPP
