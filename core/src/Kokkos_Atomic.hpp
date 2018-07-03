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
// Questions? Contact Christian R. Trott (crtrott@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

#ifndef KOKKOS_ATOMIC_HPP
#define KOKKOS_ATOMIC_HPP

#include <Kokkos_Macros.hpp>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <type_traits>

//------------------------------------------------------------------------------
// check that KOKKOS_ENABLE_###_ATOMICS is not already defined
//------------------------------------------------------------------------------
#if defined( KOKKOS_ENABLE_CUDA_ATOMICS )
  #error "KOKKOS_ENABLE_CUDA_ATOMICS already defined"
#endif
#if defined( KOKKOS_ENABLE_ROCM_ATOMICS )
  #error "KOKKOS_ENABLE_ROCM_ATOMICS already defined"
#endif
#if defined( KOKKOS_ENABLE_SERIAL_ATOMICS )
  #error "KOKKOS_ENABLE_SERIAL_ATOMICS already defined"
#endif
#if defined( KOKKOS_ENABLE_OPENMP_ATOMICS )
  #error "KOKKOS_ENABLE_OPENMP_ATOMICS already defined"
#endif
#if defined( KOKKOS_ENABLE_WIN_ATOMICS )
  #error "KOKKOS_ENABLE_WIN_ATOMICS already defined"
#endif
#if defined( KOKKOS_ENABLE_GNU_ATOMICS )
  #error "KOKKOS_ENABLE_GNU_ATOMICS already defined"
#endif
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// Enable the correct atomic implementation
//------------------------------------------------------------------------------
// TODO CUDA_CLANG_WORKAROUND needs rethinking
#if  defined( KOKKOS_ENABLE_CUDA ) \
 && (defined( __CUDA_ARCH__ ) || defined( KOKKOS_IMPL_CUDA_CLANG_WORKAROUND ))
  // offload onto a cuda enabled gpu
  #define KOKKOS_ENABLE_CUDA_ATOMICS
  #include <atomic/Kokkos_Atomic_Cuda.hpp>
#elif defined( KOKKOS_ENABLE_ROCM ) && defined( __HCC_ACCELERATOR__ )
  // offload onto a rocm enabled gpu
  #define KOKKOS_ENABLE_ROCM_ATOMICS
  #include <atomic/Kokkos_Atomic_ROCm.hpp>
#elif defined( KOKKOS_INTERNAL_NOT_PARALLEL )
  // host is serial only
  #define KOKKOS_ENABLE_SERIAL_ATOMICS
  #include <atomic/Kokkos_Atomic_Serial.hpp>
#elif defined( KOKKOS_INTERNAL_OPENMP_ATOMICS )
  // host uses openmp atomics
  #define KOKKOS_ENABLE_OPENMP_ATOMICS
  #include <atomic/Kokkos_Atomic_OpenMP.hpp>
#elif defined( _WIN32 )
  // host uses win atomics
  #define KOKKOS_ENABLE_WIN_ATOMICS
  #include <atomic/Kokkos_Atomic_Windows.hpp>
#else
  // host uses gnu atomic builtins
  // Linux, Power, Arm, Apple
  //   g++, clang++, icpc, pgc++, xlc++, CC
  #define KOKKOS_ENABLE_GNU_ATOMICS
  #include <atomic/Kokkos_Atomic_GNU.hpp>
#endif
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// All atomic functions with std::memory_order omitted.
//
// When the memory_order is omitted default to std::memory_order_relaxed.
//
// NOTE: This is a change in behavior from the C++ standard.  Per the standard
// whenever the memory_order is omitted an atomic function should default to
// std::memory_order_seq_cst.
//------------------------------------------------------------------------------
#include <atomic/Kokkos_Atomic_Simple.hpp>
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// Enable Kokkos::atomic_ref<T>
//------------------------------------------------------------------------------
//#include <atomic/Kokkos_Atomic_Ref.hpp>
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// Enable Kokkos::memory_fence()
// Enable Kokkos::store_fence()
// Enable Kokkos::load_fence()
//------------------------------------------------------------------------------
#include <atomic/Kokkos_Memory_Fence.hpp>

//------------------------------------------------------------------------------
// Enable the deprecated atomic api
// TODO: protect with KOKKOS_ENABLE_DEPRECATED_CODE
//------------------------------------------------------------------------------
#include <atomic/Kokkos_Atomic_Deprecated.hpp>

namespace Kokkos {

using std::memory_order_relaxed;
using std::memory_order_acquire;
using std::memory_order_release;
using std::memory_order_acq_rel;
using std::memory_order_seq_cst;

inline constexpr const char * atomic_query_version() noexcept
{
#if   defined( KOKKOS_ENABLE_CUDA_ATOMICS )
  return "KOKKOS_ENABLE_CUDA_ATOMICS" ;
#elif defined( KOKKOS_ENABLE_ROCM_ATOMICS )
  return "KOKKOS_ENABLE_ROCM_ATOMICS" ;
#elif defined( KOKKOS_ENABLE_GNU_ATOMICS )
  return "KOKKOS_ENABLE_GNU_ATOMICS" ;
#elif defined( KOKKOS_ENABLE_OPENMP_ATOMICS )
  return "KOKKOS_ENABLE_OPENMP_ATOMICS" ;
#elif defined( KOKKOS_ENABLE_WINDOWS_ATOMICS )
  return "KOKKOS_ENABLE_WINDOWS_ATOMICS";
#elif defined( KOKKOS_ENABLE_SERIAL_ATOMICS )
  return "KOKKOS_ENABLE_SERIAL_ATOMICS";
#endif
}

} // namespace Kokkos

#endif /* KOKKOS_ATOMIC_HPP */

