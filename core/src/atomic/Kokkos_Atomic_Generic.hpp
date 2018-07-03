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

#ifndef KOKKOS_ATOMIC_GENERIC_HPP
#define KOKKOS_ATOMIC_GENERIC_HPP

#include <Kokkos_Macros.hpp>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <type_traits>


namespace Kokkos { namespace Impl {

struct AddOp {
  template <typename T>
  KOKKOS_FORCEINLINE_FUNCTION
  static constexpr
  T apply( const T& a, const typename std::remove_cv<T>::type& b) noexcept
  { return static_cast<T>(a+b); }
};

struct SubOp {
  template <typename T>
  KOKKOS_FORCEINLINE_FUNCTION
  static constexpr
  T apply( const T& a, const typename std::remove_cv<T>::type& b) noexcept
  { return static_cast<T>(a-b); }
};

struct MulOp {
  template <typename T>
  KOKKOS_FORCEINLINE_FUNCTION
  static constexpr
  T apply( const T& a, const typename std::remove_cv<T>::type& b) noexcept
  { return static_cast<T>(a*b); }
};

struct DivOp {
  template <typename T>
  KOKKOS_FORCEINLINE_FUNCTION
  static constexpr
  T apply( const T& a, const typename std::remove_cv<T>::type& b) noexcept
  { return static_cast<T>(a/b); }
};

struct ModOp {
  template <typename T>
  KOKKOS_FORCEINLINE_FUNCTION
  static constexpr
  T apply( const T& a, const typename std::remove_cv<T>::type& b) noexcept
  { return static_cast<T>(a%b); }
};

struct MinOp {
  template <typename T>
  KOKKOS_FORCEINLINE_FUNCTION
  static constexpr
  T apply( const T& a, const typename std::remove_cv<T>::type& b) noexcept
  { return a < b ? a : b; }
};

struct MaxOp {
  template <typename T>
  KOKKOS_FORCEINLINE_FUNCTION
  static constexpr
  T apply( const T& a, const typename std::remove_cv<T>::type& b) noexcept
  { return b < a ? a : b; }
};

struct AndOp {
  template <typename T>
  KOKKOS_FORCEINLINE_FUNCTION
  static constexpr
  T apply( const T& a, const typename std::remove_cv<T>::type& b) noexcept
  { return static_cast<T>(a & b); }
};

struct OrOp {
  template <typename T>
  KOKKOS_FORCEINLINE_FUNCTION
  static constexpr
  T apply( const T& a, const typename std::remove_cv<T>::type& b) noexcept
  { return static_cast<T>(a | b); }
};

struct XorOp {
  template <typename T>
  KOKKOS_FORCEINLINE_FUNCTION
  static constexpr
  T apply( const T& a, const typename std::remove_cv<T>::type& b) noexcept
  { return static_cast<T>(a ^ b); }
};

struct NandOp {
  template <typename T>
  KOKKOS_FORCEINLINE_FUNCTION
  static constexpr
  T apply( const T& a, const typename std::remove_cv<T>::type& b) noexcept
  { return static_cast<T>(~(a & b)); }
};

struct LshiftOp {
  template <typename T>
  KOKKOS_FORCEINLINE_FUNCTION
  static constexpr
  T apply( const T& a, unsigned b) noexcept
  { return static_cast<T>(a << b); }
};

struct RshiftOp {
  template <typename T>
  KOKKOS_FORCEINLINE_FUNCTION
  static constexpr
  T apply( const T& a, unsigned b) noexcept
  { return static_cast<T>(a >> b); }
};

}} // namespace Kokkos::Impl

#endif //KOKKOS_ATOMIC_GENERIC_HPP

