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

#ifndef KOKKOS_ATOMIC_SERIAL_HPP
#define KOKKOS_ATOMIC_SERIAL_HPP

#include <Kokkos_Macros.hpp>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <atomic>

//------------------------------------------------------------------------------
#define KOKKOS_INTERNAL_CHECK_VALID_ATOMIC_TYPE( type )                        \
  static_assert( !std::is_volatile<type>::value                                \
               , "Error: atomic operations require non-volatile types ("       \
                  #type " is volatile)." );                                    \
  static_assert( std::is_trivially_copyable<type>::value                       \
               , "Error: atomic operations require trivially copyable types (" \
                 #type " is not trivially copyable)." )
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
#define KOKKOS_INTERNAL_FORCEINLINE inline __attribute__((always_inline))


namespace Kokkos {

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
// Returns true if objects of size bytes always generate lock-free atomic
// instructions for the target architecture. size must resolve to a
// compile-time constant and the result also resolves to a compile-time
// constant.
//
// ptr is an optional pointer to the object that may be used to determine
// alignment. A value of 0 indicates typical alignment should be used. The
// compiler may also ignore this parameter.
KOKKOS_INTERNAL_FORCEINLINE
constexpr bool atomic_always_lock_free( size_t size, void * ptr = nullptr ) noexcept
{ return true; }


// Returns true if objects of size bytes always generate lock-free atomic
// instructions for the target architecture. If the built-in function is not
// known to be lock-free, a call is made to a runtime routine.
//
// ptr is an optional pointer to the object that may be used to determine
// alignment. A value of 0 indicates typical alignment should be used. The
// compiler may also ignore this parameter.
KOKKOS_INTERNAL_FORCEINLINE
bool atomic_is_lock_free( size_t size, void * ptr = nullptr ) noexcept
{ return true; }


// This built-in function acts as a synchronization fence between threads based
// on the specified memory order.
//
// All memory orders are valid.
KOKKOS_INTERNAL_FORCEINLINE
void atomic_thread_fence( std::memory_order order ) noexcept
{}
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------


//------------------------------------------------------------------------------
// T atomic_load(T* ptr, std::memory_order order) noexcept
//
// Returns the contents of *ptr
//
// Valid memory order variants are std::memory_order_relaxed,
// std::memory_order_seq_cst, and std::memory_order_acquire
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
template <typename T>
KOKKOS_INTERNAL_FORCEINLINE
T atomic_load( T* ptr, const std::memory_order order ) noexcept
{
  KOKKOS_INTERNAL_CHECK_VALID_ATOMIC_TYPE( T );
  return *ptr;
}
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------


//------------------------------------------------------------------------------
// void atomic_store(T* ptr, T val, std::memory_order order) noexcept
//
// Writes val into *ptr
//
// Valid memory order variants are std::memory_order_relaxed,
// std::memory_order_seq_cst, and std::memory_order_release
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
template <typename T>
KOKKOS_INTERNAL_FORCEINLINE
void atomic_store( T* ptr
            , typename std::remove_cv<T>::type val
            , const std::memory_order order
            ) noexcept
{
  KOKKOS_INTERNAL_CHECK_VALID_ATOMIC_TYPE( T );
  *ptr = val;
}
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------


//------------------------------------------------------------------------------
// T atomic_exchange(T* ptr, T val, std::memory_order order) noexcept
//
// Writes val into *ptr and returns the previous contents of *ptr.
//
// Valid memory order variants are std::memory_order_relaxed,
// std::memory_order_seq_cst, std::memory_order_acquire,
// std::memory_order_release, and std::memory_order_acq_rel
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
template <typename T>
KOKKOS_INTERNAL_FORCEINLINE
T atomic_exchange( T* ptr
               , typename std::remove_cv<T>::type val
               , const std::memory_order order
               ) noexcept
{
  KOKKOS_INTERNAL_CHECK_VALID_ATOMIC_TYPE( T );
  T tmp = *ptr;
  *ptr = val;
  return tmp;
}
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------


//------------------------------------------------------------------------------
// bool atomic_compare_exchange_weak(T* ptr, T& expected, T desired
//                                  , std::memory_order success_memorder
//                                  , std::memory_order failure_memorder
//                                  ) noexcept
// bool atomic_compare_exchange_strong(T* ptr, T& expected, T desired
//                                    , std::memory_order success_memorder
//                                    , std::memory_order failure_memorder
//                                    ) noexcept
//
// Implements an atomic compare and exchange operation.  This compares the
// contents of *ptr with the contents of expected. If equal, the operation is a
// read-modify-write operation that writes desired into *ptr. If they are not
// equal, the operation is a read and the current contents of *ptr are written
// into expected.
//
// Weak compare_exchange may fail spuriously, and the strong variation never
// fails spuriously.  When in doubt, use the strong variation.
//
// If desired is written into *ptr then true is returned and memory is affected
// according to the memory order specified by success_memorder. There are no
// restrictions on what memory order can be used here.
//
// Otherwise, false is returned and memory is affected according to
// failure_memorder. This memory order cannot be std::memory_order_release nor
// std::memory_order_acq_rel. It also cannot be a stronger order than that
// specified by success_memorder.
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
template <typename T>
KOKKOS_INTERNAL_FORCEINLINE
bool atomic_compare_exchange_weak( T* ptr
                            , typename std::remove_cv<T>::type& expected
                            , typename std::remove_cv<T>::type  desired
                            , const std::memory_order success
                            , const std::memory_order failure
                            ) noexcept
{
  KOKKOS_INTERNAL_CHECK_VALID_ATOMIC_TYPE( T );
  T tmp = *ptr;
  const bool result = tmp == expected;
  if (result) {
    *ptr = desired;
  }
  else {
    expected = tmp;
  }

  return result;
}

//------------------------------------------------------------------------------

template <typename T>
KOKKOS_INTERNAL_FORCEINLINE
bool atomic_compare_exchange_strong( T* ptr
                                   , typename std::remove_cv<T>::type& expected
                                   , typename std::remove_cv<T>::type  desired
                                   , const std::memory_order success
                                   , const std::memory_order failure
                                   ) noexcept
{
  return atomic_compare_exchange_weak(ptr, expected, desired, success, failure);
}
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------

} // namespace Kokkos


namespace Kokkos {

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
// T atomic_fetch_add(T* ptr, T val, std::memory_order order) noexcept
// T atomic_fetch_sub(T* ptr, T val, std::memory_order order) noexcept
// pointer atomic_fetch_add(pointer* ptr, std::ptrdiff_t val, std::memory_order order) noexcept
// pointer atomic_fetch_sub(pointer* ptr, std::ptrdiff_t val, std::memory_order order) noexcept
//
// T atomic_fetch_mul(T* ptr, T val, std::memory_order order) noexcept
// T atomic_fetch_div(T* ptr, T val, std::memory_order order) noexcept
//
// T atomic_fetch_min(T* ptr, T val, std::memory_order order) noexcept
// T atomic_fetch_max(T* ptr, T val, std::memory_order order) noexcept
//
// integral atomic_fetch_mod(integral* ptr, integral val, std::memory_order order) noexcept
// integral atomic_fetch_and(integral* ptr, integral val, std::memory_order order) noexcept
// integral atomic_fetch_or(integral* ptr, integral val, std::memory_order order) noexcept
// integral atomic_fetch_xor(integral* ptr, integral val, std::memory_order order) noexcept
// integral atomic_fetch_nand(integral* ptr, integral val, std::memory_order order) noexcept
//
// integral atomic_fetch_lshift(integral* ptr, unsigned val, std::memory_order order) noexcept
// integral atomic_fetch_rshift(integral* ptr, unsigned val, std::memory_order order) noexcept
//
// Perform the operation suggested by the name, and return the value that had
// previously been in *ptr.  All memory orders are valid.
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------

namespace Impl {

template <typename T>
using use_atomic_arithmetic_integral =
  std::integral_constant< bool
                        , !std::is_same<T,bool>::value
                          && (  std::is_integral<T>::value
                             || std::is_enum<T>::value
                             )
                        >;
} // namespace Impl

//------------------------------------------------------------------------------
// fetch_add
//------------------------------------------------------------------------------
template <typename T>
KOKKOS_INTERNAL_FORCEINLINE
T atomic_fetch_add( T* ptr, typename std::remove_cv<T>::type val, std::memory_order order ) noexcept
{
  KOKKOS_INTERNAL_CHECK_VALID_ATOMIC_TYPE( T );
  T tmp = *ptr;
  *ptr = static_cast<T>(*ptr + val);
  return tmp;
}

template <typename T>
KOKKOS_INTERNAL_FORCEINLINE
T* atomic_fetch_add( T** ptr, std::ptrdiff_t val, std::memory_order order ) noexcept
{
  KOKKOS_INTERNAL_CHECK_VALID_ATOMIC_TYPE( T* );
  T tmp = *ptr;
  *ptr = static_cast<T>(*ptr + val);
  return tmp;
}
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// fetch_sub
//------------------------------------------------------------------------------
template <typename T>
KOKKOS_INTERNAL_FORCEINLINE
T atomic_fetch_sub( T* ptr, typename std::remove_cv<T>::type val, std::memory_order order ) noexcept
{
  KOKKOS_INTERNAL_CHECK_VALID_ATOMIC_TYPE( T );
  T tmp = *ptr;
  *ptr = static_cast<T>(*ptr - val);
  return tmp;
}

template <typename T>
KOKKOS_INTERNAL_FORCEINLINE
T* atomic_fetch_sub( T** ptr, std::ptrdiff_t val, std::memory_order order ) noexcept
{
  KOKKOS_INTERNAL_CHECK_VALID_ATOMIC_TYPE( T* );
  T tmp = *ptr;
  *ptr = static_cast<T>(*ptr - val);
  return tmp;
}
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// fetch_mul
//------------------------------------------------------------------------------
template <typename T>
KOKKOS_INTERNAL_FORCEINLINE
T atomic_fetch_mul( T* ptr, typename std::remove_cv<T>::type val, std::memory_order order ) noexcept
{
  KOKKOS_INTERNAL_CHECK_VALID_ATOMIC_TYPE( T );
  T tmp = *ptr;
  *ptr = static_cast<T>(*ptr * val);
  return tmp;
}
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// fetch_div
//------------------------------------------------------------------------------
template <typename T>
KOKKOS_INTERNAL_FORCEINLINE
T atomic_fetch_div( T* ptr, typename std::remove_cv<T>::type val, std::memory_order order ) noexcept
{
  KOKKOS_INTERNAL_CHECK_VALID_ATOMIC_TYPE( T );
  T tmp = *ptr;
  *ptr = static_cast<T>(*ptr / val);
  return tmp;
}
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// fetch_mod
//------------------------------------------------------------------------------
template <typename T>
KOKKOS_INTERNAL_FORCEINLINE
typename std::enable_if< Impl::use_atomic_arithmetic_integral<T>::value, T>::type
atomic_fetch_mod( T* ptr, typename std::remove_cv<T>::type val, std::memory_order order ) noexcept
{
  KOKKOS_INTERNAL_CHECK_VALID_ATOMIC_TYPE( T );
  T tmp = *ptr;
  *ptr = static_cast<T>(*ptr % val);
  return tmp;
}
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// fetch_and
//------------------------------------------------------------------------------
template <typename T>
KOKKOS_INTERNAL_FORCEINLINE
typename std::enable_if< Impl::use_atomic_arithmetic_integral<T>::value, T>::type
atomic_fetch_and( T* ptr, typename std::remove_cv<T>::type val, std::memory_order order ) noexcept
{
  KOKKOS_INTERNAL_CHECK_VALID_ATOMIC_TYPE( T );
  T tmp = *ptr;
  *ptr = static_cast<T>(*ptr & val);
  return tmp;
}
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// fetch_or
//------------------------------------------------------------------------------
template <typename T>
KOKKOS_INTERNAL_FORCEINLINE
typename std::enable_if< Impl::use_atomic_arithmetic_integral<T>::value, T>::type
atomic_fetch_or( T* ptr, typename std::remove_cv<T>::type val, std::memory_order order ) noexcept
{
  KOKKOS_INTERNAL_CHECK_VALID_ATOMIC_TYPE( T );
  T tmp = *ptr;
  *ptr = static_cast<T>(*ptr | val);
  return tmp;
}
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// fetch_xor
//------------------------------------------------------------------------------
template <typename T>
KOKKOS_INTERNAL_FORCEINLINE
typename std::enable_if< Impl::use_atomic_arithmetic_integral<T>::value, T>::type
atomic_fetch_xor( T* ptr, typename std::remove_cv<T>::type val, std::memory_order order ) noexcept
{
  KOKKOS_INTERNAL_CHECK_VALID_ATOMIC_TYPE( T );
  T tmp = *ptr;
  *ptr = static_cast<T>(*ptr ^ val);
  return tmp;
}
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// fetch_nand
//------------------------------------------------------------------------------
template <typename T>
KOKKOS_INTERNAL_FORCEINLINE
typename std::enable_if< Impl::use_atomic_arithmetic_integral<T>::value, T>::type
atomic_fetch_nand( T* ptr, typename std::remove_cv<T>::type val, std::memory_order order ) noexcept
{
  KOKKOS_INTERNAL_CHECK_VALID_ATOMIC_TYPE( T );
  T tmp = *ptr;
  *ptr = static_cast<T>(~(*ptr & val));
  return tmp;
}
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// fetch_lshift
//------------------------------------------------------------------------------
template <typename T>
KOKKOS_INTERNAL_FORCEINLINE
typename std::enable_if< Impl::use_atomic_arithmetic_integral<T>::value, T>::type
atomic_fetch_lshift( T* ptr, unsigned val, std::memory_order order ) noexcept
{
  KOKKOS_INTERNAL_CHECK_VALID_ATOMIC_TYPE( T );
  T tmp = *ptr;
  *ptr = static_cast<T>(*ptr << val);
  return tmp;
}
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// fetch_rshift
//------------------------------------------------------------------------------
template <typename T>
KOKKOS_INTERNAL_FORCEINLINE
typename std::enable_if< Impl::use_atomic_arithmetic_integral<T>::value, T>::type
atomic_fetch_rshift( T* ptr, unsigned val, std::memory_order order ) noexcept
{
  KOKKOS_INTERNAL_CHECK_VALID_ATOMIC_TYPE( T );
  T tmp = *ptr;
  *ptr = static_cast<T>(*ptr >> val);
  return tmp;
}
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
// T atomic_add_fetch(T* ptr, T val, std::memory_order order) noexcept
// T atomic_sub_fetch(T* ptr, T val, std::memory_order order) noexcept
// pointer atomic_add_fetch(pointer* ptr, std::ptrdiff_t val, std::memory_order order) noexcept
// pointer atomic_sub_fetch(pointer* ptr, std::ptrdiff_t val, std::memory_order order) noexcept
//
// T atomic_mul_fetch(T* ptr, T val, std::memory_order order) noexcept
// T atomic_div_fetch(T* ptr, T val, std::memory_order order) noexcept
// T atomic_mod_fetch(T* ptr, T val, std::memory_order order) noexcept
//
// T atomic_min_fetch(T* ptr, T val, std::memory_order order) noexcept
// T atomic_max_fetch(T* ptr, T val, std::memory_order order) noexcept
//
// integral atomic_mod_fetch(integral* ptr, integral val, std::memory_order order) noexcept
// integral atomic_and_fetch(integral* ptr, integral val, std::memory_order order) noexcept
// integral atomic_or_fetch(integral* ptr, integral val, std::memory_order order) noexcept
// integral atomic_xor_fetch(integral* ptr, integral val, std::memory_order order) noexcept
//
// integral atomic_lshift_fetch(integral* ptr, unsigned val, std::memory_order order) noexcept
// integral atomic_rshift_fetch(integral* ptr, unsigned val, std::memory_order order) noexcept
//
// Perform the operation suggested by the name, and return resul. All memory
// orders are valid.
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// add_fetch
//------------------------------------------------------------------------------
template <typename T>
KOKKOS_INTERNAL_FORCEINLINE
T atomic_add_fetch( T* ptr, typename std::remove_cv<T>::type val, std::memory_order order ) noexcept
{
  KOKKOS_INTERNAL_CHECK_VALID_ATOMIC_TYPE( T );
  *ptr = static_cast<T>(*ptr + val);
  return *ptr;
}

template <typename T>
KOKKOS_INTERNAL_FORCEINLINE
T* atomic_add_fetch( T** ptr, std::ptrdiff_t val, std::memory_order order ) noexcept
{
  KOKKOS_INTERNAL_CHECK_VALID_ATOMIC_TYPE( T* );
  *ptr = static_cast<T>(*ptr + val);
  return *ptr;
}
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// sub_fetch
//------------------------------------------------------------------------------
template <typename T>
KOKKOS_INTERNAL_FORCEINLINE
T atomic_sub_fetch( T* ptr, typename std::remove_cv<T>::type val, std::memory_order order ) noexcept
{
  KOKKOS_INTERNAL_CHECK_VALID_ATOMIC_TYPE( T );
  *ptr = static_cast<T>(*ptr - val);
  return *ptr;
}

template <typename T>
KOKKOS_INTERNAL_FORCEINLINE
T* atomic_sub_fetch( T** ptr, std::ptrdiff_t val, std::memory_order order ) noexcept
{
  KOKKOS_INTERNAL_CHECK_VALID_ATOMIC_TYPE( T* );
  *ptr = static_cast<T>(*ptr - val);
  return *ptr;
}
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// mul_fetch
//------------------------------------------------------------------------------
template <typename T>
KOKKOS_INTERNAL_FORCEINLINE
T atomic_mul_fetch( T* ptr, typename std::remove_cv<T>::type val, std::memory_order order ) noexcept
{
  KOKKOS_INTERNAL_CHECK_VALID_ATOMIC_TYPE( T );
  *ptr = static_cast<T>(*ptr * val);
  return *ptr;
}
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// div_fetch
//------------------------------------------------------------------------------
template <typename T>
KOKKOS_INTERNAL_FORCEINLINE
T atomic_div_fetch( T* ptr, typename std::remove_cv<T>::type val, std::memory_order order ) noexcept
{
  KOKKOS_INTERNAL_CHECK_VALID_ATOMIC_TYPE( T );
  *ptr = static_cast<T>(*ptr / val);
  return *ptr;
}
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// mod_fetch
//------------------------------------------------------------------------------
template <typename T>
KOKKOS_INTERNAL_FORCEINLINE
typename std::enable_if< Impl::use_atomic_arithmetic_integral<T>::value, T>::type
atomic_mod_fetch( T* ptr, typename std::remove_cv<T>::type val, std::memory_order order ) noexcept
{
  KOKKOS_INTERNAL_CHECK_VALID_ATOMIC_TYPE( T );
  *ptr = static_cast<T>(*ptr % val);
  return *ptr;
}
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// and_fetch
//------------------------------------------------------------------------------
template <typename T>
KOKKOS_INTERNAL_FORCEINLINE
typename std::enable_if< Impl::use_atomic_arithmetic_integral<T>::value, T>::type
atomic_and_fetch( T* ptr, typename std::remove_cv<T>::type val, std::memory_order order ) noexcept
{
  KOKKOS_INTERNAL_CHECK_VALID_ATOMIC_TYPE( T );
  *ptr = static_cast<T>(*ptr & val);
  return *ptr;
}
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// or_fetch
//------------------------------------------------------------------------------
template <typename T>
KOKKOS_INTERNAL_FORCEINLINE
typename std::enable_if< Impl::use_atomic_arithmetic_integral<T>::value, T>::type
atomic_or_fetch( T* ptr, typename std::remove_cv<T>::type val, std::memory_order order ) noexcept
{
  KOKKOS_INTERNAL_CHECK_VALID_ATOMIC_TYPE( T );
  *ptr = static_cast<T>(*ptr | val);
  return *ptr;
}
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// xor_fetch
//------------------------------------------------------------------------------
template <typename T>
KOKKOS_INTERNAL_FORCEINLINE
typename std::enable_if< Impl::use_atomic_arithmetic_integral<T>::value, T>::type
atomic_xor_fetch( T* ptr, typename std::remove_cv<T>::type val, std::memory_order order ) noexcept
{
  KOKKOS_INTERNAL_CHECK_VALID_ATOMIC_TYPE( T );
  *ptr = static_cast<T>(*ptr ^ val);
  return *ptr;
}
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// nand_fetch
//------------------------------------------------------------------------------
template <typename T>
KOKKOS_INTERNAL_FORCEINLINE
typename std::enable_if< Impl::use_atomic_arithmetic_integral<T>::value, T>::type
atomic_nand_fetch( T* ptr, typename std::remove_cv<T>::type val, std::memory_order order ) noexcept
{
  KOKKOS_INTERNAL_CHECK_VALID_ATOMIC_TYPE( T );
  *ptr = static_cast<T>(~(*ptr & val));
  return *ptr;
}
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// lshift_fetch
//------------------------------------------------------------------------------
template <typename T>
KOKKOS_INTERNAL_FORCEINLINE
typename std::enable_if< Impl::use_atomic_arithmetic_integral<T>::value, T>::type
atomic_lshift_fetch( T* ptr, unsigned val, std::memory_order order ) noexcept
{
  KOKKOS_INTERNAL_CHECK_VALID_ATOMIC_TYPE( T );
  *ptr = static_cast<T>(*ptr << val);
  return *ptr;
}
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// rshift_fetch
//------------------------------------------------------------------------------
template <typename T>
KOKKOS_INTERNAL_FORCEINLINE
typename std::enable_if< Impl::use_atomic_arithmetic_integral<T>::value, T>::type
atomic_rshift_fetch( T* ptr, unsigned val, std::memory_order order ) noexcept
{
  KOKKOS_INTERNAL_CHECK_VALID_ATOMIC_TYPE( T );
  *ptr = static_cast<T>(*ptr >> val);
  return *ptr;
}
//------------------------------------------------------------------------------


} // namespace Kokkos

#undef KOKKOS_INTERNAL_CHECK_VALID_ATOMIC_TYPE
#undef KOKKOS_INTERNAL_FORCEINLINE

#endif // KOKKOS_ATOMIC_SERIAL_HPP

