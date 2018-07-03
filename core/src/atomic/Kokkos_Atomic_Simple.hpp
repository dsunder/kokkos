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

#ifndef KOKKOS_ATOMIC_SIMPLE_HPP
#define KOKKOS_ATOMIC_SIMPLE_HPP

#include <Kokkos_Macros.hpp>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <atomic>


#define KOKKOS_INTERNAL_DEFAULT_MEMORY_ORDER std::memory_order_relaxed

namespace Kokkos {



//------------------------------------------------------------------------------
// T atomic_load(T* ptr) noexcept
//
// Returns the contents of *ptr
//------------------------------------------------------------------------------
template <typename T>
KOKKOS_FORCEINLINE_FUNCTION
T atomic_load( T* ptr ) noexcept
{
  return atomic_load( ptr, KOKKOS_INTERNAL_DEFAULT_MEMORY_ORDER );
}
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// void atomic_store(T* ptr, T val) noexcept
//
// Writes val into *ptr
//------------------------------------------------------------------------------
template <typename T>
KOKKOS_FORCEINLINE_FUNCTION
void atomic_store( T* ptr
                 , typename std::remove_cv<T>::type val
                 ) noexcept
{
  atomic_store( ptr, val, KOKKOS_INTERNAL_DEFAULT_MEMORY_ORDER );
}
//------------------------------------------------------------------------------


//------------------------------------------------------------------------------
// T atomic_exchange(T* ptr, T val) noexcept
//
// Writes val into *ptr and returns the previous contents of *ptr.
//------------------------------------------------------------------------------
template <typename T>
KOKKOS_FORCEINLINE_FUNCTION
T atomic_exchange( T* ptr
                 , typename std::remove_cv<T>::type val
                 ) noexcept
{
  return atomic_exchange( ptr, val, KOKKOS_INTERNAL_DEFAULT_MEMORY_ORDER );
}
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// bool atomic_compare_exchange_weak(T* ptr, T& expected, T desired) noexcept
//
// Implements an atomic compare and exchange operation.  This compares the
// contents of *ptr with the contents of expected. If equal, the operation is a
// read-modify-write operation that writes desired into *ptr. If they are not
// equal, the operation is a read and the current contents of *ptr are written
// into expected.
//
// Weak compare_exchange may fail spuriously.
//------------------------------------------------------------------------------
template <typename T>
KOKKOS_FORCEINLINE_FUNCTION
bool atomic_compare_exchange_weak( T* ptr
                                 , typename std::remove_cv<T>::type& expected
                                 , typename std::remove_cv<T>::type  desired
                                 ) noexcept
{
  return atomic_compare_exchange_weak( ptr
                                     , expected
                                     , desired
                                     , KOKKOS_INTERNAL_DEFAULT_MEMORY_ORDER
                                     , KOKKOS_INTERNAL_DEFAULT_MEMORY_ORDER
                                     );
}
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// bool atomic_compare_exchange_strong(T* ptr, T& expected, T desired) noexcept
//
// Implements an atomic compare and exchange operation.  This compares the
// contents of *ptr with the contents of expected. If equal, the operation is a
// read-modify-write operation that writes desired into *ptr. If they are not
// equal, the operation is a read and the current contents of *ptr are written
// into expected.
//
// The strong variation never fails spuriously.  When in doubt, use the strong
// variation.
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
template <typename T>
KOKKOS_FORCEINLINE_FUNCTION
bool atomic_compare_exchange_strong( T* ptr
                                   , typename std::remove_cv<T>::type& expected
                                   , typename std::remove_cv<T>::type  desired
                                   ) noexcept
{
  return atomic_compare_exchange_strong( ptr
                                       , expected
                                       , desired
                                       , KOKKOS_INTERNAL_DEFAULT_MEMORY_ORDER
                                       , KOKKOS_INTERNAL_DEFAULT_MEMORY_ORDER
                                       );
}
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
// T atomic_fetch_add(T* ptr, T val) noexcept
// T atomic_fetch_sub(T* ptr, T val) noexcept
//
// T atomic_fetch_mul(T* ptr, T val) noexcept
// T atomic_fetch_div(T* ptr, T val) noexcept
//
// T atomic_fetch_min(T* ptr, T val) noexcept
// T atomic_fetch_max(T* ptr, T val) noexcept
//
// integral atomic_fetch_mod(integral* ptr, integral val) noexcept
// integral atomic_fetch_and(integral* ptr, integral val) noexcept
// integral atomic_fetch_or(integral* ptr, integral val) noexcept
// integral atomic_fetch_xor(integral* ptr, integral val) noexcept
// integral atomic_fetch_nand(integral* ptr, integral val) noexcept
//
// integral atomic_fetch_lshift(integral* ptr, unsigned val) noexcept
// integral atomic_fetch_rshift(integral* ptr, unsigned val) noexcept
//
// Perform the operation suggested by the name, and return the value that had
// previously been in *ptr.
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
template <typename T>
KOKKOS_FORCEINLINE_FUNCTION
T atomic_fetch_add( T* ptr, typename std::remove_cv<T>::type val ) noexcept
{
  return atomic_fetch_add( ptr, val, KOKKOS_INTERNAL_DEFAULT_MEMORY_ORDER );
}
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
template <typename T>
KOKKOS_FORCEINLINE_FUNCTION
T atomic_fetch_sub( T* ptr, typename std::remove_cv<T>::type val ) noexcept
{
  return atomic_fetch_sub( ptr, val, KOKKOS_INTERNAL_DEFAULT_MEMORY_ORDER );
}
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
template <typename T>
KOKKOS_FORCEINLINE_FUNCTION
T atomic_fetch_mul( T* ptr, typename std::remove_cv<T>::type val ) noexcept
{
  return atomic_fetch_mul( ptr, val, KOKKOS_INTERNAL_DEFAULT_MEMORY_ORDER );
}
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
template <typename T>
KOKKOS_FORCEINLINE_FUNCTION
T atomic_fetch_div( T* ptr, typename std::remove_cv<T>::type val ) noexcept
{
  return atomic_fetch_div( ptr, val, KOKKOS_INTERNAL_DEFAULT_MEMORY_ORDER );
}
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
template <typename T>
KOKKOS_FORCEINLINE_FUNCTION
T atomic_fetch_mod( T* ptr, typename std::remove_cv<T>::type val ) noexcept
{
  return atomic_fetch_mod( ptr, val, KOKKOS_INTERNAL_DEFAULT_MEMORY_ORDER );
}
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
template <typename T>
KOKKOS_FORCEINLINE_FUNCTION
T atomic_fetch_and( T* ptr, typename std::remove_cv<T>::type val ) noexcept
{
  return atomic_fetch_and( ptr, val, KOKKOS_INTERNAL_DEFAULT_MEMORY_ORDER );
}
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
template <typename T>
KOKKOS_FORCEINLINE_FUNCTION
T atomic_fetch_or( T* ptr, typename std::remove_cv<T>::type val ) noexcept
{
  return atomic_fetch_or( ptr, val, KOKKOS_INTERNAL_DEFAULT_MEMORY_ORDER );
}
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
template <typename T>
KOKKOS_FORCEINLINE_FUNCTION
T atomic_fetch_xor( T* ptr, typename std::remove_cv<T>::type val ) noexcept
{
  return atomic_fetch_xor( ptr, val, KOKKOS_INTERNAL_DEFAULT_MEMORY_ORDER );
}
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
template <typename T>
KOKKOS_FORCEINLINE_FUNCTION
T atomic_fetch_nand( T* ptr, typename std::remove_cv<T>::type val ) noexcept
{
  return atomic_fetch_nand( ptr, val, KOKKOS_INTERNAL_DEFAULT_MEMORY_ORDER );
}
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
template <typename T>
KOKKOS_FORCEINLINE_FUNCTION
T atomic_fetch_lshift( T* ptr, unsigned val ) noexcept
{
  return atomic_fetch_lshift( ptr, val, KOKKOS_INTERNAL_DEFAULT_MEMORY_ORDER );
}
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
template <typename T>
KOKKOS_FORCEINLINE_FUNCTION
T atomic_fetch_rshift( T* ptr, unsigned val ) noexcept
{
  return atomic_fetch_rshift( ptr, val, KOKKOS_INTERNAL_DEFAULT_MEMORY_ORDER );
}
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
// integral atomic_mod_fetch(integral* ptr, integral val, std::memory_order order) noexcept
// integral atomic_and_fetch(integral* ptr, integral val, std::memory_order order) noexcept
// integral atomic_or_fetch(integral* ptr, integral val, std::memory_order order) noexcept
// integral atomic_xor_fetch(integral* ptr, integral val, std::memory_order order) noexcept
// integral atomic_nand_fetch(integral* ptr, integral val, std::memory_order order) noexcept
//
// integral atomic_lshift_fetch(integral* ptr, unsigned val, std::memory_order order) noexcept
// integral atomic_rshift_fetch(integral* ptr, unsigned val, std::memory_order order) noexcept
//
// T atomic_add_fetch(T* ptr, T val) noexcept
// T atomic_sub_fetch(T* ptr, T val) noexcept
//
// T atomic_mul_fetch(T* ptr, T val) noexcept
// T atomic_div_fetch(T* ptr, T val) noexcept
//
// T atomic_min_fetch(T* ptr, T val) noexcept
// T atomic_max_fetch(T* ptr, T val) noexcept
//
// integral atomic_mod_fetch(integral* ptr, integral val) noexcept
// integral atomic_and_fetch(integral* ptr, integral val) noexcept
// integral atomic_or_fetch(integral* ptr, integral val) noexcept
// integral atomic_xor_fetch(integral* ptr, integral val) noexcept
// integral atomic_nand_fetch(integral* ptr, integral val) noexcept
//
// integral atomic_lshift_fetch(integral* ptr, unsigned val) noexcept
// integral atomic_rshift_fetch(integral* ptr, unsigned val) noexcept
//
// Perform the operation suggested by the name, and return the result.
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
template <typename T>
KOKKOS_FORCEINLINE_FUNCTION
T atomic_add_fetch( T* ptr, typename std::remove_cv<T>::type val ) noexcept
{
  return atomic_add_fetch( ptr, val, KOKKOS_INTERNAL_DEFAULT_MEMORY_ORDER );
}
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
template <typename T>
KOKKOS_FORCEINLINE_FUNCTION
T atomic_sub_fetch( T* ptr, typename std::remove_cv<T>::type val ) noexcept
{
  return atomic_sub_fetch( ptr, val, KOKKOS_INTERNAL_DEFAULT_MEMORY_ORDER );
}
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
template <typename T>
KOKKOS_FORCEINLINE_FUNCTION
T atomic_mul_fetch( T* ptr, typename std::remove_cv<T>::type val ) noexcept
{
  return atomic_mul_fetch( ptr, val, KOKKOS_INTERNAL_DEFAULT_MEMORY_ORDER );
}
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
template <typename T>
KOKKOS_FORCEINLINE_FUNCTION
T atomic_div_fetch( T* ptr, typename std::remove_cv<T>::type val ) noexcept
{
  return atomic_div_fetch( ptr, val, KOKKOS_INTERNAL_DEFAULT_MEMORY_ORDER );
}
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
template <typename T>
KOKKOS_FORCEINLINE_FUNCTION
T atomic_mod_fetch( T* ptr, typename std::remove_cv<T>::type val ) noexcept
{
  return atomic_mod_fetch( ptr, val, KOKKOS_INTERNAL_DEFAULT_MEMORY_ORDER );
}
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
template <typename T>
KOKKOS_FORCEINLINE_FUNCTION
T atomic_and_fetch( T* ptr, typename std::remove_cv<T>::type val ) noexcept
{
  return atomic_and_fetch( ptr, val, KOKKOS_INTERNAL_DEFAULT_MEMORY_ORDER );
}
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
template <typename T>
KOKKOS_FORCEINLINE_FUNCTION
T atomic_or_fetch( T* ptr, typename std::remove_cv<T>::type val ) noexcept
{
  return atomic_or_fetch( ptr, val, KOKKOS_INTERNAL_DEFAULT_MEMORY_ORDER );
}
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
template <typename T>
KOKKOS_FORCEINLINE_FUNCTION
T atomic_xor_fetch( T* ptr, typename std::remove_cv<T>::type val ) noexcept
{
  return atomic_xor_fetch( ptr, val, KOKKOS_INTERNAL_DEFAULT_MEMORY_ORDER );
}
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
template <typename T>
KOKKOS_FORCEINLINE_FUNCTION
T atomic_nand_fetch( T* ptr, typename std::remove_cv<T>::type val ) noexcept
{
  return atomic_nand_fetch( ptr, val, KOKKOS_INTERNAL_DEFAULT_MEMORY_ORDER );
}
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
template <typename T>
KOKKOS_FORCEINLINE_FUNCTION
T atomic_lshift_fetch( T* ptr, unsigned val ) noexcept
{
  return atomic_lshift_fetch( ptr, val, KOKKOS_INTERNAL_DEFAULT_MEMORY_ORDER );
}
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
template <typename T>
KOKKOS_FORCEINLINE_FUNCTION
T atomic_rshift_fetch( T* ptr, unsigned val ) noexcept
{
  return atomic_rshift_fetch( ptr, val, KOKKOS_INTERNAL_DEFAULT_MEMORY_ORDER );
}
//------------------------------------------------------------------------------

} // namespace Kokkos

#undef KOKKOS_INTERNAL_DEFAULT_MEMORY_ORDER

#endif //KOKKOS_ATOMIC_SIMPLE_HPP

