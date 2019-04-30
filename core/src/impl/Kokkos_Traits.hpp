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

#ifndef KOKKOSTRAITS_HPP
#define KOKKOSTRAITS_HPP

#include <cstddef>
#include <cstdint>
#include <Kokkos_Macros.hpp>
#include <impl/Kokkos_BitOps.hpp>
#include <string>
#include <type_traits>

namespace Kokkos {
namespace Impl {

//----------------------------------------------------------------------------
template <unsigned I, typename... Pack>
struct get_type {
  static_assert(sizeof...(Pack) == 0u, "Error: should only match the base case");
  using type = void;
};

template <typename T, typename... Pack>
struct get_type<0u, T, Pack...>
{ using type = T; };

template <unsigned I, typename T, typename... Pack>
struct get_type<I, T, Pack...>
  : public get_type<I-1u, Pack...> {};
//----------------------------------------------------------------------------

//----------------------------------------------------------------------------
template <typename T, typename... Pack>
struct has_type
{
  static_assert(sizeof...(Pack) == 0u, "Error: should only match the base case");
  static constexpr bool value = false;
};

template <typename T, typename... Pack>
struct has_type< T, T, Pack...>
{
  static constexpr bool value = true;
  static_assert( has_type<T, Pack...>::value == false
               , "Error: more than one member of the argument pack matches the type"
               );
};

template <typename T, typename U, typename... Pack>
struct has_type< T, U, Pack...>
  : public has_type <T, Pack...>
{};
//----------------------------------------------------------------------------


//----------------------------------------------------------------------------
template < template <typename> class Condition
         , bool Value
         , typename Type
         , typename... Pack
         >
struct has_condition_impl
{
  static_assert(sizeof...(Pack) == 0u, "Error: should only match the base case");
  static constexpr bool value = Value;
  using type = Type;
};

template < template <typename> class Condition
         , bool Value
         , typename Type
         , typename Head
         , typename... Pack
         >
struct has_condition_impl< Condition, Value, Type, Head, Pack...>
  : public has_condition_impl< Condition
                             , Value || Condition<Head>::value
                             , typename std::conditional< Condition<Head>::value, Head, Type >::type
                             , Pack...
                             >
{
  static_assert( !(Value && Condition<Head>::value)
               , "Error: more than one member of the argument pack satisfies condition"
               );
};

template< typename DefaultType
        , template< typename > class Condition
        , typename ... Pack >
struct has_condition
  : public has_condition_impl< Condition, false, DefaultType, Pack...>
{};
//----------------------------------------------------------------------------

//----------------------------------------------------------------------------
template <bool Value, typename... Types>
struct are_integral_impl {
  static_assert(sizeof...(Types) == 0u, "Error: should only match the base case");
  static constexpr bool value = Value;
};

template <bool Value, typename T, typename... Types>
struct are_integral_impl<Value, T, Types...>
  : public are_integral_impl< Value && (std::is_integral<T>::value || std::is_enum<T>::value)
                            , Types...
                            >
{};

template <typename... Types>
struct are_integral : public are_integral_impl< true, Types...> {};

template <>
struct are_integral<> { static constexpr bool value = false; };

//----------------------------------------------------------------------------
/* C++11 conformal compile-time type traits utilities.
 * Prefer to use C++11 when portably available.
 */
//----------------------------------------------------------------------------
// C++11 Helpers:

template <typename T, T v>
using integral_constant = std::integral_constant<T,v>;

using false_type = std::false_type;
using true_type  = std::true_type;

//----------------------------------------------------------------------------
// C++11 type_traits

template< class T , class U > using is_same = std::is_same<T,U>;

template< class T > using is_const = std::is_const<T>;
template< class T > using is_array = std::is_array<T>;

template< class T > using add_const        = std::add_const<T>;
template< class T > using remove_const     = std::remove_const<T>;
template< class T > using remove_reference = std::remove_reference<T>;

template< class T > using remove_extent = std::remove_extent<T>;

//----------------------------------------------------------------------------
// C++11 Other type generators:

template< bool Cond, class T= void >
using enable_if = std::enable_if<Cond,T>;

//----------------------------------------------------------------------------

} // namespace Impl
} // namespace Kokkos

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
// Other traits

namespace Kokkos {
namespace Impl {

//----------------------------------------------------------------------------

template< class , class T = void >
struct enable_if_type { using type = T; };

//----------------------------------------------------------------------------

//----------------------------------------------------------------------------


template < bool Cond , typename TrueType , typename FalseType>
using conditional = std::conditional< Cond, TrueType, FalseType >;

template < bool Cond , typename TrueType , typename FalseType>
using conditional_t = typename std::conditional< Cond, TrueType, FalseType >::type;

template < bool Cond , typename TrueType , typename FalseType>
struct if_c
{
  enum { value = Cond };

  typedef FalseType type;


  typedef typename remove_const<
          typename remove_reference<type>::type >::type value_type ;

  typedef typename add_const<value_type>::type const_value_type ;

  static KOKKOS_INLINE_FUNCTION
  const_value_type & select( const_value_type & v ) { return v ; }

  static KOKKOS_INLINE_FUNCTION
  value_type & select( value_type & v ) { return v ; }

  template< class T >
  static KOKKOS_INLINE_FUNCTION
  value_type & select( const T & ) { value_type * ptr(0); return *ptr ; }


  template< class T >
  static KOKKOS_INLINE_FUNCTION
  const_value_type & select( const T & , const_value_type & v ) { return v ; }

  template< class T >
  static KOKKOS_INLINE_FUNCTION
  value_type & select( const T & , value_type & v ) { return v ; }
};

template <typename TrueType, typename FalseType>
struct if_c< true , TrueType , FalseType >
{
  enum { value = true };

  typedef TrueType type;


  typedef typename remove_const<
          typename remove_reference<type>::type >::type value_type ;

  typedef typename add_const<value_type>::type const_value_type ;

  static KOKKOS_INLINE_FUNCTION
  const_value_type & select( const_value_type & v ) { return v ; }

  static KOKKOS_INLINE_FUNCTION
  value_type & select( value_type & v ) { return v ; }

  template< class T >
  static KOKKOS_INLINE_FUNCTION
  value_type & select( const T & ) { value_type * ptr(0); return *ptr ; }


  template< class F >
  static KOKKOS_INLINE_FUNCTION
  const_value_type & select( const_value_type & v , const F & ) { return v ; }

  template< class F >
  static KOKKOS_INLINE_FUNCTION
  value_type & select( value_type & v , const F & ) { return v ; }
};

template< typename TrueType >
struct if_c< false , TrueType , void >
{
  enum { value = false };

  typedef void type ;
  typedef void value_type ;
};

template< typename FalseType >
struct if_c< true , void , FalseType >
{
  enum { value = true };

  typedef void type ;
  typedef void value_type ;
};

//----------------------------------------------------------------------------

template <typename T>
using is_integral = std::is_integral<T>;
//----------------------------------------------------------------------------

template<typename T>
struct is_label : public false_type {};

template<>
struct is_label<const char*> : public true_type {};

template<>
struct is_label<char*> : public true_type {};


template<int N>
struct is_label<const char[N]> : public true_type {};

template<int N>
struct is_label<char[N]> : public true_type {};


template<>
struct is_label<const std::string> : public true_type {};

template<>
struct is_label<std::string> : public true_type {};

// These 'constexpr'functions can be used as
// both regular functions and meta-function.

/**\brief  There exists integral 'k' such that N = 2^k */
KOKKOS_INLINE_FUNCTION
constexpr bool is_integral_power_of_two( const size_t N )
{ return ( 0 < N ) && ( 0 == ( N & ( N - 1 ) ) ); }

/**\brief  Return integral 'k' such that N = 2^k, assuming valid.  */
KOKKOS_INLINE_FUNCTION
constexpr unsigned integral_power_of_two_assume_valid( const size_t N )
{ return N == 1 ? 0 : 1 + integral_power_of_two_assume_valid( N >> 1 ); }

/**\brief  Return integral 'k' such that N = 2^k, if exists.
 *         If does not exist return ~0u.
 */
KOKKOS_INLINE_FUNCTION
constexpr unsigned integral_power_of_two( const size_t N )
{ return is_integral_power_of_two(N) ? integral_power_of_two_assume_valid(N) : ~0u ; }

//----------------------------------------------------------------------------

template < size_t N >
struct is_power_of_two
{
  enum type { value = (N > 0) && !(N & (N-1)) };
};

template < size_t N , bool OK = is_power_of_two<N>::value >
struct power_of_two ;

template < size_t N >
struct power_of_two<N,true>
{
  enum type { value = 1+ power_of_two<(N>>1),true>::value };
};

template <>
struct power_of_two<2,true>
{
  enum type { value = 1 };
};

template <>
struct power_of_two<1,true>
{
  enum type { value = 0 };
};

/** \brief  If power of two then return power,
 *          otherwise return ~0u.
 */
KOKKOS_FORCEINLINE_FUNCTION
unsigned power_of_two_if_valid( const unsigned N )
{
  unsigned p = ~0u ;
  if ( is_integral_power_of_two ( N ) ) {
    p = bit_scan_forward ( N ) ;
  }
  return p ;
}

//----------------------------------------------------------------------------

template< typename T , T v , bool NonZero = ( v != T(0) ) >
struct integral_nonzero_constant
{
  // Declaration of 'static const' causes an unresolved linker symbol in debug
  // static const T value = v ;
  enum { value = T(v) };
  typedef T value_type ;
  typedef integral_nonzero_constant<T,v> type ;
  KOKKOS_INLINE_FUNCTION integral_nonzero_constant( const T & ) {}
};

template< typename T , T zero >
struct integral_nonzero_constant<T,zero,false>
{
  const T value ;
  typedef T value_type ;
  typedef integral_nonzero_constant<T,0> type ;
  KOKKOS_INLINE_FUNCTION integral_nonzero_constant( const T & v ) : value(v) {}
};

//----------------------------------------------------------------------------


template <class T>
struct make_all_extents_into_pointers
{
  using type = T;
};

template <class T, unsigned N>
struct make_all_extents_into_pointers<T[N]>
{
  using type = typename make_all_extents_into_pointers<T>::type*;
};

template <class T>
struct make_all_extents_into_pointers<T*>
{
  using type = typename make_all_extents_into_pointers<T>::type*;
};

} // namespace Impl
} // namespace Kokkos

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

#endif /* #ifndef KOKKOSTRAITS_HPP */

