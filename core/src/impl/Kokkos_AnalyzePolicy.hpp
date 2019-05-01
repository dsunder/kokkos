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

#ifndef KOKKOS_IMPL_ANALYZE_POLICY_HPP
#define KOKKOS_IMPL_ANALYZE_POLICY_HPP

#include <Kokkos_Core_fwd.hpp>
#include <Kokkos_Concepts.hpp>
#include <impl/Kokkos_Tags.hpp>

namespace Kokkos { namespace Impl {

template < typename ExecutionSpace   = void
         , typename Schedule         = void
         , typename WorkTag          = void
         , typename IndexType        = void
         , typename IterationPattern = void
         , typename LaunchBounds     = void
         , typename MyWorkItemProperty = Kokkos::Experimental::WorkItemProperty::None_t
         >
struct PolicyTraitsBase
{
  using type = PolicyTraitsBase< ExecutionSpace, Schedule, WorkTag, IndexType,
               IterationPattern, LaunchBounds, MyWorkItemProperty>;

  using execution_space   = ExecutionSpace;
  using schedule_type     = Schedule;
  using work_tag          = WorkTag;
  using index_type        = IndexType;
  using iteration_pattern = IterationPattern;
  using launch_bounds     = LaunchBounds;
  using work_item_property = MyWorkItemProperty;
};

template <typename PolicyBase, typename Property>
struct SetWorkItemProperty
{
  static_assert( std::is_same<typename PolicyBase::work_item_property,Kokkos::Experimental::WorkItemProperty::None_t>::value
               , "Kokkos Error: More than one work item property given" );
  using type = PolicyTraitsBase< typename PolicyBase::execution_space
                               , typename PolicyBase::schedule_type
                               , typename PolicyBase::work_tag
                               , typename PolicyBase::index_type
                               , typename PolicyBase::iteration_pattern
                               , typename PolicyBase::launch_bounds
                               , Property
                               >;
};

template <typename PolicyBase, typename ExecutionSpace>
struct SetExecutionSpace
{
  static_assert( is_void<typename PolicyBase::execution_space>::value
               , "Kokkos Error: More than one execution space given" );
  using type = PolicyTraitsBase< ExecutionSpace
                               , typename PolicyBase::schedule_type
                               , typename PolicyBase::work_tag
                               , typename PolicyBase::index_type
                               , typename PolicyBase::iteration_pattern
                               , typename PolicyBase::launch_bounds
                               , typename PolicyBase::work_item_property
                               >;
};

template <typename PolicyBase, typename Schedule>
struct SetSchedule
{
  static_assert( is_void<typename PolicyBase::schedule_type>::value
               , "Kokkos Error: More than one schedule type given" );
  using type = PolicyTraitsBase< typename PolicyBase::execution_space
                               , Schedule
                               , typename PolicyBase::work_tag
                               , typename PolicyBase::index_type
                               , typename PolicyBase::iteration_pattern
                               , typename PolicyBase::launch_bounds
                               , typename PolicyBase::work_item_property
                               >;
};

template <typename PolicyBase, typename WorkTag>
struct SetWorkTag
{
  static_assert( is_void<typename PolicyBase::work_tag>::value
               , "Kokkos Error: More than one work tag given" );
  using type = PolicyTraitsBase< typename PolicyBase::execution_space
                               , typename PolicyBase::schedule_type
                               , WorkTag
                               , typename PolicyBase::index_type
                               , typename PolicyBase::iteration_pattern
                               , typename PolicyBase::launch_bounds
                               , typename PolicyBase::work_item_property
                               >;
};

template <typename PolicyBase, typename IndexType>
struct SetIndexType
{
  static_assert( is_void<typename PolicyBase::index_type>::value
               , "Kokkos Error: More than one index type given" );
  using type = PolicyTraitsBase< typename PolicyBase::execution_space
                               , typename PolicyBase::schedule_type
                               , typename PolicyBase::work_tag
                               , IndexType
                               , typename PolicyBase::iteration_pattern
                               , typename PolicyBase::launch_bounds
                               , typename PolicyBase::work_item_property
                               >;
};


template <typename PolicyBase, typename IterationPattern>
struct SetIterationPattern
{
  static_assert( is_void<typename PolicyBase::iteration_pattern>::value
               , "Kokkos Error: More than one iteration_pattern given" );
  using type = PolicyTraitsBase< typename PolicyBase::execution_space
                               , typename PolicyBase::schedule_type
                               , typename PolicyBase::work_tag
                               , typename PolicyBase::index_type
                               , IterationPattern
                               , typename PolicyBase::launch_bounds
                               , typename PolicyBase::work_item_property
                               >;
};


template <typename PolicyBase, typename LaunchBounds>
struct SetLaunchBounds
{
  static_assert( is_void<typename PolicyBase::launch_bounds>::value
               , "Kokkos Error: More than one launch_bounds given" );
  using type = PolicyTraitsBase< typename PolicyBase::execution_space
                               , typename PolicyBase::schedule_type
                               , typename PolicyBase::work_tag
                               , typename PolicyBase::index_type
                               , typename PolicyBase::iteration_pattern
                               , LaunchBounds
                               , typename PolicyBase::work_item_property
                               >;
};

template <typename List> struct FindWorkTag;

template <typename T, typename... Types> struct FindWorkTag<TypeList<T, Types...>> {
  static constexpr bool value = true;
  using type = T;
  using modified_list = TypeList<Types...>;
};

template <> struct FindWorkTag<TypeList<>> {
  static constexpr bool value = false;
  using type = void;
  using modified_list = TypeList<>;
};

template <typename T>
using is_index_or_integral_type = std::integral_constant< bool, is_index_type<T>::value || std::is_integral<T>::value>;

template <typename... Traits>
struct AnalyzePolicy
{
  using find_execution_space   = TypeListFind< DefaultExecutionSpace, is_execution_space, TypeList<Traits...>>;
  using find_schedule_type     = TypeListFind< Schedule< Static >, is_schedule_type, typename find_execution_space::modified_list>;
  using find_index_type        = TypeListFind< typename find_execution_space::type::size_type, is_index_or_integral_type, typename find_schedule_type::modified_list>;
  // TODO: choose a good default for the iteration pattern
  using find_iteration_pattern = TypeListFind< void, is_iteration_pattern, typename find_index_type::modified_list>;
  using find_launch_bounds     = TypeListFind< LaunchBounds<>, is_launch_bounds, typename find_iteration_pattern::modified_list>;

  using find_work_item_property = TypeListFind< Kokkos::Experimental::WorkItemProperty::None_t
                                              , Kokkos::Experimental::is_work_item_property
                                              , typename find_launch_bounds::modified_list>;

  // if there is a Type remaining in the list it is the WorkTag
  using find_work_tag = FindWorkTag<typename find_work_item_property::modified_list>;

  static_assert( TypeListSize<typename find_work_tag::modified_list>::value == 0u
               , "Kokkos Error: Unknown parameters passed to the execution policy." );

  using index_type_impl = typename find_index_type::type;
  using index_type = conditional_t< std::is_integral< index_type_impl >::value
                                  , xType<index_type_impl>
                                  , index_type_impl
                                  >;

  using type = PolicyTraitsBase< typename find_execution_space::type
                               , typename find_schedule_type::type
                               , typename find_work_item_property::type
                               , index_type
                               , typename find_iteration_pattern::type
                               , typename find_launch_bounds::type
                               , typename find_work_item_property::type
                              >;

};


template <typename... Traits>
struct PolicyTraits
  : public AnalyzePolicy<Traits...>::type
{};

}} // namespace Kokkos::Impl


#endif //KOKKOS_IMPL_ANALYZE_POLICY_HPP

