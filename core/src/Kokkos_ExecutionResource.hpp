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

#ifndef KOKKOS_EXECUTION_RESOURCE_HPP
#define KOKKOS_EXECUTION_RESOURCE_HPP

#include <iosfwd>

namespace Kokkos {

// unsafe to use any method (besides is_initialized) before Kokkos::initialize(...)
// and after Kokkos::finalize()
class ExecutionResource
{
public:

  // is the class safe to use?
  static bool is_initialized() noexcept;

  // root of process topology tree
  static ExecutionResource process() noexcept;

  //----------------------------------------------------------------------------

  // distance from the root node
  int depth() const noexcept;

  // absolute id of the resource
  int id() const noexcept;

  // number of concurrent threads of execution the process supports on this resource
  int concurrency() const noexcept;

  // number of leaves from the current node
  int num_leaves()  const noexcept;

  // the i'th leaf of the current node
  // with hwloc sort by locality, otherwise sort on id
  ExecutionResource leaf(int i) const noexcept;

  // Parent of the current node
  ExecutionResource member_of() const noexcept;

  // Children of the current node
  int num_partitions() const noexcept;

  // with hwloc sort by locality, otherwise sort on id
  ExecutionResource partition(int i) const noexcept;

  // is the resource symmetric
  bool is_symmetric() const noexcept;

  // is the resource valid
  explicit operator bool() const noexcept
  { return m_pimpl != nullptr; }

  //
  bool operator==( ExecutionResource const & rhs ) const noexcept
  { return m_pimpl == rhs.m_pimpl; }

  bool operator!=( ExecutionResource const & rhs ) const noexcept
  { return m_pimpl != rhs.m_pimpl; }

  //----------------------------------------------------------------------------

  ExecutionResource()                                         noexcept = default;
  ExecutionResource( ExecutionResource const &  )             noexcept = default;
  ExecutionResource( ExecutionResource       && )             noexcept = default;
  ExecutionResource & operator=( ExecutionResource const &  ) noexcept = default;
  ExecutionResource & operator=( ExecutionResource       && ) noexcept = default;

  //----------------------------------------------------------------------------

  class Pimpl;

  explicit ExecutionResource( Pimpl const * p ) noexcept
    : m_pimpl{ p }
  {}

  // only used in impl/Kokkos_ExecutionResource.cpp
  static Pimpl const * impl_get_pimpl( ExecutionResource r ) noexcept { return r.m_pimpl; }

  //----------------------------------------------------------------------------
private:
  Pimpl const * m_pimpl{nullptr};
};

// Print the ids of ExecutionResource leaves and its concurrency
std::ostream & operator <<( std::ostream & out, const ExecutionResource );

// Can be used by any host backend
// May return an ancestor node (parent, grandparent, etc) of the actual binding
// Will return the process ExecutionResource if unable to detect binding
ExecutionResource this_thread_get_binding() noexcept;

// Can be used by any host backend
//
// May return an ancestor node (parent, grandparent, etc) of the actual binding
//
// Will return the process ExecutionResource if unable to detect binding
//
// If hwloc is enabled and hwloc is able to detect thread bindings it will return
// an ExecutionResourse that represents a pu that the thread ran on at sometime
// during this call (though the OS may have moved it since)
ExecutionResource this_thread_get_resource() noexcept;

} // namespace Kokkos

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------

namespace Kokkos { namespace Impl {

// called by Kokkos::initialize()
void initialize_execution_resources() noexcept;

// called by Kokkos::finalize()
void finalize_execution_resources()   noexcept;

//------------------------------------------------------------------------------

// Used in the pthread and stdthread backends
constexpr bool this_thread_can_bind()
{
  #if    !defined(_OPENMP)   \
      && !defined(__APPLE__) \
      && ( defined( KOKKOS_ENABLE_HWLOC ) || defined( _GNU_SOURCE ) )
    return true;
  #else
    return false;
  #endif
}

// Used in the pthread and stdthread backends
// will return false if cannot bind for any reason
bool this_thread_set_binding( const ExecutionResource ) noexcept;

}} // namespace Kokkos::Impl

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------


//------------------------------------------------------------------------------
//------------------------------------------------------------------------------

// Deprecated APIs
namespace Kokkos { namespace hwloc {

// is hwloc available
#if defined( KOKKOS_ENABLE_HWLOC )
constexpr bool available() { return true; }
#else
constexpr bool available() { return false; }
#endif

constexpr bool can_bind_threads() { return Kokkos::Impl::this_thread_can_bind(); }

inline
unsigned get_available_threads_per_core() noexcept
{
  const ExecutionResource process = ExecutionResource::process();
  const ExecutionResource leaf    = process.leaf(0);

  if (process.is_symmetric() && leaf.depth() >= 2) {
    return leaf.member_of().concurrency();
  }

  return 1;
}

inline
unsigned get_available_cores_per_numa() noexcept
{
  const ExecutionResource process = ExecutionResource::process();
  const ExecutionResource leaf    = process.leaf(0);

  if (process.is_symmetric() && leaf.depth() >= 2) {
    return leaf.member_of().num_partitions();
  }

  return process.concurrency();
}

inline
unsigned get_available_numa_count() noexcept
{
  const ExecutionResource process = ExecutionResource::process();
  const ExecutionResource leaf    = process.leaf(0);

  if (process.is_symmetric() && leaf.depth() > 2) {
    return leaf.member_of().member_of().member_of().num_partitions();
  }

  return 1;
}

}} // namespace Kokkos::hwloc

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
#endif // KOKKOS_EXECUTION_RESOURCE_HPP
