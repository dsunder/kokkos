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

#ifndef KOKKOS_THREAD_EXECUTION_RESOURCE_HPP
#define KOKKOS_THREAD_EXECUTION_RESOURCE_HPP

#include <iosfwd>

namespace Kokkos { namespace Impl {

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------

class ThreadExecutionResource
{
public:
  //----------------------------------------------------------------------------

  static ThreadExecutionResource root() noexcept;

  static int num_leaves() noexcept;

  static ThreadExecutionResource leaf( int i ) noexcept;

  //----------------------------------------------------------------------------

  explicit operator bool() const noexcept
  { return m_pimpl != nullptr; }

  friend bool operator==( ThreadExecutionResource const & a, ThreadExecutionResource const & b ) noexcept
  { return a.get_impl() == b.get_impl(); }

  friend bool operator!=( ThreadExecutionResource const & a, ThreadExecutionResource const & b ) noexcept
  { return a.get_impl() != b.get_impl(); }

  //----------------------------------------------------------------------------

  int global_id() const noexcept;

  int concurrency() const noexcept;

  ThreadExecutionResource member_of() const noexcept;

  int num_partitions() const noexcept;

  ThreadExecutionResource partition(int i) const noexcept;

  //----------------------------------------------------------------------------

  ThreadExecutionResource()                                               noexcept = default;
  ThreadExecutionResource( ThreadExecutionResource const &  )             noexcept = default;
  ThreadExecutionResource( ThreadExecutionResource       && )             noexcept = default;
  ThreadExecutionResource & operator=( ThreadExecutionResource const &  ) noexcept = default;
  ThreadExecutionResource & operator=( ThreadExecutionResource       && ) noexcept = default;

  //----------------------------------------------------------------------------

  struct Impl;

  explicit ThreadExecutionResource( Impl const * p ) noexcept
    : m_pimpl{ p }
  {}

  const Impl * get_impl() const noexcept { return m_pimpl; }

  //----------------------------------------------------------------------------
private:

  Impl const * m_pimpl{nullptr};
};

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------

bool this_thread_set_binding( const ThreadExecutionResource ) noexcept;
ThreadExecutionResource this_thread_get_binding() noexcept;
ThreadExecutionResource this_thread_get_resource() noexcept;

std::ostream & operator <<( std::ostream & out, const ThreadExecutionResource );

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------

// Deprecated APIs
namespace Kokkos { namespace hwloc {

#if defined( KOKKOS_ENABLE_HWLOC )
constexpr bool available() { return true; }
#else
constexpr bool available() { return false; }
#endif

// These functions assume symmetry and will return incorrect results for
// non-symmetric process bindings
unsigned get_available_numa_count()       noexcept;
unsigned get_available_cores_per_numa()   noexcept;
unsigned get_available_threads_per_core() noexcept;

}} // namespace Kokkos::hwloc

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------

}} // namespace Kokkos::Impl

#endif // KOKKOS_THREAD_EXECUTION_RESOURCE_HPP
