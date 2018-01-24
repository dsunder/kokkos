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

#ifndef KOKKOS_THREAD_RESOURCE_HPP
#define KOKKOS_THREAD_RESOURCE_HPP

#include <iosfwd>

namespace Kokkos { namespace Impl {

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------

class ThreadResource
{
public:
  //----------------------------------------------------------------------------

  // initialize must be called after main()
  // invalid to use any ThreadResource before calling initialize
  static void initialize();

  // must be called before returning from main to avoid memory leaks
  static void finialize();

  static bool is_initialie();

  // Handle to the detected hardware topology
  // that this process is bound to
  static ThreadResource process() noexcept;

  //----------------------------------------------------------------------------

  explicit operator bool() const noexcept
  { return m_pimpl != nullptr; }

  //----------------------------------------------------------------------------

  // how far down the the topology tree is the resource
  // root and process are at level 0
  int depth() const;

  // id independant of process binding
  int id() const noexcept;

  // number of concurrent threads of execution that can be bound to this resource
  int concurrency() const noexcept;

  // position in the parent's partition span
  int index() const noexcept;

  // Handle to parent resource
  // returns an empty handle for root and process
  ThreadResource member_of() const noexcept;

  // number of direct children resources
  int num_partitions() const noexcept;

  // this i'th child
  ThreadResource partition(int i) const noexcept;

  int num_leaves() const noexcept;

  ThreadResource leaf( int i ) const noexcept;

  // is the given resource symmetric over it's children
  bool is_symmetric() const noexcept;

  //----------------------------------------------------------------------------

  ThreadResource()                                      noexcept = default;
  ThreadResource( ThreadResource const &  )             noexcept = default;
  ThreadResource( ThreadResource       && )             noexcept = default;
  ThreadResource & operator=( ThreadResource const &  ) noexcept = default;
  ThreadResource & operator=( ThreadResource       && ) noexcept = default;

  //----------------------------------------------------------------------------

  class Pimpl;

  explicit ThreadResource( Pimpl const * p ) noexcept
    : m_pimpl{ p }
  {}

  static const Pimpl * impl_get_pimpl( ThreadResource const& r) noexcept
  {
    return r.m_pimpl;
  }

  //----------------------------------------------------------------------------
private:

  Pimpl const * m_pimpl{nullptr};
};

inline
bool operator==( ThreadResource const & a, ThreadResource const & b ) noexcept
{
  return ThreadResource::impl_get_pimpl(a) == ThreadResource::impl_get_pimpl(b);
}

inline bool operator!=( ThreadResource const & a, ThreadResource const & b ) noexcept
{ return !(a==b); }

//------------------------------------------------------------------------------
std::ostream & operator <<( std::ostream & out, const ThreadResource );

//------------------------------------------------------------------------------

// Will return the smallest ThreadResource which completely covers the detected
// binding
ThreadResource this_thread_get_binding() noexcept;
ThreadResource this_thread_get_resource() noexcept;
bool this_thread_set_binding( const ThreadResource ) noexcept;

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


//------------------------------------------------------------------------------


}} // namespace Kokkos::Impl

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------

// Deprecated APIs
namespace Kokkos { namespace hwloc {

#if defined( KOKKOS_ENABLE_HWLOC )
constexpr bool available() { return true; }
#else
constexpr bool available() { return false; }
#endif

constexpr bool can_bind_threads() { return Kokkos::Impl::this_thread_can_bind(); }


inline
unsigned get_available_threads_per_core() noexcept
{
  using ThreadResource = Kokkos::Impl::ThreadResource;

  const ThreadResource process = ThreadResource::process();
  const ThreadResource leaf    = process.leaf(0);

  if (process.is_symmetric() && leaf.depth() >= 2) {
    return leaf.member_of().concurrency();
  }

  return 1;
}

inline
unsigned get_available_cores_per_numa() noexcept
{
  using ThreadResource = Kokkos::Impl::ThreadResource;

  const ThreadResource process = ThreadResource::process();
  const ThreadResource leaf    = process.leaf(0);

  if (process.is_symmetric() && leaf.depth() >= 2) {
    return leaf.member_of().member_of().num_partitions();
  }

  return process.concurrency();
}

inline
unsigned get_available_numa_count() noexcept
{
  using ThreadResource = Kokkos::Impl::ThreadResource;

  const ThreadResource process = ThreadResource::process();
  const ThreadResource leaf    = process.leaf(0);

  if (process.is_symmetric() && leaf.depth() > 2) {
    return leaf.member_of().member_of().member_of().num_partitions();
  }

  return 1;
}

}} // namespace Kokkos::hwloc

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------


#endif // KOKKOS_THREAD_RESOURCE_HPP

