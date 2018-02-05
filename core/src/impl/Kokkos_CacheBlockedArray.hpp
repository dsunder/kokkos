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

#ifndef KOKKOS_CACHE_BLOCKED_ARRAY_HPP
#define KOKKOS_CACHE_BLOCKED_ARRAY_HPP

#include <Kokkos_Macros.hpp>
#include <Kokkos_Core_fwd.hpp>
#include <Kokkos_HostSpace.hpp>

#include <type_traits>
#include <cstdint>
#include <string>

namespace Kokkos {
namespace Impl {

template <typename T, typename MemorySpace = HostSpace, int64_t CacheBytes = 64>
class CacheBlockedArray
{
  static_assert( CacheBytes % 16 == 0, "Error: CacheBytes must be a multiple of 16");

  using allocator = SharedAllocationRecord< MemorySpace, void >;

public:
  using value_type      = T;
  using reference       = T&;
  using pointer         = T*;
  using size_type       = int64_t;

  using memory_space    = MemorySpace;

  enum : size_type { cache_bytes  = CacheBytes };
  enum : size_type { stride_bytes = ((sizeof(T) + cache_bytes - 1) / cache_bytes) * cache_bytes };

  reference operator[](const size_type i) const noexcept
  {
    return *reinterpret_cast<pointer>(m_buffer + i*stride_bytes);
  }

  size_type size() const noexcept
  {
    return m_size;
  }

  CacheBlockedArray() = default;

  CacheBlockedArray( const std::string & label, const size_type n )
    : CacheBlockedArray()
  {
    if ( n <= 0 ) return;

    m_buffer = reinterpret_cast<char*>( allocator::allocate_tracked( memory_space{}, label, n*stride_bytes ) );
    m_size   = n;

    for (size_type i=0; i<n; ++i) {
      new (m_buffer + i*stride_bytes) T{};
    }
  }

  CacheBlockedArray( const size_type n )
    : CacheBlockedArray( "unnamed CacheBlockArray", n )
  {}


  ~CacheBlockedArray() noexcept
  {
    if (m_buffer) {
      const size_type n = m_size;
      for (size_type i=0; i<n; ++i) {
        reinterpret_cast<pointer>(m_buffer + i*stride_bytes)->~T();
      }

      allocator::deallocate_tracked( m_buffer );

      m_buffer = nullptr;
      m_size   = 0;
    }
  }

  CacheBlockedArray( CacheBlockedArray && rhs ) noexcept
    : m_buffer{ rhs.m_buffer }
    , m_size{ rhs.m_size }
  {
    rhs.m_buffer = nullptr;
    rhs.m_size   = 0;
  }


  CacheBlockedArray & operator=( CacheBlockedArray && rhs )
  {
    char *    tmp_buffer = m_buffer;
    size_type tmp_size   = m_size;

    m_buffer = rhs.m_buffer;
    m_size   = rhs.m_size;

    rhs.m_buffer = tmp_buffer;
    rhs.m_size   = tmp_size;
  }

  CacheBlockedArray( const CacheBlockedArray & rhs )
    : CacheBlockedArray( rhs.m_size )
  {
    for (int i=0; i<m_size; ++i) {
      (*this)[i] = rhs[i];
    }
  }

  CacheBlockedArray & operator=( const CacheBlockedArray & rhs )
  {
    if ( this != rhs ) {
      this->~CacheBlockedArray();

      CacheBlockedArray tmp( rhs );

      m_buffer = tmp.m_buffer;
      m_size   = tmp.m_size;

      tmp.m_buffer = nullptr;
      tmp.m_size   = 0;
    }
  }

private:
  char *    m_buffer  {nullptr};
  size_type m_size    {0};
};

}} // namespace Kokkos::Impl

#endif //KOKKOS_CACHE_BLOCKED_ARRAY_HPP

