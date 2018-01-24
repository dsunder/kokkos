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

#include <Kokkos_Macros.hpp>

#if  !defined( KOKKOS_ENABLE_HWLOC ) && !defined( KOKKOS_ENABLE_SCHED )


#include <impl/Kokkos_ThreadResource.hpp>

#include <iostream>

#if defined(_OPENMP)
#include <omp.h>
#endif

#if defined( _GNU_SOURCE )
#include <unistd.h>
#endif

namespace Kokkos { namespace Impl {

class ThreadResource::Pimpl
{
public:
  static int s_concurrency;
};

int ThreadResource::Pimpl::s_concurrency {0};

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------

std::ostream & operator<<( std::ostream & out, const ThreadResource )
{
  ThreadResource::initialize();
  out << "{ root : "  << ThreadResource::Pimpl::s_concurrency << " }";

  return out;
}

ThreadResource this_thread_get_bind() noexcept
{
  ThreadResource::initialize();
  return ThreadResource{ reinterpret_cast<ThreadResource::Pimpl*>(1) };
}

ThreadResource this_thread_get_resource() noexcept
{
  ThreadResource::initialize();
  return ThreadResource{ reinterpret_cast<ThreadResource::Pimpl*>(1) };
}

bool this_thread_set_binding( const ThreadResource res ) noexcept
{
  return false;
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------

void ThreadResource::initialize()
{
  if ( Pimpl::s_concurrency != 0 ) return;

  #if defined( _OPENMP )
    int num_procs = omp_get_num_procs();
    #pragma omp parallel num_threads(num_procs)
    {
      #pragma omp atomic
      ++Pimpl::s_concurrency;
    }
  #elif defined( _GNU_SOURCE )
    Pimpl::s_concurrency = sysconf( _SC_NPROCESSORS_ONLN );
  #else
    Pimpl::s_concurrency = 1;
  #endif
}

void ThreadResource::finalize()
{
  Pimpl::s_concurrency = 0;
}


bool ThreadResource::is_initialized() noexcept
{ return Pimpl::s_concurrency > 0; }

ThreadResource ThreadResource::process() noexcept
{ return ThreadResource{ reinterpret_cast<Pimpl*>(1) }; }

int ThreadResource::depth() const noexcept
{ return 0; }

int ThreadResource::id() const noexcept
{ return 0; }

int ThreadResource::concurrency() const noexcept
{
  ThreadResource::initialize();
  return  Pimpl::s_concurrency;
}

int ThreadResource::num_leaves() const noexcept
{
  return  1;
}

ThreadResource ThreadResource::leaf(int i) const noexcept
{
  return  *this;
}

ThreadResource ThreadResource::member_of() const noexcept
{
  return  ThreadResource{ nullptr };
}

int ThreadResource::num_partitions() const noexcept
{
  return  0;
}

ThreadResource ThreadResource::partition(int i) const noexcept
{
  return *this;
}

bool ThreadResource::is_symmetric() const noexcept
{ return true; }

}} // namespace Kokkos::Impl

#else

void KOKKOS_CORE_SRC_IMPL_THREAD_RESOURCE_TRIVIAL() {}

#endif // sched.h

