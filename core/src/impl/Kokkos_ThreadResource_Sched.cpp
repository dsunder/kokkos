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

#if defined( KOKKOS_ENABLE_SCHED )

#include <impl/Kokkos_ThreadResource.hpp>

#include <algorithm>
#include <cstdlib>
#include <iostream>

#if defined(_OPENMP)
#include <omp.h>
#endif

#include <sched.h>
#include <unistd.h>

namespace Kokkos { namespace Impl {

struct Sentinel
{
  Sentinel() : initialized{true} {}
  ~Sentinel();
  bool initialized {false};
};


class ThreadResource::Pimpl
{
public:

  static void initialize()
  {
    if (s_size != 0) return;

    const static Sentinel sential;

    CPU_ZERO( &s_process );

    s_size = sysconf( _SC_NPROCESSORS_ONLN );

    if (s_size <= 0 ) {
      s_size = 0;
    }
    else {
      #if defined( _OPENMP )
        int num_procs = omp_get_num_procs();
        #pragma omp parallel num_threads(num_procs)
        {
          cpu_set_t tmp;
          sched_getaffinity( 0
                           , sizeof(cpu_set_t)
                           , &tmp
                           );
          #pragma omp critical
          {
            CPU_OR( &s_process, &s_process, &tmp );
          }
        }
      #else
        sched_getaffinity( 0
                         , sizeof(cpu_set_t)
                         , &s_process
                         );
      #endif
    }

    s_concurrency = CPU_COUNT( &s_process );

    s_nodes = new Pimpl[ s_concurrency + 1 ]{};

    s_nodes[0].m_depth = 0;
    s_nodes[0].m_id    = 0;
    s_nodes[0].m_index = 0;

    for (int i=0, j=1; i<s_size; ++i) {
      if( CPU_ISSET( i , &s_process ) ) {
        s_nodes[j].m_depth = 1;
        s_nodes[j].m_id    = i;
        s_nodes[j].m_index = j;
        ++j;
      }
    }
  }

  static void finalize()
  {
    if (s_size > 0) {
      CPU_ZERO( &s_process );
      s_size = 0;
      s_concurrency = 0;
      delete [] s_nodes;
      s_nodes = nullptr;
    }
  }

  int m_depth {0};
  int m_id    {0};
  int m_index {0};

  static int                     s_size;
  static int                     s_concurrency;
  static cpu_set_t               s_process;
  static ThreadResource::Pimpl * s_nodes;
};

int                     ThreadResource::Pimpl::s_size        {0};
int                     ThreadResource::Pimpl::s_concurrency {0};
int                     ThreadResource::Pimpl::s_process     {};
ThreadResource::Pimpl * ThreadResource::Pimpl::s_nodes       {nullptr};

Sentinel::~Sentinel()
{
  if (initialized) {
    ThreadResource::Pimpl::finalize();
  }
}


//------------------------------------------------------------------------------
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------

std::ostream & operator<<( std::ostream & out, const ThreadResource res )
{
  ThreadResource::initialize();
  const auto p = ThreadResource::impl_get_pimpl(res);
  if ( p->m_depth ) {
    out << "{ root : "  << ThreadResource::Pimpl::s_concurrency << " }";
  }
  else {
    out << "{ " << p->m_id << " : " << ThreadResource::Pimpl::s_concurrency << " }";
  }

  return out;
}

ThreadResource ThreadResource::this_thread_get_binding() noexcept
{
  ThreadResource::initialize();
  cpu_set_t tmp;
  CPU_ZERO( &tmp );
  const int err = sched_getaffinity( 0
                                   , sizeof( cpu_set_t )
                                   , & tmp
                                   );

  const int count = CPU_COUNT( &tmp );
  if ( err || count > 1 || count <= 0 ) {
    return ThreadResource{ ThreadResource::Pimpl::s_nodes };
  }

  int gid = 0;
  for (; gid < ThreadResource::Pimpl::s_size; ++gid) {
    if ( CPU_ISSET( gid, &tmp ) ) break;
  }

  int i = 0;
  for (; i < ThreadResource::Pimpl::s_concurrency; ++i) {
    if ( gid == ThreadResource::Pimpl::s_nodes[i+1].m_id ) break;
  }

  return ThreadResource{ &ThreadResource::Pimpl::s_nodes[i+1] };
}

ThreadResource ThreadResource::this_thread_get_resource() noexcept
{
  ThreadResource::initialize();
  const int gid = sched_getcpu();

  for (; gid < ThreadResource::Pimpl::s_size; ++gid) {
    if ( CPU_ISSET( gid, &tmp ) ) break;
  }

  int i = 0;
  for (; i < ThreadResource::Pimpl::s_concurrency; ++i) {
    if ( gid == ThreadResource::Pimpl::s_nodes[i+1].m_id ) break;
  }

  return ThreadResource{ &ThreadResource::Pimpl::s_nodes[i+1] };
}

bool ThreadResource::this_thread_set_binding( const ThreadResource res ) noexcept
{
  #if !defined( _OPENMP )
  if (res) {
    ThreadResource::initialize();
    const auto p = ThreadResource::impl_get_pimpl(res);
    int err = 0;
    if (p->m_depth == 0) {
      err = sched_setaffinity( 0
                             , sizeof(cpu_set_t)
                             , &ThreadResource::Pimpl::s_process
                             );
    } else {
      cpu_set_t tmp;
      CPU_ZERO( &tmp );
      CPU_SET( p->m_id, &tmp );
      err = sched_setaffinity( 0
                             , sizeof(cpu_set_t)
                             , &tmp
                             );
    }
    return !err;
  }
  #endif
  return false;
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------

void ThreadResource::initialize()
{
  Pimpl::initialize();
  const static Sentinel sential;
}

void ThreadResource::finalize()
{
  Pimpl::finalize();
}


bool ThreadResource::is_initialized() noexcept
{ return Pimpl::s_size > 0; }

ThreadResource ThreadResource::process() noexcept
{ initialize(); return ThreadResource{ Pimpl::s_nodes }; }

int ThreadResource::depth() const noexcept
{ initialize(); return m_pimpl->m_depth; }

int ThreadResource::id() const noexcept
{ initialize(); return m_pimpl->m_id; }

int ThreadResource::concurrency() const noexcept
{
  initialize();
  return  m_pimpl->m_depth == 0
       ?  Pimpl::s_concurrency
       :  1;
}

int ThreadResource::num_leaves() const noexcept
{
  initialize();
  return  m_pimpl->m_depth == 0
       ?  Pimpl::s_concurrency
       :  1;
}

ThreadResource ThreadResource::leaf(int i) const noexcept
{
  initialize();
  return  m_pimpl->m_depth == 0 && i < Pimpl::s_concurrency
       ?  ThreadResource{ Pimpl::s_nodes + i + 1 }
       :  *this;
}

ThreadResource ThreadResource::member_of() const noexcept
{
  initialize();
  return  m_pimpl->m_depth == 0
       ?  ThreadResource{ nullptr }
       :  ThreadResource{ Pimpl::s_nodes }
       ;
}

int ThreadResource::num_partitions() const noexcept
{
  initialize();
  return  m_pimpl->m_depth == 0
       ?  Pimpl::s_concurrency
       :  0
       ;
}

ThreadResource ThreadResource::partition(int i) const noexcept
{
  initialize();
  return  m_pimpl->m_depth == 0 && i < Pimpl::s_concurrency
       ?  ThreadResource{ Pimpl::s_nodes + i }
       :  ThreadResource{ nullptr }
       ;
}

bool ThreadResource::is_symmetric() const noexcept
{ return true; }

}} // namespace Kokkos::Impl

#else

void KOKKOS_CORE_SRC_IMPL_THREAD_RESOURCE_SCHED() {}

#endif // sched.h

