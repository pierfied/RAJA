/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing RAJA index set and segment iteration
 *          template methods for Apollo-guided execution.
 *
 *          These methods should work on any platform.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-18, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-689114
//
// All rights reserved.
//
// This file is part of RAJA.
//
// For details about use and distribution, please read RAJA/LICENSE.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_forall_apollo_HPP
#define RAJA_forall_apollo_HPP

#include "RAJA/config.hpp"

#include <iostream>
#include <type_traits>

#include <string>
#include <sstream>
#include <functional>
#include <unordered_set>

#include <omp.h>

#include "RAJA/util/types.hpp"

#include "RAJA/index/IndexSet.hpp"
#include "RAJA/index/ListSegment.hpp"
#include "RAJA/index/RangeSegment.hpp"

#include "RAJA/pattern/forall.hpp"
#include "RAJA/pattern/region.hpp"

#include "RAJA/policy/apollo/policy.hpp"
#include "RAJA/internal/fault_tolerance.hpp"

#include "apollo/Apollo.h"
#include "apollo/Region.h"


// ----------


namespace RAJA
{
namespace policy
{
namespace apollo
{

template <typename Iterable, typename Func>
RAJA_INLINE void forall_impl(const RAJA::apollo_omp_auto&, int num_threads, Iterable&& iter, Func&& loop_body) {
    RAJA_EXTRACT_BED_IT(iter);
#pragma omp parallel num_threads(num_threads)
    {
        using RAJA::internal::thread_privatize;
        auto body = thread_privatize(loop_body);
#pragma omp for schedule(auto)
        for (decltype(distance_it) i = 0; i < distance_it; ++i) {
            body.get_priv()(begin_it[i]);
        }
    }
}

template <typename Iterable, typename Func>
RAJA_INLINE void forall_impl(const RAJA::apollo_omp_static&, int num_threads, Iterable&& iter, Func&& loop_body) {
    RAJA_EXTRACT_BED_IT(iter);
#pragma omp parallel num_threads(num_threads)
    {
        using RAJA::internal::thread_privatize;
        auto body = thread_privatize(loop_body);
#pragma omp for schedule(static)
        for (decltype(distance_it) i = 0; i < distance_it; ++i) {
            body.get_priv()(begin_it[i]);
        }
    }
}


template <typename Iterable, typename Func>
RAJA_INLINE void forall_impl(const RAJA::apollo_omp_dynamic&, int num_threads, Iterable&& iter, Func&& loop_body) {
    RAJA_EXTRACT_BED_IT(iter);
#pragma omp parallel num_threads(num_threads)
    {
        using RAJA::internal::thread_privatize;
        auto body = thread_privatize(loop_body);
#pragma omp for schedule(dynamic)
        for (decltype(distance_it) i = 0; i < distance_it; ++i) {
            body.get_priv()(begin_it[i]);
        }
    }
}


template <typename Iterable, typename Func>
RAJA_INLINE void forall_impl(const RAJA::apollo_omp_guided&, int num_threads, Iterable&& iter, Func&& loop_body) {
    RAJA_EXTRACT_BED_IT(iter);
#pragma omp parallel num_threads(num_threads)
    {
        using RAJA::internal::thread_privatize;
        auto body = thread_privatize(loop_body);
#pragma omp for schedule(guided)
        for (decltype(distance_it) i = 0; i < distance_it; ++i) {
            body.get_priv()(begin_it[i]);
        }
    }
}



//
//////////////////////////////////////////////////////////////////////
//
// The following function template switches between various RAJA
// execution policies based on feedback from the Apollo system.
//
//////////////////////////////////////////////////////////////////////
//

#ifndef RAJA_ENABLE_OPENMP
#error "*** RAJA_ENABLE_OPENMP is not defined!" \
    "This build of RAJA requires OpenMP to be enabled! ***"
#endif

using apolloPolicySeq      = RAJA::seq_exec;
using apolloPolicySIMD     = RAJA::simd_exec;
using apolloPolicyLoopExec = RAJA::loop_exec;
using apolloPolicyOMPDefault = RAJA::omp_parallel_for_exec;
using apolloPolicyOMPAuto    = RAJA::apollo_omp_auto;
using apolloPolicyOMPStatic  = RAJA::apollo_omp_static;
using apolloPolicyOMPDynamic = RAJA::apollo_omp_dynamic;
using apolloPolicyOMPGuided  = RAJA::apollo_omp_guided;

#define APOLLO_OMP_SET_THREADS(__threads) \
{ \
    Apollo::instance()->setFeature((double) __threads); \
    g_apollo_num_threads = __threads; \
};

template <typename Iterable, typename Func>
RAJA_INLINE void apolloPolicySwitcher(int policy, int tc[], Iterable &&iter, Func &&loop_body, Apollo::Region *apolloRegion) {
    Apollo *apollo             = Apollo::instance();
    switch(policy) {
        case   0: // The 0th policy is always a "safe" choice in Apollo as a
                  // default, or fail-safe when models are broken or partial..
                  // In the case of this OpenMP exploration template, the
                  // 0'th policy uses whatever was already set by the previous
                  // Apollo::Region's model, or the system defaults, if it is
                  // the first loop to get executed.
                  apollo->numThreads = apollo->ompDefaultNumThreads;
                  break;
        case   1: // The 1st policy is a Sequential option, which will come into
                  // play for iterations when the number of elements a loop is
                  // operating over is low enough that the overhead of distrubuting
                  // the tasks to OpenMP is not worth paying. Learning will disrupt
                  // the performance of the application more, when this option is
                  // available, but the learned model will be able to make
                  // more significant performance improvements for applications
                  // with ocassional sparse inputs to loops.
                  {apolloRegion->setFeature(1.0);
                  apollo->numThreads = 1;

                  //body(apolloPolicySeq{});
                  RAJA_EXTRACT_BED_IT(iter);
                  using RAJA::internal::thread_privatize;
                  auto body = thread_privatize(loop_body);
                  for (decltype(distance_it) i = 0; i < distance_it; ++i) {
                      body.get_priv()(begin_it[i]);
                  }
                  return;}
        case   2: apollo->numThreads = tc[0]; break;
        case   3: apollo->numThreads = tc[1]; break;
        case   4: apollo->numThreads = tc[2]; break;
        case   5: apollo->numThreads = tc[3]; break;
        case   6: apollo->numThreads = tc[4]; break;
        case   7: apollo->numThreads = tc[5]; break;
        case   8: apollo->numThreads = tc[0]; break;
        case   9: apollo->numThreads = tc[1]; break;
        case  10: apollo->numThreads = tc[2]; break;
        case  11: apollo->numThreads = tc[3]; break;
        case  12: apollo->numThreads = tc[4]; break;
        case  13: apollo->numThreads = tc[5]; break;
        case  14: apollo->numThreads = tc[0]; break;
        case  15: apollo->numThreads = tc[1]; break;
        case  16: apollo->numThreads = tc[2]; break;
        case  17: apollo->numThreads = tc[3]; break;
        case  18: apollo->numThreads = tc[4]; break;
        case  19: apollo->numThreads = tc[5]; break;
    }

    apolloRegion->setFeature((double) apollo->numThreads);

    switch(policy) {
        case   0:
            forall_impl(apolloPolicyOMPAuto{}, apollo->numThreads, iter, loop_body);
            break;
        case   1:
            // NOTE(chad): case 1 (cpu_seq) has already returned, see above.
            break;
        case   2:
        case   3:
        case   4:
        case   5:
        case   6:
        case   7:
            forall_impl(apolloPolicyOMPStatic{}, apollo->numThreads, iter, loop_body);
            break;
        case   8:
        case   9:
        case  10:
        case  11:
        case  12:
        case  13:
            forall_impl(apolloPolicyOMPDynamic{}, apollo->numThreads, iter, loop_body);
            break;
        case  14:
        case  15:
        case  16:
        case  17:
        case  18:
        case  19:
            forall_impl(apolloPolicyOMPGuided{}, apollo->numThreads, iter, loop_body);
            break;
    }

    return;
}

const int POLICY_COUNT = 20;


template <typename Iterable, typename Func>
RAJA_INLINE void forall_impl(const apollo_exec &, Iterable &&iter, Func &&loop_body)
{
    static Apollo         *apollo             = Apollo::instance();
    static Apollo::Region *apolloRegion       = nullptr;
    static int             policy_index       = 0;
    static int             th_count_opts[6]   = {2, 2, 2, 2, 2, 2};
    if (apolloRegion == nullptr) {
        std::string code_location = apollo->getCallpathOffset();
        apolloRegion = new Apollo::Region(
            1,
            code_location.c_str(),
            RAJA::policy::apollo::POLICY_COUNT);
        // Set the range of thread counts we want to make available for
        // bootstrapping and use by this Apollo::Region.
        th_count_opts[0] = 2;
        th_count_opts[1] = std::min(4,  std::max(2, (int)(apollo->numThreadsPerProcCap * 0.25)));
        th_count_opts[2] = std::min(8,  std::max(2, (int)(apollo->numThreadsPerProcCap * 0.50)));
        th_count_opts[3] = std::min(16, std::max(2, (int)(apollo->numThreadsPerProcCap * 0.75)));
        th_count_opts[4] = std::min(32, std::max(2, apollo->numThreadsPerProcCap));
        th_count_opts[5] = std::max(2, apollo->numThreadsPerProcCap);
	}

    // Count the number of elements.
    double num_elements = 0.0;
    num_elements = (double) std::distance(std::begin(iter), std::end(iter));

    apolloRegion->begin();
    apolloRegion->setFeature(num_elements);

    policy_index = apolloRegion->getPolicyIndex();
    apolloPolicySwitcher(policy_index, th_count_opts, iter, loop_body, apolloRegion);

    apolloRegion->end();
}

//////////
}  // closing brace for apollo namespace
}  // closing brace for policy namespace
}  // closing brace for RAJA namespace

#endif  // closing endif for header file include guard



// Examples of loops using a global instead of lookup for thread count:
//
//namespace RAJA
//{
//namespace policy
//{
//namespace omp
//{


