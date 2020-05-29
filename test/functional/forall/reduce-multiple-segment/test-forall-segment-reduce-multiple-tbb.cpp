//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "tests/test-forall-segment-reduce-multiple.hpp"

#if defined(RAJA_ENABLE_TBB)

#include "../test-forall-execpol.hpp"
#include "../test-reducepol.hpp"

// Cartesian product of types for TBB tests
using TBBForallSegmentReduceMultipleTypes =
  Test< camp::cartesian_product<ReduceMultipleDataTypeList, 
                                HostResourceList, 
                                TBBForallExecPols,
                                TBBReducePols>>::Types;

INSTANTIATE_TYPED_TEST_SUITE_P(TBB,
                               ForallSegmentReduceMultipleTest,
                               TBBForallSegmentReduceMultipleTypes);

#endif
