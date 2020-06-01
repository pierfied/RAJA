//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "tests/test-forall-range-reduce-multiple.hpp"

#include "../test-reducepol.hpp"
#include "../test-forall-execpol.hpp"

// Cartesian product of types for Sequential tests
using SequentialForallRangeReduceMultipleTypes =
  Test< camp::cartesian_product<ReduceMultipleDataTypeList, 
                                HostResourceList, 
                                SequentialForallReduceExecPols,
                                SequentialReducePols>>::Types;

INSTANTIATE_TYPED_TEST_SUITE_P(Sequential,
                               ForallRangeReduceMultipleTest,
                               SequentialForallRangeReduceMultipleTypes);
