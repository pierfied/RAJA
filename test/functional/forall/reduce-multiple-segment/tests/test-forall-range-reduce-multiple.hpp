//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_FORALL_RANGE_REDUCE_MULTIPLE_HPP__
#define __TEST_FORALL_RANGE_REDUCE_MULTIPLE_HPP__

#include "gtest/gtest.h"

#include "../../test-forall-utils.hpp"

TYPED_TEST_SUITE_P(ForallRangeReduceMultipleTest);
template <typename T>
class ForallRangeReduceMultipleTest : public ::testing::Test
{
};


//
// Data types for multiple reduction tests
//
using ReduceMultipleDataTypeList = camp::list<int,
                                              float,
                                              double>;

#include "test-forall-range-reducesum-multiple.hpp"
#include "test-forall-range-reducemin-multiple.hpp"
#include "test-forall-range-reducemax-multiple.hpp"

REGISTER_TYPED_TEST_SUITE_P(ForallRangeReduceMultipleTest,
                            ReduceSumMultipleRangeForall,
                            ReduceMinMultipleRangeForall,
                            ReduceMaxMultipleRangeForall);

#endif  // __TEST_FORALL_RANGE_REDUCE_MULTIPLE_HPP__
