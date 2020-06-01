//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_FORALL_SEGMENT_REDUCE_MULTIPLE_HPP__
#define __TEST_FORALL_SEGMENT_REDUCE_MULTIPLE_HPP__

#include "gtest/gtest.h"

#include "../../test-forall-utils.hpp"

TYPED_TEST_SUITE_P(ForallSegmentReduceMultipleTest);
template <typename T>
class ForallSegmentReduceMultipleTest : public ::testing::Test
{
};


//
// Data types for multiple reduction tests
//
using ReduceMultipleDataTypeList = camp::list<int,
                                              float,
                                              double>;

#include "test-forall-segment-reducesum-multiple.hpp"
#include "test-forall-segment-reducemin-multiple.hpp"

REGISTER_TYPED_TEST_SUITE_P(ForallSegmentReduceMultipleTest,
                            ReduceSumMultipleSegmentForall,
                            ReduceMinMultipleSegmentForall);

#endif  // __TEST_FORALL_SEGMENT_REDUCE_MULTIPLE_HPP__
