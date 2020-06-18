//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "test-forall-segment.hpp"

#if defined(RAJA_ENABLE_OPENMP)

// Cartesian product of types for OpenMP tests
using OpenMPForallSegmentTypes =
  Test< camp::cartesian_product<StrongIdxTypeList,
                                HostResourceList,
                                OpenMPForallExecPols> >::Types;

INSTANTIATE_TYPED_TEST_SUITE_P(OpenMP,
                               ForallSegmentTest,
                               OpenMPForallSegmentTypes);

#endif
