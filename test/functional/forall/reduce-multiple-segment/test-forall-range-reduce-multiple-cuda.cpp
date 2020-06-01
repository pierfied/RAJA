//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "tests/test-forall-range-reduce-multiple.hpp"

#if defined(RAJA_ENABLE_CUDA)

#include "../test-forall-execpol.hpp"
#include "../test-reducepol.hpp"

// Cartesian product of types for Cuda tests
using CudaForallRangeReduceMultipleTypes =
  Test< camp::cartesian_product<ReduceMultipleDataTypeList, 
                                HostResourceList, 
                                CudaForallExecPols,
                                CudaReducePols>>::Types;

INSTANTIATE_TYPED_TEST_SUITE_P(Cuda,
                               ForallRangeReduceMultipleTest,
                               CudaForallRangeReduceMultipleTypes);

#endif
