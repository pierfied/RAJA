//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "test-forall-atomic-ref.hpp"

#if defined(RAJA_ENABLE_OPENMP)
using OmpAtomicForallRefOtherTypes = 
  Test< camp::cartesian_product< OpenMPForallExecPols,
                                 OpenMPAtomicPols,
                                 HostResourceList,
                                 AtomicDataTypeList > >::Types;

INSTANTIATE_TYPED_TEST_SUITE_P( OmpTest,
                                ForallAtomicRefOtherFunctionalTest,
                                OmpAtomicForallRefOtherTypes );
#endif
