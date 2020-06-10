/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing RAJA headers for apollo (CUDA) execution.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-17, Lawrence Livermore National Security, LLC.
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

#ifndef RAJA_apollo_cuda_HPP
#define RAJA_apollo_cuda_HPP

#include "RAJA/config.hpp"

#include <cuda.h>
#include <cuda_runtime.h>

#include "RAJA/policy/apollo_cuda/forall.hpp"
#include "RAJA/policy/apollo_cuda/policy.hpp"
//TODO: We'll probably need this too...
//#include "RAJA/policy/apollo_cuda/scan.hpp"


#endif  // closing endif for header file include guard
