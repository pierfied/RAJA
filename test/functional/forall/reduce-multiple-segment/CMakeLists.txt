###############################################################################
# Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

raja_add_test(
  NAME test-forall-segment-reduce-multiple-seq
  SOURCES test-forall-segment-reduce-multiple-seq.cpp)

if(RAJA_ENABLE_OPENMP)
  raja_add_test(
    NAME test-forall-segment-reduce-multiple-openmp
    SOURCES test-forall-segment-reduce-multiple-openmp.cpp)
endif()

#if(RAJA_ENABLE_TARGET_OPENMP)
#  raja_add_test(
#    NAME test-forall-segment-reduce-multiple-openmp-target
#    SOURCES test-forall-segment-reduce-multiple-openmp-target.cpp)
#endif()
 
if(RAJA_ENABLE_TBB)
  raja_add_test(
    NAME test-forall-segment-reduce-multiple-tbb
    SOURCES test-forall-segment-reduce-multiple-tbb.cpp)
endif()

if(RAJA_ENABLE_CUDA)
  raja_add_test(
    NAME test-forall-segment-reduce-multiple-cuda
    SOURCES test-forall-segment-reduce-multiple-cuda.cpp)
endif()
 
#if(RAJA_ENABLE_HIP)
#  raja_add_test(
#    NAME test-forall-segment-reduce-multiple-hip
#    SOURCES test-forall-segment-reduce-multiple-hip.cpp)
#endif()
