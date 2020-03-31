/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file defining a SIMD register abstraction.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifdef __AVX2__

#ifndef RAJA_policy_vector_register_avx2_int32_HPP
#define RAJA_policy_vector_register_avx2_int32_HPP

#include "RAJA/config.hpp"
#include "RAJA/util/macros.hpp"
#include "RAJA/pattern/register.hpp"

// Include SIMD intrinsics header file
#include <immintrin.h>
#include <cmath>


namespace RAJA
{


  template<size_t N>
  class Register<avx2_register, int, N> :
    public internal::RegisterBase<Register<avx2_register, int, N>>
  {
    static_assert(N >= 1, "Vector must have at least 1 lane");
    static_assert(N <= 8, "AVX2 can only have 8 lanes of 32-bit ints");

    public:
      using self_type = Register<avx2_register, int, N>;
      using element_type = int;
      using register_type = __m256i;

      static constexpr size_t s_num_elem = N;

    private:
      register_type m_value;

      RAJA_INLINE
      __m256i createMask() const {
        // Generate a mask
        return  _mm256_set_epi32(
            N >= 8 ? -1 : 0,
            N >= 7 ? -1 : 0,
            N >= 6 ? -1 : 0,
            N >= 5 ? -1 : 0,
            N >= 4 ? -1 : 0,
            N >= 3 ? -1 : 0,
            N >= 2 ? -1 : 0,
            -1);
      }

      RAJA_INLINE
      __m256i createStridedOffsets(camp::idx_t stride) const {
        // Generate a strided offset list
        return  _mm256_set_epi32(
            7*stride, 6*stride, 5*stride, 4*stride,
            3*stride, 2*stride, stride, 0);
      }

      RAJA_INLINE
      __m256i createPermute1() const {
        // Generate a permutation for first round of min/max routines
        return  _mm256_set_epi32(
            N >= 7 ? 6 : 0,
            N >= 8 ? 7 : 0,
            N >= 5 ? 4 : 0,
            N >= 6 ? 5 : 0,
            N >= 3 ? 2 : 0,
            N >= 4 ? 3 : 0,
            N >= 1 ? 0 : 0,
            N >= 2 ? 1 : 0);
      }

      RAJA_INLINE
      __m256i createPermute2() const {
        // Generate a permutation for second round of min/max routines
        return  _mm256_set_epi32(
            N >= 6 ? 5 : 0,
            N >= 5 ? 4 : 0,
            N >= 8 ? 7 : 0,
            N >= 7 ? 6 : 0,
            N >= 2 ? 1 : 0,
            N >= 1 ? 0 : 0,
            N >= 4 ? 3 : 0,
            N >= 2 ? 2 : 0);
      }

    public:

      /*!
       * @brief Default constructor, zeros register contents
       */
      RAJA_INLINE
      Register() : m_value(_mm256_setzero_si256()) {
      }

      /*!
       * @brief Copy constructor from underlying simd register
       */
      RAJA_INLINE
      constexpr
      explicit Register(register_type const &c) : m_value(c) {}


      /*!
       * @brief Copy constructor
       */
      RAJA_INLINE
      constexpr
      Register(self_type const &c) : m_value(c.m_value) {}


      /*!
       * @brief Construct from scalar.
       * Sets all elements to same value (broadcast).
       */
      RAJA_INLINE
      Register(element_type const &c) : m_value(_mm256_set1_epi32(c)) {}


      /*!
       * @brief Strided load constructor, when scalars are located in memory
       * locations ptr, ptr+stride, ptr+2*stride, etc.
       *
       *
       * Note: this could be done with "gather" instructions if they are
       * available. (like in avx2, but not in avx)
       */
      RAJA_INLINE
      self_type &load(element_type const *ptr, camp::idx_t stride = 1){
        // Full vector width uses regular load/gather instruction
        if(N == 8){

          // Packed Load
          if(stride == 1){
            m_value = _mm256_loadu_si256((__m256i const *)ptr);
          }

          // Gather
          else{
            m_value = _mm256_i32gather_epi32(ptr,
                                          createStridedOffsets(stride),
                                          sizeof(element_type));
          }
        }

        // Not-full vector (1,2 or 3 doubles) uses a masked load/gather
        else {

          // Masked Packed Load
          if(stride == 1){
            m_value = _mm256_maskload_epi32(ptr, createMask());
          }

          // Masked Gather
          else{
            m_value = _mm256_mask_i32gather_epi32(_mm256_setzero_ps(),
                                          ptr,
                                          createStridedOffsets(stride),
                                          createMask(),
                                          sizeof(element_type));
          }
        }
        return *this;
      }



      /*!
       * @brief Strided store operation, where scalars are stored in memory
       * locations ptr, ptr+stride, ptr+2*stride, etc.
       *
       *
       * Note: this could be done with "scatter" instructions if they are
       * available.
       */
      RAJA_INLINE
      self_type const &store(element_type *ptr, size_t stride = 1) const{
        // Is this a packed store?
        if(stride == 1){
          // Is it full-width?
          if(N == 8){
            _mm256_storeu_epi32(ptr, m_value);
          }
          // Need to do a masked store
          else{
            _mm256_maskstore_epi32(ptr, createMask(), m_value);
          }

        }

        // Scatter operation:  AVX2 doesn't have a scatter, so it's manual
        else{
          for(size_t i = 0;i < N;++ i){
            ptr[i*stride] = get(i);
          }
        }
        return *this;
      }

      /*!
       * @brief Get scalar value from vector register
       * @param i Offset of scalar to get
       * @return Returns scalar value at i
       */
      template<typename IDX>
      constexpr
      RAJA_INLINE
      element_type get(IDX i) const
      {
        // got to be a nicer way to do this!?!?
        switch(i){
          case 0: return _mm256_extract_epi32(m_value, 0);
          case 1: return _mm256_extract_epi32(m_value, 1);
          case 2: return _mm256_extract_epi32(m_value, 2);
          case 3: return _mm256_extract_epi32(m_value, 3);
          case 4: return _mm256_extract_epi32(m_value, 4);
          case 5: return _mm256_extract_epi32(m_value, 5);
          case 6: return _mm256_extract_epi32(m_value, 6);
          case 7: return _mm256_extract_epi32(m_value, 7);
        }
        return 0;
      }


      /*!
       * @brief Set scalar value in vector register
       * @param i Offset of scalar to set
       * @param value Value of scalar to set
       */
      template<typename IDX>
      RAJA_INLINE
      self_type &set(IDX i, element_type value)
      {
        // got to be a nicer way to do this!?!?
        switch(i){
          case 0: m_value = _mm256_insert_epi32(m_value, value, 0); break;
          case 1: m_value = _mm256_insert_epi32(m_value, value, 1); break;
          case 2: m_value = _mm256_insert_epi32(m_value, value, 2); break;
          case 3: m_value = _mm256_insert_epi32(m_value, value, 3); break;
          case 4: m_value = _mm256_insert_epi32(m_value, value, 4); break;
          case 5: m_value = _mm256_insert_epi32(m_value, value, 5); break;
          case 6: m_value = _mm256_insert_epi32(m_value, value, 6); break;
          case 7: m_value = _mm256_insert_epi32(m_value, value, 7); break;
        }

        return *this;
      }

      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type &broadcast(element_type const &value){
        m_value =  _mm256_set1_epi32(value);
        return *this;
      }


      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type &copy(self_type const &src){
        m_value = src.m_value;
        return *this;
      }

      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type add(self_type const &b) const {
        return self_type(_mm256_add_epi32(m_value, b.m_value));
      }

      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type subtract(self_type const &b) const {
        return self_type(_mm256_sub_epi32(m_value, b.m_value));
      }

      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type multiply(self_type const &b) const {

        // the AVX2 epi32 multiply only multiplies the even elements
        // and provides 64-bit results
        // need to do some repacking to get this to work

        // multiply 0, 2, 4, 6
        auto prod_even = _mm256_mul_epi32(m_value, b.m_value);

        // Swap 32-bit words
        auto sh_a = _mm256_permute_ps(m_value, 0xB1);
        auto sh_b = _mm256_permute_ps(b.m_value, 0xB1);

        // multiply 1, 3, 5, 7
        auto prod_odd = _mm256_mul_epi32(sh_a, sh_b);

        // Stitch prod_odd and prod_even back together
        auto sh_odd = _mm256_permute_ps(prod_odd, 0xB1);
        return self_type(_mm256_blend_epi32(prod_even, sh_odd, 0xAA));
      }

      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type divide(self_type const &b) const {
        // AVX2 does not supply an integer divide, so do it manually
        return self_type(_mm256_set_epi32(
            N >= 8 ? get(7)/b.get(7) : 0,
            N >= 7 ? get(6)/b.get(6) : 0,
            N >= 6 ? get(5)/b.get(5) : 0,
            N >= 5 ? get(4)/b.get(4) : 0,
            N >= 4 ? get(3)/b.get(3) : 0,
            N >= 3 ? get(2)/b.get(2) : 0,
            N >= 2 ? get(1)/b.get(1) : 0,
            N >= 1 ? get(0)/b.get(0) : 0
            ));
      }



      /*!
       * @brief Sum the elements of this vector
       * @return Sum of the values of the vectors scalar elements
       */
      RAJA_INLINE
      element_type sum() const
      {
        // Some simple cases
        if(N == 1){
          return get(0);
        }
        if(N == 2){
          return get(0) + get(1);
        }

        // swap odd-even pairs and add
        auto sh1 = _mm256_permute_ps(m_value, 0xB1);
        auto red1 = _mm256_add_epi32(m_value, sh1);

        if(N == 3 || N == 4){
          return _mm256_extract_epi32(red1, 0) + _mm256_extract_epi32(red1, 2);
        }

        // swap odd-even quads and add
        auto sh2 = _mm256_permute_ps(red1, 0x4E);
        auto red2 = _mm256_add_epi32(red1, sh2);

        return _mm256_extract_epi32(red2, 0) + _mm256_extract_epi32(red2, 4);
      }


      /*!
       * @brief Returns the largest element
       * @return The largest scalar element in the register
       */
      RAJA_INLINE
      element_type max() const
      {
        // Some simple cases
        if(N == 1){
          return get(0);
        }

        if(N == 2){
          return std::max<element_type>(get(0), get(1));
        }

        // swap odd-even pairs and add
        auto sh1 = _mm256_permutevar8x32_epi32(m_value, createPermute1());
        auto red1 = _mm256_max_epi32(m_value, sh1);

        if(N == 3){
          return std::max<element_type>(_mm256_extract_epi32(red1, 0), get(2));
        }
        if(N == 4){
          return std::max<element_type>(_mm256_extract_epi32(red1, 0), _mm256_extract_epi32(red1, 2));
        }

        // swap odd-even quads and add
        auto sh2 = _mm256_permutevar8x32_epi32(red1, createPermute2());
        auto red2 = _mm256_max_epi32(red1, sh2);

        return std::max<element_type>(_mm256_extract_epi32(red2, 0), _mm256_extract_epi32(red2, 4));
      }

      /*!
       * @brief Returns element-wise largest values
       * @return Vector of the element-wise max values
       */
      RAJA_INLINE
      self_type vmax(self_type a) const
      {
        return self_type(_mm256_max_epi32(m_value, a.m_value));
      }

      /*!
       * @brief Returns the largest element
       * @return The largest scalar element in the register
       */
      RAJA_INLINE
      element_type min() const
      {
        // Some simple cases
        if(N == 1){
          return get(0);
        }

        if(N == 2){
          return std::min<element_type>(get(0), get(1));
        }

        // swap odd-even pairs and add
        auto sh1 = _mm256_permutevar8x32_epi32(m_value, createPermute1());
        auto red1 = _mm256_min_epi32(m_value, sh1);

        if(N == 3){
          return std::min<element_type>(_mm256_extract_epi32(red1, 0), get(2));
        }
        if(N == 4){
          return std::min<element_type>(_mm256_extract_epi32(red1, 0), _mm256_extract_epi32(red1, 2));
        }

        // swap odd-even quads and add
        auto sh2 = _mm256_permutevar8x32_epi32(red1, createPermute2());
        auto red2 = _mm256_min_epi32(red1, sh2);

        return std::min<element_type>(_mm256_extract_epi32(red2, 0), _mm256_extract_epi32(red2, 4));
      }

      /*!
       * @brief Returns element-wise largest values
       * @return Vector of the element-wise max values
       */
      RAJA_INLINE
      self_type vmin(self_type a) const
      {
        return self_type(_mm256_min_epi32(m_value, a.m_value));
      }
  };



}  // namespace RAJA


#endif

#endif //__AVX2__