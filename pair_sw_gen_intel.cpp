/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov
   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.
   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: PotC / Markus Hoehnerbach (RWTH Aachen)
------------------------------------------------------------------------- */

#include "pair_sw_gen_intel.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fenv.h>
#include "atom.h"
#include "comm.h"
#include "force.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "memory.h"
#include "error.h"
#include "math_const.h"

using namespace LAMMPS_NS;
using namespace MathConst;

#include "suffix.h"
#include "modify.h"
#include "fix_intel.h"
#include <immintrin.h>

/* ---------------------------------------------------------------------- */

PairSwGenIntel::PairSwGenIntel(LAMMPS *lmp) : Pair(lmp) {
  manybody_flag = 1;
  one_coeff = 1;
  single_enable = 0;
  restartinfo = 0;
  ghostneigh = 1;
  suffix_flag |= Suffix::INTEL;
}

/* ---------------------------------------------------------------------- */

PairSwGenIntel::~PairSwGenIntel() {
  if (copymode) return;
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);

    memory->destroy(type_map);

    memory->destroy(cutghost);

    memory->destroy(param_A);
    memory->destroy(param_a);
    memory->destroy(param_B);
    memory->destroy(param_p);
    memory->destroy(param_q);
    memory->destroy(param_epsilon);
    memory->destroy(param_sigma);
    memory->destroy(param_gamma);
    memory->destroy(param_lambda);
    memory->destroy(param_cos_theta0);
  }
}

/* ---------------------------------------------------------------------- */

void PairSwGenIntel::compute(int eflag, int vflag) {
  if (fix->precision() == FixIntel::PREC_MODE_MIXED)
    compute<float,double>(eflag, vflag, fix->get_mixed_buffers(),
                          force_const_single);
  else if (fix->precision() == FixIntel::PREC_MODE_DOUBLE)
    compute<double,double>(eflag, vflag, fix->get_double_buffers(),
                           force_const_double);
  else
    compute<float,float>(eflag, vflag, fix->get_single_buffers(),
                         force_const_single);

  fix->balance_stamp();
  vflag_fdotr = 0;
}

/* ---------------------------------------------------------------------- */

template <class flt_t, class acc_t>
void PairSwGenIntel::compute(int eflag, int vflag,
    IntelBuffers<flt_t,acc_t> *buffers,
    const ForceConst<flt_t> &fc
) {
  if (eflag || vflag) ev_setup(eflag,vflag);
  else evflag = vflag_fdotr = 0;

  const int inum = list->inum;
  const int nthreads = comm->nthreads;
  const int host_start = fix->host_start_pair();
  const int offload_end = fix->offload_end_pair();
  const int ago = neighbor->ago;

  if (ago != 0 && fix->separate_buffers() == 0) {
    fix->start_watch(TIME_PACK);

    int packthreads;
    if (nthreads > INTEL_HTHREADS) packthreads = nthreads;
    else packthreads = 1;
    #if defined(_OPENMP)
    #pragma omp parallel if(packthreads > 1)
    #endif
    {
      int ifrom, ito, tid;
      IP_PRE_omp_range_id_align(ifrom, ito, tid, atom->nlocal + atom->nghost,
                                packthreads, sizeof(ATOM_T));
      buffers->thr_pack(ifrom, ito, ago);
    }

    fix->stop_watch(TIME_PACK);
  }

  int ovflag = 0;
  if (vflag_fdotr) ovflag = 2;
  else if (vflag) ovflag = 1;
  if (eflag) {
    eval<1>(1, ovflag, buffers, fc, 0, offload_end);
    eval<1>(0, ovflag, buffers, fc, host_start, inum);
  } else {
    eval<0>(1, ovflag, buffers, fc, 0, offload_end);
    eval<0>(0, ovflag, buffers, fc, host_start, inum);
  }
}
template <int EFLAG, class flt_t, class acc_t>
void PairSwGenIntel::eval(const int offload, const int vflag,
                        IntelBuffers<flt_t,acc_t> *buffers,
                        const ForceConst<flt_t> &fc,
                        const int astart, const int aend)
{
  const bool NEWTON_PAIR = true;
  const int inum = aend - astart;
  if (inum == 0) return;

  int nlocal, nall, minlocal;
  fix->get_buffern(offload, nlocal, nall, minlocal);

  const int ago = neighbor->ago;
  IP_PRE_pack_separate_buffers(fix, buffers, ago, offload, nlocal, nall);

  ATOM_T * _noalias const x = buffers->get_x(offload);

  const int * _noalias const numneigh = list->numneigh;
  const int * _noalias const cnumneigh = buffers->cnumneigh(list);
  const int * _noalias const firstneigh = buffers->firstneigh(list);
  const int eatom = this->eflag_atom;
  int tp1 = atom->ntypes + 1;

  // Determine how much data to transfer
  int x_size, q_size, f_stride, ev_size, separate_flag;
  IP_PRE_get_transfern(ago, NEWTON_PAIR, EFLAG, vflag,
                       buffers, offload, fix, separate_flag,
                       x_size, q_size, ev_size, f_stride);

  int tc;
  FORCE_T * _noalias f_start;
  acc_t * _noalias ev_global;
  IP_PRE_get_buffers(offload, buffers, fix, tc, f_start, ev_global);
  const int nthreads = tc;
  int *overflow = fix->get_off_overflow_flag();

  {
    #if defined(__MIC__) && defined(_LMP_INTEL_OFFLOAD)
    *timer_compute = MIC_Wtime();
    #endif

    IP_PRE_repack_for_offload(NEWTON_PAIR, separate_flag, nlocal, nall,
                              f_stride, x, 0);

    acc_t oevdwl, ov0, ov1, ov2, ov3, ov4, ov5;
    if (EFLAG) oevdwl = (acc_t)0;
    if (vflag) ov0 = ov1 = ov2 = ov3 = ov4 = ov5 = (acc_t)0;

    // loop over neighbors of my atoms
    #if defined(_OPENMP)
    #pragma omp parallel reduction(+:oevdwl,ov0,ov1,ov2,ov3,ov4,ov5)
    #endif
    {
      int iifrom, iito, tid;
      IP_PRE_omp_range_id_vec(iifrom, iito, tid, inum, nthreads,
                              INTEL_VECTOR_WIDTH);
      iifrom += astart;
      iito += astart;

      FORCE_T * _noalias const f = f_start - minlocal + (tid * f_stride);
      memset(f + minlocal, 0, f_stride * sizeof(FORCE_T));


      __m512d i_a_2 = _mm512_setzero_pd();
      __m512d i_mul_adj_36 = _mm512_mul_pd(_mm512_set1_pd(1), _mm512_recip_pd(_mm512_set1_pd(2)));
      __m512d i_a_124 = _mm512_setzero_pd();
      for (int t_0 = iifrom; t_0 < iito; t_0 += 8) {
        __m512i i_i_3 = _mm512_add_epi32(_mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0), _mm512_set1_epi32(t_0));
        __mmask8 t_1 = _mm512_kand(0xFF, _mm512_cmplt_epi32_mask(i_i_3, _mm512_set1_epi32(iito)));
        __m512d i_px_4 = _mm512_mask_i32gather_pd(_mm512_undefined_pd(), t_1, _mm512_castsi512_si256(_mm512_slli_epi32(i_i_3, 3)), &x[0].x, 4);
        __m512d i_py_4 = _mm512_mask_i32gather_pd(_mm512_undefined_pd(), t_1, _mm512_castsi512_si256(_mm512_slli_epi32(i_i_3, 3)), &x[0].y, 4);
        __m512d i_pz_4 = _mm512_mask_i32gather_pd(_mm512_undefined_pd(), t_1, _mm512_castsi512_si256(_mm512_slli_epi32(i_i_3, 3)), &x[0].z, 4);
        __m512i i_ty_4 = _mm512_mask_i32gather_epi32(_mm512_undefined_epi32(), t_1, _mm512_slli_epi32(i_i_3, 3), &x[0].w, 4);
        __m512d i_a_5 = _mm512_setzero_pd();
        __m512d i_fx_39 = _mm512_setzero_pd();
        __m512d i_fy_39 = _mm512_setzero_pd();
        __m512d i_fz_39 = _mm512_setzero_pd();
        __m512d i_a_127 = _mm512_setzero_pd();
        __m512d i_fx_176 = _mm512_setzero_pd();
        __m512d i_fy_176 = _mm512_setzero_pd();
        __m512d i_fz_176 = _mm512_setzero_pd();
        __m512i i_i_6 = _mm512_setzero_epi32();
        __m512i t_2 = _mm512_mask_i32gather_epi32(_mm512_undefined_epi32(), t_1, i_i_3, numneigh, 4);
        for (;;) {
          __mmask8 t_3 = _mm512_kand(t_1, _mm512_cmplt_epi32_mask(i_i_6, t_2));
          if (_mm512_kortestz(t_3, t_3)) break;
          __m512i i_a_7 = _mm512_mask_i32gather_epi32(_mm512_undefined_epi32(), t_3, _mm512_add_epi32(_mm512_mask_i32gather_epi32(_mm512_undefined_epi32(), t_3, i_i_3, cnumneigh, 4), i_i_6), firstneigh, 4);
          __m512d i_px_8 = _mm512_mask_i32gather_pd(_mm512_undefined_pd(), t_3, _mm512_castsi512_si256(_mm512_slli_epi32(i_a_7, 3)), &x[0].x, 4);
          __m512d i_py_8 = _mm512_mask_i32gather_pd(_mm512_undefined_pd(), t_3, _mm512_castsi512_si256(_mm512_slli_epi32(i_a_7, 3)), &x[0].y, 4);
          __m512d i_pz_8 = _mm512_mask_i32gather_pd(_mm512_undefined_pd(), t_3, _mm512_castsi512_si256(_mm512_slli_epi32(i_a_7, 3)), &x[0].z, 4);
          __m512i i_ty_8 = _mm512_mask_i32gather_epi32(_mm512_undefined_epi32(), t_3, _mm512_slli_epi32(i_a_7, 3), &x[0].w, 4);
          __m512d i_dx_9 = _mm512_sub_pd(i_px_4, i_px_8);
          __m512d i_dy_9 = _mm512_sub_pd(i_py_4, i_py_8);
          __m512d i_dz_9 = _mm512_sub_pd(i_pz_4, i_pz_8);
          __m512d i_rsq_9 = _mm512_add_pd(_mm512_add_pd(_mm512_mul_pd(i_dx_9, i_dx_9), _mm512_mul_pd(i_dy_9, i_dy_9)), _mm512_mul_pd(i_dz_9, i_dz_9));
          __m512d i_param_11 = _mm512_mask_i32gather_pd(_mm512_undefined_pd(), t_3, _mm512_castsi512_si256(_mm512_add_epi32(i_ty_8, _mm512_mullo_epi32(_mm512_set1_epi32(tp1), _mm512_add_epi32(i_ty_4, _mm512_mullo_epi32(_mm512_set1_epi32(tp1), _mm512_setzero_epi32()))))), &this->param_sigma[0][0], 8);
          __m512d i_mul_val_12 = _mm512_mul_pd(_mm512_mask_i32gather_pd(_mm512_undefined_pd(), t_3, _mm512_castsi512_si256(_mm512_add_epi32(i_ty_8, _mm512_mullo_epi32(_mm512_set1_epi32(tp1), _mm512_add_epi32(i_ty_4, _mm512_mullo_epi32(_mm512_set1_epi32(tp1), _mm512_setzero_epi32()))))), &this->param_a[0][0], 8), i_param_11);
          __mmask8 t_4 = _mm512_kand(_mm512_cmpnle_pd_mask(i_rsq_9, _mm512_mul_pd(i_mul_val_12, i_mul_val_12)), t_3);
          if (! _mm512_kortestz(t_4, t_4)) {
            i_i_6 = _mm512_mask_add_epi32(i_i_6, t_4, i_i_6, _mm512_set1_epi32(1));
            continue;
          }
          __m512d i_r_13 = _mm512_sqrt_pd(i_rsq_9);
          __m512d i_param_14 = _mm512_mask_i32gather_pd(_mm512_undefined_pd(), t_3, _mm512_castsi512_si256(_mm512_add_epi32(i_ty_8, _mm512_mullo_epi32(_mm512_set1_epi32(tp1), _mm512_add_epi32(i_ty_4, _mm512_mullo_epi32(_mm512_set1_epi32(tp1), _mm512_setzero_epi32()))))), &this->param_A[0][0], 8);
          __m512d i_param_15 = _mm512_mask_i32gather_pd(_mm512_undefined_pd(), t_3, _mm512_castsi512_si256(_mm512_add_epi32(i_ty_8, _mm512_mullo_epi32(_mm512_set1_epi32(tp1), _mm512_add_epi32(i_ty_4, _mm512_mullo_epi32(_mm512_set1_epi32(tp1), _mm512_setzero_epi32()))))), &this->param_epsilon[0][0], 8);
          __m512d i_param_16 = _mm512_mask_i32gather_pd(_mm512_undefined_pd(), t_3, _mm512_castsi512_si256(_mm512_add_epi32(i_ty_8, _mm512_mullo_epi32(_mm512_set1_epi32(tp1), _mm512_add_epi32(i_ty_4, _mm512_mullo_epi32(_mm512_set1_epi32(tp1), _mm512_setzero_epi32()))))), &this->param_B[0][0], 8);
          __m512d i_mul_val_18 = _mm512_mul_pd(i_param_11, _mm512_recip_pd(i_r_13));
          __m512d i_param_19 = _mm512_mask_i32gather_pd(_mm512_undefined_pd(), t_3, _mm512_castsi512_si256(_mm512_add_epi32(i_ty_8, _mm512_mullo_epi32(_mm512_set1_epi32(tp1), _mm512_add_epi32(i_ty_4, _mm512_mullo_epi32(_mm512_set1_epi32(tp1), _mm512_setzero_epi32()))))), &this->param_p[0][0], 8);
          __m512d i_param_24 = _mm512_mask_i32gather_pd(_mm512_undefined_pd(), t_3, _mm512_castsi512_si256(_mm512_add_epi32(i_ty_8, _mm512_mullo_epi32(_mm512_set1_epi32(tp1), _mm512_add_epi32(i_ty_4, _mm512_mullo_epi32(_mm512_set1_epi32(tp1), _mm512_setzero_epi32()))))), &this->param_q[0][0], 8);
          __m512d i_v_26 = _mm512_sub_pd(_mm512_mul_pd(i_param_16, _mm512_pow_pd(i_mul_val_18, i_param_19)), _mm512_pow_pd(i_mul_val_18, i_param_24));
          __m512d i_v_31 = _mm512_sub_pd(i_r_13, i_mul_val_12);
          __m512d i_bultin_33 = _mm512_exp_pd(_mm512_mul_pd(i_param_11, _mm512_recip_pd(i_v_31)));
          i_a_5 = _mm512_mask_add_pd(i_a_5, t_3, i_a_5, _mm512_mul_pd(_mm512_mul_pd(_mm512_mul_pd(i_param_14, i_param_15), i_v_26), i_bultin_33));
          __m512d i_mul_adj_72 = _mm512_mul_pd(_mm512_mul_pd(_mm512_mul_pd(i_mul_adj_36, i_param_14), i_param_15), i_bultin_33);
          __m512d i_adj_by_r_121 = _mm512_mul_pd(_mm512_sub_pd(_mm512_sub_pd(_mm512_mul_pd(_mm512_mul_pd(_mm512_mul_pd(i_mul_adj_72, i_param_11), _mm512_mul_pd(i_param_24, _mm512_pow_pd(i_mul_val_18, _mm512_sub_pd(i_param_24, _mm512_set1_pd(1))))), _mm512_recip_pd(_mm512_mul_pd(i_r_13, i_r_13))), _mm512_mul_pd(_mm512_mul_pd(_mm512_mul_pd(_mm512_mul_pd(i_mul_adj_72, i_param_11), _mm512_mul_pd(i_param_16, i_param_19)), _mm512_pow_pd(i_mul_val_18, _mm512_sub_pd(i_param_19, _mm512_set1_pd(1)))), _mm512_recip_pd(_mm512_mul_pd(i_r_13, i_r_13)))), _mm512_mul_pd(_mm512_mul_pd(_mm512_mul_pd(_mm512_mul_pd(i_bultin_33, i_mul_adj_36), _mm512_mul_pd(i_param_11, i_param_14)), _mm512_mul_pd(i_param_15, i_v_26)), _mm512_recip_pd(_mm512_mul_pd(i_v_31, i_v_31)))), _mm512_recip_pd(i_r_13));
          i_fx_39 = _mm512_mask_add_pd(i_fx_39, t_3, i_fx_39, _mm512_sub_pd(_mm512_setzero_pd(), _mm512_mul_pd(i_adj_by_r_121, i_dx_9)));
          i_fy_39 = _mm512_mask_add_pd(i_fy_39, t_3, i_fy_39, _mm512_sub_pd(_mm512_setzero_pd(), _mm512_mul_pd(i_adj_by_r_121, i_dy_9)));
          i_fz_39 = _mm512_mask_add_pd(i_fz_39, t_3, i_fz_39, _mm512_sub_pd(_mm512_setzero_pd(), _mm512_mul_pd(i_adj_by_r_121, i_dz_9)));
          __m256i t_6 = _mm512_castsi512_si256(_mm512_slli_epi32(i_a_7, 3));
          __m512d t_7 = _mm512_mul_pd(i_adj_by_r_121, i_dx_9);
          __m512d t_8 = _mm512_mul_pd(i_adj_by_r_121, i_dy_9);
          __m512d t_9 = _mm512_mul_pd(i_adj_by_r_121, i_dz_9);
          __mmask8 t_5 = t_3;
          while (t_5) {
            __m512i conf0 = _mm512_maskz_conflict_epi32(t_5, _mm512_castsi256_si512(t_6));
            __m512i conf1 = _mm512_broadcastmw_epi32(t_5);
            __m512i conf2  = _mm512_and_si512(conf0, conf1);
            __mmask8 conf3 = _mm512_mask_testn_epi32_mask(t_5, conf2, conf2);
            t_5 = t_5 & (~conf3);
            _mm512_mask_i32scatter_pd(&f[0].x, conf3, t_6, _mm512_add_pd(t_7, _mm512_mask_i32gather_pd(_mm512_undefined_pd(), conf3, t_6, &f[0].x, 4)), 4);
            _mm512_mask_i32scatter_pd(&f[0].y, conf3, t_6, _mm512_add_pd(t_8, _mm512_mask_i32gather_pd(_mm512_undefined_pd(), conf3, t_6, &f[0].y, 4)), 4);
            _mm512_mask_i32scatter_pd(&f[0].z, conf3, t_6, _mm512_add_pd(t_9, _mm512_mask_i32gather_pd(_mm512_undefined_pd(), conf3, t_6, &f[0].z, 4)), 4);
          }
          __m512d i_a_135 = _mm512_setzero_pd();
          __m512d i_fx_180 = _mm512_setzero_pd();
          __m512d i_fy_180 = _mm512_setzero_pd();
          __m512d i_fz_180 = _mm512_setzero_pd();
          __m512i i_i_136 = _mm512_setzero_epi32();
          __m512i t_10 = _mm512_mask_i32gather_epi32(_mm512_undefined_epi32(), t_3, i_i_3, numneigh, 4);
          for (;;) {
            __mmask8 t_11 = _mm512_kand(t_3, _mm512_cmplt_epi32_mask(i_i_136, t_10));
            if (_mm512_kortestz(t_11, t_11)) break;
            __m512i i_a_137 = _mm512_mask_i32gather_epi32(_mm512_undefined_epi32(), t_11, _mm512_add_epi32(_mm512_mask_i32gather_epi32(_mm512_undefined_epi32(), t_11, i_i_3, cnumneigh, 4), i_i_136), firstneigh, 4);
            __m512d i_px_138 = _mm512_mask_i32gather_pd(_mm512_undefined_pd(), t_11, _mm512_castsi512_si256(_mm512_slli_epi32(i_a_137, 3)), &x[0].x, 4);
            __m512d i_py_138 = _mm512_mask_i32gather_pd(_mm512_undefined_pd(), t_11, _mm512_castsi512_si256(_mm512_slli_epi32(i_a_137, 3)), &x[0].y, 4);
            __m512d i_pz_138 = _mm512_mask_i32gather_pd(_mm512_undefined_pd(), t_11, _mm512_castsi512_si256(_mm512_slli_epi32(i_a_137, 3)), &x[0].z, 4);
            __m512i i_ty_138 = _mm512_mask_i32gather_epi32(_mm512_undefined_epi32(), t_11, _mm512_slli_epi32(i_a_137, 3), &x[0].w, 4);
            __m512d i_dx_139 = _mm512_sub_pd(i_px_4, i_px_138);
            __m512d i_dy_139 = _mm512_sub_pd(i_py_4, i_py_138);
            __m512d i_dz_139 = _mm512_sub_pd(i_pz_4, i_pz_138);
            __m512d i_rsq_139 = _mm512_add_pd(_mm512_add_pd(_mm512_mul_pd(i_dx_139, i_dx_139), _mm512_mul_pd(i_dy_139, i_dy_139)), _mm512_mul_pd(i_dz_139, i_dz_139));
            __m512d i_param_141 = _mm512_mask_i32gather_pd(_mm512_undefined_pd(), t_11, _mm512_castsi512_si256(_mm512_add_epi32(i_ty_138, _mm512_mullo_epi32(_mm512_set1_epi32(tp1), _mm512_add_epi32(i_ty_4, _mm512_mullo_epi32(_mm512_set1_epi32(tp1), _mm512_setzero_epi32()))))), &this->param_sigma[0][0], 8);
            __m512d i_mul_val_142 = _mm512_mul_pd(_mm512_mask_i32gather_pd(_mm512_undefined_pd(), t_11, _mm512_castsi512_si256(_mm512_add_epi32(i_ty_138, _mm512_mullo_epi32(_mm512_set1_epi32(tp1), _mm512_add_epi32(i_ty_4, _mm512_mullo_epi32(_mm512_set1_epi32(tp1), _mm512_setzero_epi32()))))), &this->param_a[0][0], 8), i_param_141);
            __mmask8 t_12 = _mm512_kand(_mm512_cmpnle_pd_mask(i_rsq_139, _mm512_mul_pd(i_mul_val_142, i_mul_val_142)), t_11);
            if (! _mm512_kortestz(t_12, t_12)) {
              i_i_136 = _mm512_mask_add_epi32(i_i_136, t_12, i_i_136, _mm512_set1_epi32(1));
              continue;
            }
            __mmask8 t_13 = _mm512_kand(_mm512_cmpeq_epi32_mask(i_a_137, i_a_7), t_11);
            if (! _mm512_kortestz(t_13, t_13)) {
              i_i_136 = _mm512_mask_add_epi32(i_i_136, t_13, i_i_136, _mm512_set1_epi32(1));
              continue;
            }
            __m512d i_r_144 = _mm512_sqrt_pd(i_rsq_139);
            __m512d i_cos_148 = _mm512_mul_pd(_mm512_add_pd(_mm512_add_pd(_mm512_mul_pd(i_dx_9, i_dx_139), _mm512_mul_pd(i_dy_9, i_dy_139)), _mm512_mul_pd(i_dz_9, i_dz_139)), _mm512_recip_pd(_mm512_mul_pd(i_r_13, i_r_144)));
            __m512d i_param_149 = _mm512_mask_i32gather_pd(_mm512_undefined_pd(), t_11, _mm512_castsi512_si256(_mm512_add_epi32(i_ty_138, _mm512_mullo_epi32(_mm512_set1_epi32(tp1), _mm512_add_epi32(i_ty_8, _mm512_mullo_epi32(_mm512_set1_epi32(tp1), _mm512_add_epi32(i_ty_4, _mm512_mullo_epi32(_mm512_set1_epi32(tp1), _mm512_setzero_epi32()))))))), &this->param_lambda[0][0][0], 8);
            __m512d i_v_153 = _mm512_sub_pd(i_cos_148, _mm512_mask_i32gather_pd(_mm512_undefined_pd(), t_11, _mm512_castsi512_si256(_mm512_add_epi32(i_ty_138, _mm512_mullo_epi32(_mm512_set1_epi32(tp1), _mm512_add_epi32(i_ty_8, _mm512_mullo_epi32(_mm512_set1_epi32(tp1), _mm512_add_epi32(i_ty_4, _mm512_mullo_epi32(_mm512_set1_epi32(tp1), _mm512_setzero_epi32()))))))), &this->param_cos_theta0[0][0][0], 8));
            __m512d i_v_154 = _mm512_mul_pd(i_v_153, i_v_153);
            __m512d i_param_155 = _mm512_mask_i32gather_pd(_mm512_undefined_pd(), t_11, _mm512_castsi512_si256(_mm512_add_epi32(i_ty_8, _mm512_mullo_epi32(_mm512_set1_epi32(tp1), _mm512_add_epi32(i_ty_4, _mm512_mullo_epi32(_mm512_set1_epi32(tp1), _mm512_setzero_epi32()))))), &this->param_gamma[0][0], 8);
            __m512d i_bultin_162 = _mm512_exp_pd(_mm512_mul_pd(_mm512_mul_pd(i_param_11, i_param_155), _mm512_recip_pd(i_v_31)));
            __m512d i_param_163 = _mm512_mask_i32gather_pd(_mm512_undefined_pd(), t_11, _mm512_castsi512_si256(_mm512_add_epi32(i_ty_138, _mm512_mullo_epi32(_mm512_set1_epi32(tp1), _mm512_add_epi32(i_ty_4, _mm512_mullo_epi32(_mm512_set1_epi32(tp1), _mm512_setzero_epi32()))))), &this->param_gamma[0][0], 8);
            __m512d i_v_168 = _mm512_sub_pd(i_r_144, i_mul_val_142);
            __m512d i_bultin_170 = _mm512_exp_pd(_mm512_mul_pd(_mm512_mul_pd(i_param_141, i_param_163), _mm512_recip_pd(i_v_168)));
            i_a_135 = _mm512_mask_add_pd(i_a_135, t_11, i_a_135, _mm512_mul_pd(_mm512_mul_pd(_mm512_mul_pd(_mm512_mul_pd(i_param_149, i_param_15), i_v_154), i_bultin_162), i_bultin_170));
            __m512d i_dx_199 = _mm512_sub_pd(i_px_8, i_px_138);
            __m512d i_dy_199 = _mm512_sub_pd(i_py_8, i_py_138);
            __m512d i_dz_199 = _mm512_sub_pd(i_pz_8, i_pz_138);
            __m512d i_adj_by_r_285 = _mm512_sub_pd(_mm512_setzero_pd(), _mm512_mul_pd(_mm512_mul_pd(_mm512_mul_pd(_mm512_mul_pd(i_bultin_162, i_bultin_170), _mm512_mul_pd(i_mul_adj_36, i_param_11)), _mm512_mul_pd(_mm512_mul_pd(i_param_149, i_param_15), _mm512_mul_pd(i_param_155, i_v_154))), _mm512_recip_pd(_mm512_mul_pd(_mm512_mul_pd(i_r_13, i_v_31), i_v_31))));
            i_fx_176 = _mm512_mask_add_pd(i_fx_176, t_11, i_fx_176, _mm512_sub_pd(_mm512_setzero_pd(), _mm512_mul_pd(i_adj_by_r_285, i_dx_9)));
            i_fy_176 = _mm512_mask_add_pd(i_fy_176, t_11, i_fy_176, _mm512_sub_pd(_mm512_setzero_pd(), _mm512_mul_pd(i_adj_by_r_285, i_dy_9)));
            i_fz_176 = _mm512_mask_add_pd(i_fz_176, t_11, i_fz_176, _mm512_sub_pd(_mm512_setzero_pd(), _mm512_mul_pd(i_adj_by_r_285, i_dz_9)));
            i_fx_180 = _mm512_mask_add_pd(i_fx_180, t_11, i_fx_180, _mm512_mul_pd(i_adj_by_r_285, i_dx_9));
            i_fy_180 = _mm512_mask_add_pd(i_fy_180, t_11, i_fy_180, _mm512_mul_pd(i_adj_by_r_285, i_dy_9));
            i_fz_180 = _mm512_mask_add_pd(i_fz_180, t_11, i_fz_180, _mm512_mul_pd(i_adj_by_r_285, i_dz_9));
            __m512d i_adj_by_r_287 = _mm512_sub_pd(_mm512_setzero_pd(), _mm512_mul_pd(_mm512_mul_pd(_mm512_mul_pd(_mm512_mul_pd(i_bultin_162, i_bultin_170), _mm512_mul_pd(i_mul_adj_36, i_param_141)), _mm512_mul_pd(_mm512_mul_pd(i_param_149, i_param_15), _mm512_mul_pd(i_param_163, i_v_154))), _mm512_recip_pd(_mm512_mul_pd(_mm512_mul_pd(i_r_144, i_v_168), i_v_168))));
            i_fx_176 = _mm512_mask_add_pd(i_fx_176, t_11, i_fx_176, _mm512_sub_pd(_mm512_setzero_pd(), _mm512_mul_pd(i_adj_by_r_287, i_dx_139)));
            i_fy_176 = _mm512_mask_add_pd(i_fy_176, t_11, i_fy_176, _mm512_sub_pd(_mm512_setzero_pd(), _mm512_mul_pd(i_adj_by_r_287, i_dy_139)));
            i_fz_176 = _mm512_mask_add_pd(i_fz_176, t_11, i_fz_176, _mm512_sub_pd(_mm512_setzero_pd(), _mm512_mul_pd(i_adj_by_r_287, i_dz_139)));
            __m512d i_adj_acos_292 = _mm512_mul_pd(_mm512_mul_pd(_mm512_mul_pd(_mm512_set1_pd(2), i_bultin_162), _mm512_mul_pd(i_bultin_170, i_mul_adj_36)), _mm512_mul_pd(_mm512_mul_pd(i_param_149, i_param_15), i_v_153));
            __m512d i_adj_by_r_293 = _mm512_mul_pd(_mm512_mul_pd(_mm512_sub_pd(_mm512_mul_pd(_mm512_set1_pd(1), _mm512_recip_pd(i_r_144)), _mm512_mul_pd(i_cos_148, _mm512_recip_pd(i_r_13))), i_adj_acos_292), _mm512_recip_pd(i_r_13));
            i_fx_176 = _mm512_mask_add_pd(i_fx_176, t_11, i_fx_176, _mm512_sub_pd(_mm512_setzero_pd(), _mm512_mul_pd(i_adj_by_r_293, i_dx_9)));
            i_fy_176 = _mm512_mask_add_pd(i_fy_176, t_11, i_fy_176, _mm512_sub_pd(_mm512_setzero_pd(), _mm512_mul_pd(i_adj_by_r_293, i_dy_9)));
            i_fz_176 = _mm512_mask_add_pd(i_fz_176, t_11, i_fz_176, _mm512_sub_pd(_mm512_setzero_pd(), _mm512_mul_pd(i_adj_by_r_293, i_dz_9)));
            i_fx_180 = _mm512_mask_add_pd(i_fx_180, t_11, i_fx_180, _mm512_mul_pd(i_adj_by_r_293, i_dx_9));
            i_fy_180 = _mm512_mask_add_pd(i_fy_180, t_11, i_fy_180, _mm512_mul_pd(i_adj_by_r_293, i_dy_9));
            i_fz_180 = _mm512_mask_add_pd(i_fz_180, t_11, i_fz_180, _mm512_mul_pd(i_adj_by_r_293, i_dz_9));
            __m512d i_adj_by_r_294 = _mm512_mul_pd(_mm512_mul_pd(_mm512_sub_pd(_mm512_mul_pd(_mm512_set1_pd(1), _mm512_recip_pd(i_r_13)), _mm512_mul_pd(i_cos_148, _mm512_recip_pd(i_r_144))), i_adj_acos_292), _mm512_recip_pd(i_r_144));
            i_fx_176 = _mm512_mask_add_pd(i_fx_176, t_11, i_fx_176, _mm512_sub_pd(_mm512_setzero_pd(), _mm512_mul_pd(i_adj_by_r_294, i_dx_139)));
            i_fy_176 = _mm512_mask_add_pd(i_fy_176, t_11, i_fy_176, _mm512_sub_pd(_mm512_setzero_pd(), _mm512_mul_pd(i_adj_by_r_294, i_dy_139)));
            i_fz_176 = _mm512_mask_add_pd(i_fz_176, t_11, i_fz_176, _mm512_sub_pd(_mm512_setzero_pd(), _mm512_mul_pd(i_adj_by_r_294, i_dz_139)));
            __m512d i_adj_by_r_295 = _mm512_sub_pd(_mm512_setzero_pd(), _mm512_mul_pd(i_adj_acos_292, _mm512_recip_pd(_mm512_mul_pd(i_r_13, i_r_144))));
            i_fx_180 = _mm512_mask_add_pd(i_fx_180, t_11, i_fx_180, _mm512_sub_pd(_mm512_setzero_pd(), _mm512_mul_pd(i_adj_by_r_295, i_dx_199)));
            i_fy_180 = _mm512_mask_add_pd(i_fy_180, t_11, i_fy_180, _mm512_sub_pd(_mm512_setzero_pd(), _mm512_mul_pd(i_adj_by_r_295, i_dy_199)));
            i_fz_180 = _mm512_mask_add_pd(i_fz_180, t_11, i_fz_180, _mm512_sub_pd(_mm512_setzero_pd(), _mm512_mul_pd(i_adj_by_r_295, i_dz_199)));
            __m256i t_15 = _mm512_castsi512_si256(_mm512_slli_epi32(i_a_137, 3));
            __m512d t_16 = _mm512_add_pd(_mm512_add_pd(_mm512_mul_pd(i_adj_by_r_287, i_dx_139), _mm512_mul_pd(i_adj_by_r_294, i_dx_139)), _mm512_mul_pd(i_adj_by_r_295, i_dx_199));
            __m512d t_17 = _mm512_add_pd(_mm512_add_pd(_mm512_mul_pd(i_adj_by_r_287, i_dy_139), _mm512_mul_pd(i_adj_by_r_294, i_dy_139)), _mm512_mul_pd(i_adj_by_r_295, i_dy_199));
            __m512d t_18 = _mm512_add_pd(_mm512_add_pd(_mm512_mul_pd(i_adj_by_r_287, i_dz_139), _mm512_mul_pd(i_adj_by_r_294, i_dz_139)), _mm512_mul_pd(i_adj_by_r_295, i_dz_199));
            __mmask8 t_14 = t_11;
            while (t_14) {
              __m512i conf0 = _mm512_maskz_conflict_epi32(t_14, _mm512_castsi256_si512(t_15));
              __m512i conf1 = _mm512_broadcastmw_epi32(t_14);
              __m512i conf2  = _mm512_and_si512(conf0, conf1);
              __mmask8 conf3 = _mm512_mask_testn_epi32_mask(t_14, conf2, conf2);
              t_14 = t_14 & (~conf3);
              _mm512_mask_i32scatter_pd(&f[0].x, conf3, t_15, _mm512_add_pd(t_16, _mm512_mask_i32gather_pd(_mm512_undefined_pd(), conf3, t_15, &f[0].x, 4)), 4);
              _mm512_mask_i32scatter_pd(&f[0].y, conf3, t_15, _mm512_add_pd(t_17, _mm512_mask_i32gather_pd(_mm512_undefined_pd(), conf3, t_15, &f[0].y, 4)), 4);
              _mm512_mask_i32scatter_pd(&f[0].z, conf3, t_15, _mm512_add_pd(t_18, _mm512_mask_i32gather_pd(_mm512_undefined_pd(), conf3, t_15, &f[0].z, 4)), 4);
            }
            i_i_136 = _mm512_add_epi32(i_i_136, _mm512_set1_epi32(1));
          }
          i_a_127 = _mm512_mask_add_pd(i_a_127, t_3, i_a_127, i_a_135);
          __m256i t_20 = _mm512_castsi512_si256(_mm512_slli_epi32(i_a_7, 3));
          __m512d t_21 = i_fx_180;
          __m512d t_22 = i_fy_180;
          __m512d t_23 = i_fz_180;
          __mmask8 t_19 = t_3;
          while (t_19) {
            __m512i conf0 = _mm512_maskz_conflict_epi32(t_19, _mm512_castsi256_si512(t_20));
            __m512i conf1 = _mm512_broadcastmw_epi32(t_19);
            __m512i conf2  = _mm512_and_si512(conf0, conf1);
            __mmask8 conf3 = _mm512_mask_testn_epi32_mask(t_19, conf2, conf2);
            t_19 = t_19 & (~conf3);
            _mm512_mask_i32scatter_pd(&f[0].x, conf3, t_20, _mm512_add_pd(t_21, _mm512_mask_i32gather_pd(_mm512_undefined_pd(), conf3, t_20, &f[0].x, 4)), 4);
            _mm512_mask_i32scatter_pd(&f[0].y, conf3, t_20, _mm512_add_pd(t_22, _mm512_mask_i32gather_pd(_mm512_undefined_pd(), conf3, t_20, &f[0].y, 4)), 4);
            _mm512_mask_i32scatter_pd(&f[0].z, conf3, t_20, _mm512_add_pd(t_23, _mm512_mask_i32gather_pd(_mm512_undefined_pd(), conf3, t_20, &f[0].z, 4)), 4);
          }
          i_i_6 = _mm512_add_epi32(i_i_6, _mm512_set1_epi32(1));
        }
        i_a_2 = _mm512_mask_add_pd(i_a_2, t_1, i_a_2, i_a_5);
        __m256i t_25 = _mm512_castsi512_si256(_mm512_slli_epi32(i_i_3, 3));
        __m512d t_26 = i_fx_39;
        __m512d t_27 = i_fy_39;
        __m512d t_28 = i_fz_39;
        __mmask8 t_24 = t_1;
        while (t_24) {
          __m512i conf0 = _mm512_maskz_conflict_epi32(t_24, _mm512_castsi256_si512(t_25));
          __m512i conf1 = _mm512_broadcastmw_epi32(t_24);
          __m512i conf2  = _mm512_and_si512(conf0, conf1);
          __mmask8 conf3 = _mm512_mask_testn_epi32_mask(t_24, conf2, conf2);
          t_24 = t_24 & (~conf3);
          _mm512_mask_i32scatter_pd(&f[0].x, conf3, t_25, _mm512_add_pd(t_26, _mm512_mask_i32gather_pd(_mm512_undefined_pd(), conf3, t_25, &f[0].x, 4)), 4);
          _mm512_mask_i32scatter_pd(&f[0].y, conf3, t_25, _mm512_add_pd(t_27, _mm512_mask_i32gather_pd(_mm512_undefined_pd(), conf3, t_25, &f[0].y, 4)), 4);
          _mm512_mask_i32scatter_pd(&f[0].z, conf3, t_25, _mm512_add_pd(t_28, _mm512_mask_i32gather_pd(_mm512_undefined_pd(), conf3, t_25, &f[0].z, 4)), 4);
        }
        i_a_124 = _mm512_mask_add_pd(i_a_124, t_1, i_a_124, i_a_127);
        __m256i t_30 = _mm512_castsi512_si256(_mm512_slli_epi32(i_i_3, 3));
        __m512d t_31 = i_fx_176;
        __m512d t_32 = i_fy_176;
        __m512d t_33 = i_fz_176;
        __mmask8 t_29 = t_1;
        while (t_29) {
          __m512i conf0 = _mm512_maskz_conflict_epi32(t_29, _mm512_castsi256_si512(t_30));
          __m512i conf1 = _mm512_broadcastmw_epi32(t_29);
          __m512i conf2  = _mm512_and_si512(conf0, conf1);
          __mmask8 conf3 = _mm512_mask_testn_epi32_mask(t_29, conf2, conf2);
          t_29 = t_29 & (~conf3);
          _mm512_mask_i32scatter_pd(&f[0].x, conf3, t_30, _mm512_add_pd(t_31, _mm512_mask_i32gather_pd(_mm512_undefined_pd(), conf3, t_30, &f[0].x, 4)), 4);
          _mm512_mask_i32scatter_pd(&f[0].y, conf3, t_30, _mm512_add_pd(t_32, _mm512_mask_i32gather_pd(_mm512_undefined_pd(), conf3, t_30, &f[0].y, 4)), 4);
          _mm512_mask_i32scatter_pd(&f[0].z, conf3, t_30, _mm512_add_pd(t_33, _mm512_mask_i32gather_pd(_mm512_undefined_pd(), conf3, t_30, &f[0].z, 4)), 4);
        }
      }
      oevdwl += _mm512_reduce_add_pd(_mm512_mul_pd(i_a_2, _mm512_recip_pd(_mm512_set1_pd(2))));
      oevdwl += _mm512_reduce_add_pd(_mm512_mul_pd(i_a_124, _mm512_recip_pd(_mm512_set1_pd(2))));


      IP_PRE_fdotr_reduce_omp(1, nall, minlocal, nthreads, f_start, f_stride,
                              x, offload, vflag, ov0, ov1, ov2, ov3, ov4, ov5);
    } // end omp

    IP_PRE_fdotr_reduce(1, nall, nthreads, f_stride, vflag,
                        ov0, ov1, ov2, ov3, ov4, ov5);

    if (EFLAG) {
      ev_global[0] = oevdwl;
      ev_global[1] = (acc_t)0.0;
    }
    if (vflag) {
      ev_global[2] = ov0;
      ev_global[3] = ov1;
      ev_global[4] = ov2;
      ev_global[5] = ov3;
      ev_global[6] = ov4;
      ev_global[7] = ov5;
    }
    #if defined(__MIC__) && defined(_LMP_INTEL_OFFLOAD)
    *timer_compute = MIC_Wtime() - *timer_compute;
    #endif
  } // end offload
  if (offload)
    fix->stop_watch(TIME_OFFLOAD_LATENCY);
  else
    fix->stop_watch(TIME_HOST_PAIR);

  if (EFLAG || vflag)
    fix->add_result_array(f_start, ev_global, offload, eatom, 0, vflag);
  else
    fix->add_result_array(f_start, 0, offload);
}

/* ---------------------------------------------------------------------- */

void PairSwGenIntel::init_style() {
  if (force->newton_pair == 0)
    error->all(FLERR, "Pair style sw/gen/intel requires atom IDs");
  int irequest = neighbor->request(this, instance_me);
  neighbor->requests[irequest]->half = 0;
  neighbor->requests[irequest]->full = 1;
  neighbor->requests[irequest]->ghost = 1;
  neighbor->requests[irequest]->intel = 1;
  int ifix = modify->find_fix("package_intel");
  if (ifix < 0) {
    error->all(FLERR, "The 'package intel' command is required for /intel styles");
  }
  fix = static_cast<FixIntel *>(modify->fix[ifix]);
  fix->pair_init_check();
  if (fix->precision() == FixIntel::PREC_MODE_MIXED) {
    pack_force_const(force_const_single, fix->get_mixed_buffers());
    fix->get_mixed_buffers()->need_tag(1);
  } else if (fix->precision() == FixIntel::PREC_MODE_DOUBLE) {
    pack_force_const(force_const_double, fix->get_double_buffers());
    fix->get_double_buffers()->need_tag(1);
  } else {
    pack_force_const(force_const_single, fix->get_single_buffers());
    fix->get_single_buffers()->need_tag(1);
  }
}

/* ---------------------------------------------------------------------- */

void PairSwGenIntel::allocate() {
  allocated = 1;
  int n = atom->ntypes;

  memory->create(setflag, n+1, n+1, "pair:setflag");
  memory->create(type_map, n+1, "pair:typemap");
  for (int i = 1; i <= n; i++)
    for (int j = i; j <= n; j++)
      setflag[i][j] = 0;

  memory->create(cutsq, n+1, n+1, "pair:cutsq");

  memory->create(cutghost, n+1, n+1, "pair:cutghost");

  memory->create(param_A, n+1, n+1, "pair:A");
  memory->create(param_a, n+1, n+1, "pair:a");
  memory->create(param_B, n+1, n+1, "pair:B");
  memory->create(param_p, n+1, n+1, "pair:p");
  memory->create(param_q, n+1, n+1, "pair:q");
  memory->create(param_epsilon, n+1, n+1, "pair:epsilon");
  memory->create(param_sigma, n+1, n+1, "pair:sigma");
  memory->create(param_gamma, n+1, n+1, "pair:gamma");
  memory->create(param_lambda, n+1, n+1, n+1, "pair:lambda");
  memory->create(param_cos_theta0, n+1, n+1, n+1, "pair:cos_theta0");
}

/* ---------------------------------------------------------------------- */

double PairSwGenIntel::init_one(int i, int j) {
  if (setflag[i][j] == 0) error->all(FLERR, "All pair coeffs are not set");
  cutghost[i][j] = cutmax;
  cutghost[j][i] = cutmax;
  return cutmax;
}

/* ---------------------------------------------------------------------- */

void PairSwGenIntel::settings(int narg, char **arg) {
  if (narg != 1) error->all(FLERR, "Illegal pair_style command");
  cutmax = atof(arg[0]);
}

/* ---------------------------------------------------------------------- */

void PairSwGenIntel::coeff(int narg, char **arg) {
  if (!allocated) allocate();

  if (narg != 3 + atom->ntypes)
    error->all(FLERR,"Incorrect args for pair coefficients");

  // insure I,J args are * *

  if (strcmp(arg[0], "*") != 0 || strcmp(arg[1], "*") != 0)
    error->all(FLERR, "Incorrect args for pair coefficients");

  for (int i = 3; i < narg; i++) {
  }

  char * file = arg[2];
  FILE *fp;
  if (comm->me == 0) {
    fp = force->open_potential(file);
    if (fp == NULL) {
      char str[128];
      sprintf(str, "Cannot open potential file %s",file);
      error->one(FLERR,str);
    }
  }

  const int MAXLINE = 1024;  char line[MAXLINE],*ptr;
  int n, eof = 0;

  const int MAX_WORDS = 128;
  char * words[MAX_WORDS] = {0};
  int next_word = 0;

  int read_header = 0; // 0 = read data, 1 = read header
  while (1) {
    if (comm->me == 0) {
      ptr = fgets(line,MAXLINE,fp);
      if (ptr == NULL) {
        eof = 1;
        fclose(fp);
      } else n = strlen(line) + 1;
    }
    MPI_Bcast(&eof,1,MPI_INT,0,world);
    if (eof) break;
    MPI_Bcast(&n,1,MPI_INT,0,world);
    MPI_Bcast(line,n,MPI_CHAR,0,world);

    // strip comment, skip line if blank
    if ((ptr = strchr(line,'#'))) *ptr = '\0';

    char *word = strtok(line," \t\n\r\f");
    while (word) {
      if (next_word > MAX_WORDS) error->all(FLERR, "Too many words in line.");
      free(words[next_word]);
      words[next_word] = strdup(word);
      int is_header = isalpha(words[next_word][0]);
      if ((read_header == 0) && is_header) {
        file_process_line(next_word, words, arg + 3 - 1);
        if (next_word != 0) {
          words[0] = words[next_word];
          words[next_word] = NULL;
        }
        next_word = 0;
        read_header = 1;
      } else read_header = is_header;
      next_word += 1;
      word = strtok(NULL, " \t\n\r\f");
    }
  }
  file_process_line(next_word, words, arg + 3 - 1);

  n = atom->ntypes;
  for (int i = 1; i <= n; i++)
    for (int j = i; j <= n; j++)
      setflag[i][j] = 0;

  // set setflag i,j for type pairs where both are mapped to elements

  int count = 0;
  for (int i = 1; i <= n; i++)
    for (int j = i; j <= n; j++)
      if ((strcmp("NULL", arg[i + 3 - 1]) != 0) && (strcmp("NULL", arg[j + 3 - 1]) != 0)) {
        setflag[i][j] = 1;
        count++;
      }

  if (count == 0) error->all(FLERR,"Incorrect args for pair coefficients");
}

/* ---------------------------------------------------------------------- */

void PairSwGenIntel::file_process_line(int narg, char **arg, char **coeff_arg) {
  if (narg == 0) return;
  const char * SPLINE_PREFIX = "spline:";
  if (strncmp(arg[0], SPLINE_PREFIX, strlen(SPLINE_PREFIX)) == 0) {
    return;
  }
  int num_header = 0;
  while (isalpha(arg[num_header][0])) num_header += 1;
  if (2 == num_header) {
    for (int i_0 = 1; i_0 <= this->atom->ntypes; i_0++) {
      if (strcmp(coeff_arg[i_0], arg[0]) != 0) continue;
      for (int i_1 = 1; i_1 <= this->atom->ntypes; i_1++) {
        if (strcmp(coeff_arg[i_1], arg[1]) != 0) continue;
        this->param_A[i_0][i_1] = atof(arg[2]);
        this->param_a[i_0][i_1] = atof(arg[3]);
        this->param_B[i_0][i_1] = atof(arg[4]);
        this->param_p[i_0][i_1] = atof(arg[5]);
        this->param_q[i_0][i_1] = atof(arg[6]);
        this->param_epsilon[i_0][i_1] = atof(arg[7]);
        this->param_sigma[i_0][i_1] = atof(arg[8]);
        this->param_gamma[i_0][i_1] = atof(arg[9]);
      }
    }
    return;
  }
  if (3 == num_header) {
    for (int i_0 = 1; i_0 <= this->atom->ntypes; i_0++) {
      if (strcmp(coeff_arg[i_0], arg[0]) != 0) continue;
      for (int i_1 = 1; i_1 <= this->atom->ntypes; i_1++) {
        if (strcmp(coeff_arg[i_1], arg[1]) != 0) continue;
        for (int i_2 = 1; i_2 <= this->atom->ntypes; i_2++) {
          if (strcmp(coeff_arg[i_2], arg[2]) != 0) continue;
          this->param_lambda[i_0][i_1][i_2] = atof(arg[3]);
          this->param_cos_theta0[i_0][i_1][i_2] = atof(arg[4]);
        }
      }
    }
    return;
  }
  error->all(FLERR, "Could not process file input.");
}

/* ---------------------------------------------------------------------- */

template <class flt_t, class acc_t>
void PairSwGenIntel::pack_force_const(
    ForceConst<flt_t> &fc,
    IntelBuffers<flt_t,acc_t> *buffers
) {
  int tp1 = atom->ntypes + 1;
  fc.set_ntypes(tp1,memory,-1);
  buffers->set_ntypes(tp1, 1);
  flt_t **cutneighsq = buffers->get_cutneighsq();

  // Repeat cutsq calculation because done after call to init_style
  double cut, cutneigh;
  for (int i = 1; i <= atom->ntypes; i++) {
    for (int j = i; j <= atom->ntypes; j++) {
      if (setflag[i][j] != 0 || (setflag[i][i] != 0 && setflag[j][j] != 0)) {
        cut = init_one(i,j);
        cutneigh = cut + neighbor->skin;
        cutsq[i][j] = cutsq[j][i] = cut*cut;
        cutneighsq[i][j] = cutneighsq[j][i] = cutneigh * cutneigh;
      }
    }
  }
}
template <class flt_t>
void PairSwGenIntel::ForceConst<flt_t>::set_ntypes(
    const int ntypes,
    Memory *memory,
    const int cop
) {
}
