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

#include "pair_pot_gen_intel.h"
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

PairPotGenIntel::PairPotGenIntel(LAMMPS *lmp) : Pair(lmp) {
  manybody_flag = 1;
  one_coeff = 1;
  single_enable = 0;
  restartinfo = 0;
  ghostneigh = 1;
  suffix_flag |= Suffix::INTEL;
}

/* ---------------------------------------------------------------------- */

PairPotGenIntel::~PairPotGenIntel() {
  if (copymode) return;
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);

    memory->destroy(type_map);

    memory->destroy(cutghost);

    memory->destroy(param_A);
    memory->destroy(param_B);
    memory->destroy(param_lambda_1);
    memory->destroy(param_lambda_2);
    memory->destroy(param_beta);
    memory->destroy(param_n);
    memory->destroy(param_R);
    memory->destroy(param_D);
    memory->destroy(param_lambda_3);
    memory->destroy(param_mm);
    memory->destroy(param_gamma);
    memory->destroy(param_c);
    memory->destroy(param_d);
    memory->destroy(param_cos_theta_0);
    memory->destroy(outlined_param_0);
    memory->destroy(outlined_param_1);
    memory->destroy(outlined_param_2);
    memory->destroy(outlined_param_3);
    memory->destroy(outlined_param_4);
    memory->destroy(outlined_param_5);
  }
}

/* ---------------------------------------------------------------------- */

void PairPotGenIntel::compute(int eflag, int vflag) {
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
void PairPotGenIntel::compute(int eflag, int vflag,
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
  onetype = atom->ntypes == 1;
  if (onetype) {
    if (eflag) {
      eval<1, 1>(1, ovflag, buffers, fc, 0, offload_end);
      eval<1, 1>(0, ovflag, buffers, fc, host_start, inum);
    } else {
      eval<1, 0>(1, ovflag, buffers, fc, 0, offload_end);
      eval<1, 0>(0, ovflag, buffers, fc, host_start, inum);
    }
  } else {
    if (eflag) {
      eval<0, 1>(1, ovflag, buffers, fc, 0, offload_end);
      eval<0, 1>(0, ovflag, buffers, fc, host_start, inum);
    } else {
      eval<0, 0>(1, ovflag, buffers, fc, 0, offload_end);
      eval<0, 0>(0, ovflag, buffers, fc, host_start, inum);
    }
  }
}

/* ---------------------------------------------------------------------- */

template <int ONETYPE, int EFLAG, class flt_t, class acc_t>
void PairPotGenIntel::eval(const int offload, const int vflag,
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
  const int * _noalias const numneighhalf = buffers->get_atombin();
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


  __mmask8 t_0 = 0xFF;
      __m512d i_a_2 = _mm512_setzero_pd();
      __m512d i_mul_adj_175 = _mm512_mul_pd(_mm512_set1_pd(1), _mm512_recip_pd(_mm512_set1_pd(2)));
      for (int t_1 = iifrom; t_1 < iito; t_1 += 8) {
        __m512i i_i_3 = _mm512_add_epi32(_mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0), _mm512_set1_epi32(t_1));
        __mmask8 t_2 = _mm512_kand(0xFF, _mm512_cmplt_epi32_mask(i_i_3, _mm512_set1_epi32(iito)));
        __m512d i_px_4 = _mm512_mask_i32gather_pd(_mm512_undefined_pd(), t_2, _mm512_castsi512_si256(_mm512_slli_epi32(i_i_3, 3)), &x[0].x, 4);
        __m512d i_py_4 = _mm512_mask_i32gather_pd(_mm512_undefined_pd(), t_2, _mm512_castsi512_si256(_mm512_slli_epi32(i_i_3, 3)), &x[0].y, 4);
        __m512d i_pz_4 = _mm512_mask_i32gather_pd(_mm512_undefined_pd(), t_2, _mm512_castsi512_si256(_mm512_slli_epi32(i_i_3, 3)), &x[0].z, 4);
        __m512i i_ty_4 = _mm512_mask_i32gather_epi32(_mm512_undefined_epi32(), t_2, _mm512_slli_epi32(i_i_3, 3), &x[0].w, 4);
        __m512d i_a_5 = _mm512_setzero_pd();
        __m512d i_fx_178 = _mm512_setzero_pd();
        __m512d i_fy_178 = _mm512_setzero_pd();
        __m512d i_fz_178 = _mm512_setzero_pd();
        __m512i listnum_i_snlist_1047 = _mm512_setzero_epi32();
        __m512i listhalfnum_i_snlist_1047 = _mm512_setzero_epi32();
        __m512i listentry_i_snlist_1047[neighbor->oneatom];
        __m512i i_scounter_1048 = _mm512_setzero_epi32();
        __m512i t_3 = _mm512_mask_i32gather_epi32(_mm512_undefined_epi32(), t_2, i_i_3, numneigh, 4);
        for (;;) {
          __mmask8 t_4 = _mm512_kand(t_2, _mm512_cmplt_epi32_mask(i_scounter_1048, t_3));
          if (_mm512_kortestz(t_4, t_4)) break;
          __m512i i_satom_1049 = _mm512_mask_i32gather_epi32(_mm512_undefined_epi32(), t_4, _mm512_add_epi32(_mm512_mask_i32gather_epi32(_mm512_undefined_epi32(), t_4, i_i_3, cnumneigh, 4), i_scounter_1048), firstneigh, 4);
          __m512d i_dx_1049 = _mm512_sub_pd(i_px_4, _mm512_mask_i32gather_pd(_mm512_undefined_pd(), t_4, _mm512_castsi512_si256(_mm512_slli_epi32(i_satom_1049, 3)), &x[0].x, 4));
          __m512d i_dy_1049 = _mm512_sub_pd(i_py_4, _mm512_mask_i32gather_pd(_mm512_undefined_pd(), t_4, _mm512_castsi512_si256(_mm512_slli_epi32(i_satom_1049, 3)), &x[0].y, 4));
          __m512d i_dz_1049 = _mm512_sub_pd(i_pz_4, _mm512_mask_i32gather_pd(_mm512_undefined_pd(), t_4, _mm512_castsi512_si256(_mm512_slli_epi32(i_satom_1049, 3)), &x[0].z, 4));
          __mmask8 t_5 = _mm512_kand(_mm512_cmpnle_pd_mask(_mm512_add_pd(_mm512_mul_pd(i_dx_1049, i_dx_1049), _mm512_add_pd(_mm512_mul_pd(i_dy_1049, i_dy_1049), _mm512_mul_pd(i_dz_1049, i_dz_1049))), (ONETYPE ? _mm512_set1_pd(this->cutsq[1][1]) : _mm512_mask_i32gather_pd(_mm512_undefined_pd(), t_4, _mm512_castsi512_si256(_mm512_add_epi32(_mm512_mask_i32gather_epi32(_mm512_undefined_epi32(), t_4, _mm512_slli_epi32(i_satom_1049, 3), &x[0].w, 4), _mm512_mullo_epi32(_mm512_set1_epi32(tp1), _mm512_add_epi32(i_ty_4, _mm512_mullo_epi32(_mm512_set1_epi32(tp1), _mm512_setzero_epi32()))))), &this->cutsq[0][0], 8))), t_4);
          if (! _mm512_kortestz(t_5, t_5)) {
            i_scounter_1048 = _mm512_mask_add_epi32(i_scounter_1048, t_5, i_scounter_1048, _mm512_set1_epi32(1));
            continue;
          }
          _mm512_mask_i32scatter_epi32(listentry_i_snlist_1047, t_4, _mm512_add_epi32(_mm512_set_epi32(15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0), _mm512_slli_epi32(listnum_i_snlist_1047, 3)), i_satom_1049, 4);
          listnum_i_snlist_1047 = _mm512_mask_add_epi32(listnum_i_snlist_1047, t_4, listnum_i_snlist_1047, _mm512_set1_epi32(1));
          i_scounter_1048 = _mm512_add_epi32(i_scounter_1048, _mm512_set1_epi32(1));
        }
        __m512i i_i_6 = _mm512_setzero_epi32();
        __m512i t_6 = listnum_i_snlist_1047;
        for (;;) {
          __mmask8 t_7 = _mm512_kand(t_2, _mm512_cmplt_epi32_mask(i_i_6, t_6));
          if (_mm512_kortestz(t_7, t_7)) break;
          __m512i i_a_7 = _mm512_mask_i32gather_epi32(_mm512_undefined_epi32(), t_7, _mm512_add_epi32(_mm512_set_epi32(15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0), _mm512_slli_epi32(i_i_6, 3)), listentry_i_snlist_1047, 4);
          __m512d i_px_8 = _mm512_mask_i32gather_pd(_mm512_undefined_pd(), t_7, _mm512_castsi512_si256(_mm512_slli_epi32(i_a_7, 3)), &x[0].x, 4);
          __m512d i_py_8 = _mm512_mask_i32gather_pd(_mm512_undefined_pd(), t_7, _mm512_castsi512_si256(_mm512_slli_epi32(i_a_7, 3)), &x[0].y, 4);
          __m512d i_pz_8 = _mm512_mask_i32gather_pd(_mm512_undefined_pd(), t_7, _mm512_castsi512_si256(_mm512_slli_epi32(i_a_7, 3)), &x[0].z, 4);
          __m512i i_ty_8 = _mm512_mask_i32gather_epi32(_mm512_undefined_epi32(), t_7, _mm512_slli_epi32(i_a_7, 3), &x[0].w, 4);
          __m512d i_dx_9 = _mm512_sub_pd(i_px_4, i_px_8);
          __m512d i_dy_9 = _mm512_sub_pd(i_py_4, i_py_8);
          __m512d i_dz_9 = _mm512_sub_pd(i_pz_4, i_pz_8);
          __m512d i_rsq_9 = _mm512_add_pd(_mm512_add_pd(_mm512_mul_pd(i_dx_9, i_dx_9), _mm512_mul_pd(i_dy_9, i_dy_9)), _mm512_mul_pd(i_dz_9, i_dz_9));
          __m512d i_param_10 = (ONETYPE ? _mm512_set1_pd(this->param_R[1][1][1]) : _mm512_mask_i32gather_pd(_mm512_undefined_pd(), t_7, _mm512_castsi512_si256(_mm512_add_epi32(i_ty_8, _mm512_mullo_epi32(_mm512_set1_epi32(tp1), _mm512_add_epi32(i_ty_8, _mm512_mullo_epi32(_mm512_set1_epi32(tp1), _mm512_add_epi32(i_ty_4, _mm512_mullo_epi32(_mm512_set1_epi32(tp1), _mm512_setzero_epi32()))))))), &this->param_R[0][0][0], 8));
          __m512d i_param_11 = (ONETYPE ? _mm512_set1_pd(this->param_D[1][1][1]) : _mm512_mask_i32gather_pd(_mm512_undefined_pd(), t_7, _mm512_castsi512_si256(_mm512_add_epi32(i_ty_8, _mm512_mullo_epi32(_mm512_set1_epi32(tp1), _mm512_add_epi32(i_ty_8, _mm512_mullo_epi32(_mm512_set1_epi32(tp1), _mm512_add_epi32(i_ty_4, _mm512_mullo_epi32(_mm512_set1_epi32(tp1), _mm512_setzero_epi32()))))))), &this->param_D[0][0][0], 8));
          __m512d i_v_12 = _mm512_add_pd(i_param_10, i_param_11);
          __mmask8 t_8 = _mm512_kand(_mm512_cmpnle_pd_mask(i_rsq_9, _mm512_mul_pd(i_v_12, i_v_12)), t_7);
          if (! _mm512_kortestz(t_8, t_8)) {
            i_i_6 = _mm512_mask_add_epi32(i_i_6, t_8, i_i_6, _mm512_set1_epi32(1));
            continue;
          }
          __m512d i_r_13 = _mm512_sqrt_pd(i_rsq_9);
          __m512d i_v_14;
          __m512d i_v_17 = _mm512_sub_pd(i_param_10, i_param_11);
          __m512d i_param_40 = (ONETYPE ? _mm512_set1_pd(this->param_A[1][1]) : _mm512_mask_i32gather_pd(_mm512_undefined_pd(), t_7, _mm512_castsi512_si256(_mm512_add_epi32(i_ty_8, _mm512_mullo_epi32(_mm512_set1_epi32(tp1), _mm512_add_epi32(i_ty_4, _mm512_mullo_epi32(_mm512_set1_epi32(tp1), _mm512_setzero_epi32()))))), &this->param_A[0][0], 8));
          __m512d i_param_41 = (ONETYPE ? _mm512_set1_pd(this->param_lambda_1[1][1]) : _mm512_mask_i32gather_pd(_mm512_undefined_pd(), t_7, _mm512_castsi512_si256(_mm512_add_epi32(i_ty_8, _mm512_mullo_epi32(_mm512_set1_epi32(tp1), _mm512_add_epi32(i_ty_4, _mm512_mullo_epi32(_mm512_set1_epi32(tp1), _mm512_setzero_epi32()))))), &this->param_lambda_1[0][0], 8));
          __m512d i_bultin_44 = _mm512_exp_pd(_mm512_sub_pd(_mm512_setzero_pd(), _mm512_mul_pd(i_param_41, i_r_13)));
          __m512d i_mul_val_45 = _mm512_mul_pd(i_bultin_44, i_param_40);
          __m512d i_param_46 = (ONETYPE ? _mm512_set1_pd(this->param_beta[1][1]) : _mm512_mask_i32gather_pd(_mm512_undefined_pd(), t_7, _mm512_castsi512_si256(_mm512_add_epi32(i_ty_8, _mm512_mullo_epi32(_mm512_set1_epi32(tp1), _mm512_add_epi32(i_ty_4, _mm512_mullo_epi32(_mm512_set1_epi32(tp1), _mm512_setzero_epi32()))))), &this->param_beta[0][0], 8));
          __m512d i_recip_a_84 = _mm512_mul_pd(_mm512_set1_pd(1), _mm512_recip_pd(i_r_13));
          __m512d i_param_115 = (ONETYPE ? _mm512_set1_pd(this->param_n[1][1]) : _mm512_mask_i32gather_pd(_mm512_undefined_pd(), t_7, _mm512_castsi512_si256(_mm512_add_epi32(i_ty_8, _mm512_mullo_epi32(_mm512_set1_epi32(tp1), _mm512_add_epi32(i_ty_4, _mm512_mullo_epi32(_mm512_set1_epi32(tp1), _mm512_setzero_epi32()))))), &this->param_n[0][0], 8));
          __m512d i_param_164 = (ONETYPE ? _mm512_set1_pd(this->param_B[1][1]) : _mm512_mask_i32gather_pd(_mm512_undefined_pd(), t_7, _mm512_castsi512_si256(_mm512_add_epi32(i_ty_8, _mm512_mullo_epi32(_mm512_set1_epi32(tp1), _mm512_add_epi32(i_ty_4, _mm512_mullo_epi32(_mm512_set1_epi32(tp1), _mm512_setzero_epi32()))))), &this->param_B[0][0], 8));
          __m512d i_param_165 = (ONETYPE ? _mm512_set1_pd(this->param_lambda_2[1][1]) : _mm512_mask_i32gather_pd(_mm512_undefined_pd(), t_7, _mm512_castsi512_si256(_mm512_add_epi32(i_ty_8, _mm512_mullo_epi32(_mm512_set1_epi32(tp1), _mm512_add_epi32(i_ty_4, _mm512_mullo_epi32(_mm512_set1_epi32(tp1), _mm512_setzero_epi32()))))), &this->param_lambda_2[0][0], 8));
          __m512d i_bultin_168 = _mm512_exp_pd(_mm512_sub_pd(_mm512_setzero_pd(), _mm512_mul_pd(i_param_165, i_r_13)));
          __m512d i_v_170 = _mm512_sub_pd(_mm512_setzero_pd(), _mm512_mul_pd(i_bultin_168, i_param_164));
          __m512d i_a_223 = _mm512_setzero_pd();
          __m512i i_i_224 = _mm512_setzero_epi32();
          __m512i t_9 = listnum_i_snlist_1047;
          for (;;) {
            __mmask8 t_10 = _mm512_kand(t_7, _mm512_cmplt_epi32_mask(i_i_224, t_9));
            if (_mm512_kortestz(t_10, t_10)) break;
            __m512i i_a_225 = _mm512_mask_i32gather_epi32(_mm512_undefined_epi32(), t_10, _mm512_add_epi32(_mm512_set_epi32(15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0), _mm512_slli_epi32(i_i_224, 3)), listentry_i_snlist_1047, 4);
            __m512i i_ty_226 = _mm512_mask_i32gather_epi32(_mm512_undefined_epi32(), t_10, _mm512_slli_epi32(i_a_225, 3), &x[0].w, 4);
            __m512d i_dx_227 = _mm512_sub_pd(i_px_4, _mm512_mask_i32gather_pd(_mm512_undefined_pd(), t_10, _mm512_castsi512_si256(_mm512_slli_epi32(i_a_225, 3)), &x[0].x, 4));
            __m512d i_dy_227 = _mm512_sub_pd(i_py_4, _mm512_mask_i32gather_pd(_mm512_undefined_pd(), t_10, _mm512_castsi512_si256(_mm512_slli_epi32(i_a_225, 3)), &x[0].y, 4));
            __m512d i_dz_227 = _mm512_sub_pd(i_pz_4, _mm512_mask_i32gather_pd(_mm512_undefined_pd(), t_10, _mm512_castsi512_si256(_mm512_slli_epi32(i_a_225, 3)), &x[0].z, 4));
            __m512d i_rsq_227 = _mm512_add_pd(_mm512_add_pd(_mm512_mul_pd(i_dx_227, i_dx_227), _mm512_mul_pd(i_dy_227, i_dy_227)), _mm512_mul_pd(i_dz_227, i_dz_227));
            __m512d i_param_228 = (ONETYPE ? _mm512_set1_pd(this->param_R[1][1][1]) : _mm512_mask_i32gather_pd(_mm512_undefined_pd(), t_10, _mm512_castsi512_si256(_mm512_add_epi32(i_ty_226, _mm512_mullo_epi32(_mm512_set1_epi32(tp1), _mm512_add_epi32(i_ty_8, _mm512_mullo_epi32(_mm512_set1_epi32(tp1), _mm512_add_epi32(i_ty_4, _mm512_mullo_epi32(_mm512_set1_epi32(tp1), _mm512_setzero_epi32()))))))), &this->param_R[0][0][0], 8));
            __m512d i_param_229 = (ONETYPE ? _mm512_set1_pd(this->param_D[1][1][1]) : _mm512_mask_i32gather_pd(_mm512_undefined_pd(), t_10, _mm512_castsi512_si256(_mm512_add_epi32(i_ty_226, _mm512_mullo_epi32(_mm512_set1_epi32(tp1), _mm512_add_epi32(i_ty_8, _mm512_mullo_epi32(_mm512_set1_epi32(tp1), _mm512_add_epi32(i_ty_4, _mm512_mullo_epi32(_mm512_set1_epi32(tp1), _mm512_setzero_epi32()))))))), &this->param_D[0][0][0], 8));
            __m512d i_v_230 = _mm512_add_pd(i_param_228, i_param_229);
            __mmask8 t_11 = _mm512_kand(_mm512_cmpnle_pd_mask(i_rsq_227, _mm512_mul_pd(i_v_230, i_v_230)), t_10);
            if (! _mm512_kortestz(t_11, t_11)) {
              i_i_224 = _mm512_mask_add_epi32(i_i_224, t_11, i_i_224, _mm512_set1_epi32(1));
              continue;
            }
            __mmask8 t_12 = _mm512_kand(_mm512_cmpeq_epi32_mask(i_a_225, i_a_7), t_10);
            if (! _mm512_kortestz(t_12, t_12)) {
              i_i_224 = _mm512_mask_add_epi32(i_i_224, t_12, i_i_224, _mm512_set1_epi32(1));
              continue;
            }
            __m512d i_r_231 = _mm512_sqrt_pd(i_rsq_227);
            __m512d i_v_232;
            __m512d i_v_235 = _mm512_sub_pd(i_param_228, i_param_229);
            __mmask8 t_13 = _mm512_kand(t_10, _mm512_cmple_pd_mask(i_r_231, i_v_235));
            __mmask8 t_14 = _mm512_kandn(t_13, t_10);
            if (! _mm512_kortestz(t_13, t_13)) {
              i_v_232 = _mm512_mask_blend_pd(t_13, i_v_232, _mm512_set1_pd(1));
            }
            if(! _mm512_kortestz(t_14, t_14)) {
              __mmask8 t_15 = _mm512_kand(t_14, _mm512_kand(_mm512_cmplt_pd_mask(i_r_231, i_v_230), _mm512_cmplt_pd_mask(i_v_235, i_r_231)));
              __mmask8 t_16 = _mm512_kandn(t_15, t_14);
              if (! _mm512_kortestz(t_15, t_15)) {
                i_v_232 = _mm512_mask_blend_pd(t_15, i_v_232, _mm512_sub_pd(i_mul_adj_175, _mm512_mul_pd(i_mul_adj_175, _mm512_sin_pd(_mm512_mul_pd(_mm512_mul_pd(_mm512_mul_pd(_mm512_sub_pd(i_r_231, i_param_228), _mm512_set1_pd(M_PI)), i_mul_adj_175), _mm512_recip_pd(i_param_229))))));
              }
              if(! _mm512_kortestz(t_16, t_16)) {
                __mmask8 t_17 = _mm512_kand(t_16, _mm512_cmpnlt_pd_mask(i_r_231, i_v_230));
                __mmask8 t_18 = _mm512_kandn(t_17, t_16);
                if (! _mm512_kortestz(t_17, t_17)) {
                  i_v_232 = _mm512_mask_blend_pd(t_17, i_v_232, _mm512_setzero_pd());
                }
                if(! _mm512_kortestz(t_18, t_18)) {
                  error->one(FLERR,"else no expr");
                }
              }
            }
            __m512d i_param_262 = (ONETYPE ? _mm512_set1_pd(this->param_c[1][1][1]) : _mm512_mask_i32gather_pd(_mm512_undefined_pd(), t_10, _mm512_castsi512_si256(_mm512_add_epi32(i_ty_226, _mm512_mullo_epi32(_mm512_set1_epi32(tp1), _mm512_add_epi32(i_ty_8, _mm512_mullo_epi32(_mm512_set1_epi32(tp1), _mm512_add_epi32(i_ty_4, _mm512_mullo_epi32(_mm512_set1_epi32(tp1), _mm512_setzero_epi32()))))))), &this->param_c[0][0][0], 8));
            __m512d i_param_264 = (ONETYPE ? _mm512_set1_pd(this->param_d[1][1][1]) : _mm512_mask_i32gather_pd(_mm512_undefined_pd(), t_10, _mm512_castsi512_si256(_mm512_add_epi32(i_ty_226, _mm512_mullo_epi32(_mm512_set1_epi32(tp1), _mm512_add_epi32(i_ty_8, _mm512_mullo_epi32(_mm512_set1_epi32(tp1), _mm512_add_epi32(i_ty_4, _mm512_mullo_epi32(_mm512_set1_epi32(tp1), _mm512_setzero_epi32()))))))), &this->param_d[0][0][0], 8));
            __m512d i_v_274 = _mm512_sub_pd(_mm512_mul_pd(_mm512_mul_pd(_mm512_add_pd(_mm512_add_pd(_mm512_mul_pd(i_dx_227, i_dx_9), _mm512_mul_pd(i_dy_227, i_dy_9)), _mm512_mul_pd(i_dz_227, i_dz_9)), i_recip_a_84), _mm512_recip_pd(i_r_231)), (ONETYPE ? _mm512_set1_pd(this->param_cos_theta_0[1][1][1]) : _mm512_mask_i32gather_pd(_mm512_undefined_pd(), t_10, _mm512_castsi512_si256(_mm512_add_epi32(i_ty_226, _mm512_mullo_epi32(_mm512_set1_epi32(tp1), _mm512_add_epi32(i_ty_8, _mm512_mullo_epi32(_mm512_set1_epi32(tp1), _mm512_add_epi32(i_ty_4, _mm512_mullo_epi32(_mm512_set1_epi32(tp1), _mm512_setzero_epi32()))))))), &this->param_cos_theta_0[0][0][0], 8)));
            __m512d i_param_281 = (ONETYPE ? _mm512_set1_pd(this->param_lambda_3[1][1][1]) : _mm512_mask_i32gather_pd(_mm512_undefined_pd(), t_10, _mm512_castsi512_si256(_mm512_add_epi32(i_ty_226, _mm512_mullo_epi32(_mm512_set1_epi32(tp1), _mm512_add_epi32(i_ty_8, _mm512_mullo_epi32(_mm512_set1_epi32(tp1), _mm512_add_epi32(i_ty_4, _mm512_mullo_epi32(_mm512_set1_epi32(tp1), _mm512_setzero_epi32()))))))), &this->param_lambda_3[0][0][0], 8));
            __m512d i_v_285 = _mm512_sub_pd(i_r_13, i_r_231);
            i_a_223 = _mm512_mask_add_pd(i_a_223, t_10, i_a_223, _mm512_mul_pd(_mm512_mul_pd(_mm512_sub_pd((ONETYPE ? _mm512_set1_pd(this->outlined_param_0[1][1][1]) : _mm512_mask_i32gather_pd(_mm512_undefined_pd(), t_10, _mm512_castsi512_si256(_mm512_add_epi32(i_ty_226, _mm512_mullo_epi32(_mm512_set1_epi32(tp1), _mm512_add_epi32(i_ty_8, _mm512_mullo_epi32(_mm512_set1_epi32(tp1), _mm512_add_epi32(i_ty_4, _mm512_mullo_epi32(_mm512_set1_epi32(tp1), _mm512_setzero_epi32()))))))), &this->outlined_param_0[0][0][0], 8)), _mm512_mul_pd(_mm512_mul_pd(i_param_262, i_param_262), _mm512_recip_pd(_mm512_add_pd(_mm512_mul_pd(i_param_264, i_param_264), _mm512_mul_pd(i_v_274, i_v_274))))), (ONETYPE ? _mm512_set1_pd(this->param_gamma[1][1][1]) : _mm512_mask_i32gather_pd(_mm512_undefined_pd(), t_10, _mm512_castsi512_si256(_mm512_add_epi32(i_ty_226, _mm512_mullo_epi32(_mm512_set1_epi32(tp1), _mm512_add_epi32(i_ty_8, _mm512_mullo_epi32(_mm512_set1_epi32(tp1), _mm512_add_epi32(i_ty_4, _mm512_mullo_epi32(_mm512_set1_epi32(tp1), _mm512_setzero_epi32()))))))), &this->param_gamma[0][0][0], 8))), _mm512_mul_pd(_mm512_exp_pd(_mm512_mul_pd(_mm512_mul_pd(_mm512_mul_pd(i_param_281, i_param_281), _mm512_mul_pd(i_param_281, i_v_285)), _mm512_mul_pd(i_v_285, i_v_285))), i_v_232)));
            i_i_224 = _mm512_add_epi32(i_i_224, _mm512_set1_epi32(1));
          }
          __m512d i_mul_val_290 = _mm512_mul_pd(i_a_223, i_param_46);
          __m512d i_v_292;
          __mmask8 t_19 = _mm512_kand(t_7, _mm512_cmpnle_pd_mask(i_mul_val_290, (ONETYPE ? _mm512_set1_pd(this->outlined_param_3[1][1]) : _mm512_mask_i32gather_pd(_mm512_undefined_pd(), t_7, _mm512_castsi512_si256(_mm512_add_epi32(i_ty_8, _mm512_mullo_epi32(_mm512_set1_epi32(tp1), _mm512_add_epi32(i_ty_4, _mm512_mullo_epi32(_mm512_set1_epi32(tp1), _mm512_setzero_epi32()))))), &this->outlined_param_3[0][0], 8))));
          __mmask8 t_20 = _mm512_kandn(t_19, t_7);
          if (! _mm512_kortestz(t_19, t_19)) {
            i_v_292 = _mm512_mask_blend_pd(t_19, i_v_292, _mm512_mul_pd(_mm512_set1_pd(1), _mm512_recip_pd(_mm512_sqrt_pd(i_mul_val_290))));
          }
          if(! _mm512_kortestz(t_20, t_20)) {
            __mmask8 t_21 = _mm512_kand(t_20, _mm512_cmpnle_pd_mask(i_mul_val_290, (ONETYPE ? _mm512_set1_pd(this->outlined_param_1[1][1]) : _mm512_mask_i32gather_pd(_mm512_undefined_pd(), t_20, _mm512_castsi512_si256(_mm512_add_epi32(i_ty_8, _mm512_mullo_epi32(_mm512_set1_epi32(tp1), _mm512_add_epi32(i_ty_4, _mm512_mullo_epi32(_mm512_set1_epi32(tp1), _mm512_setzero_epi32()))))), &this->outlined_param_1[0][0], 8))));
            __mmask8 t_22 = _mm512_kandn(t_21, t_20);
            if (! _mm512_kortestz(t_21, t_21)) {
              i_v_292 = _mm512_mask_blend_pd(t_21, i_v_292, _mm512_mul_pd(_mm512_sub_pd(_mm512_set1_pd(1), _mm512_mul_pd(_mm512_pow_pd(i_mul_val_290, _mm512_sub_pd(_mm512_setzero_pd(), i_param_115)), _mm512_recip_pd(_mm512_mul_pd(_mm512_set1_pd(2), i_param_115)))), _mm512_recip_pd(_mm512_sqrt_pd(i_mul_val_290))));
            }
            if(! _mm512_kortestz(t_22, t_22)) {
              __mmask8 t_23 = _mm512_kand(t_22, _mm512_cmplt_pd_mask(i_mul_val_290, (ONETYPE ? _mm512_set1_pd(this->outlined_param_4[1][1]) : _mm512_mask_i32gather_pd(_mm512_undefined_pd(), t_22, _mm512_castsi512_si256(_mm512_add_epi32(i_ty_8, _mm512_mullo_epi32(_mm512_set1_epi32(tp1), _mm512_add_epi32(i_ty_4, _mm512_mullo_epi32(_mm512_set1_epi32(tp1), _mm512_setzero_epi32()))))), &this->outlined_param_4[0][0], 8))));
              __mmask8 t_24 = _mm512_kandn(t_23, t_22);
              if (! _mm512_kortestz(t_23, t_23)) {
                i_v_292 = _mm512_mask_blend_pd(t_23, i_v_292, _mm512_set1_pd(1));
              }
              if(! _mm512_kortestz(t_24, t_24)) {
                __mmask8 t_25 = _mm512_kand(t_24, _mm512_cmplt_pd_mask(i_mul_val_290, (ONETYPE ? _mm512_set1_pd(this->outlined_param_2[1][1]) : _mm512_mask_i32gather_pd(_mm512_undefined_pd(), t_24, _mm512_castsi512_si256(_mm512_add_epi32(i_ty_8, _mm512_mullo_epi32(_mm512_set1_epi32(tp1), _mm512_add_epi32(i_ty_4, _mm512_mullo_epi32(_mm512_set1_epi32(tp1), _mm512_setzero_epi32()))))), &this->outlined_param_2[0][0], 8))));
                __mmask8 t_26 = _mm512_kandn(t_25, t_24);
                if (! _mm512_kortestz(t_25, t_25)) {
                  i_v_292 = _mm512_mask_blend_pd(t_25, i_v_292, _mm512_sub_pd(_mm512_set1_pd(1), _mm512_mul_pd(_mm512_pow_pd(i_mul_val_290, i_param_115), _mm512_recip_pd(_mm512_mul_pd(_mm512_set1_pd(2), i_param_115)))));
                }
                if(! _mm512_kortestz(t_26, t_26)) {
                  __mmask8 t_27 = _mm512_kand(t_26, _mm512_cmple_pd_mask(i_mul_val_290, i_mul_val_290));
                  __mmask8 t_28 = _mm512_kandn(t_27, t_26);
                  if (! _mm512_kortestz(t_27, t_27)) {
                    i_v_292 = _mm512_mask_blend_pd(t_27, i_v_292, _mm512_pow_pd(_mm512_add_pd(_mm512_set1_pd(1), _mm512_pow_pd(i_mul_val_290, i_param_115)), _mm512_sub_pd(_mm512_setzero_pd(), _mm512_mul_pd(_mm512_set1_pd(1), _mm512_recip_pd(_mm512_mul_pd(_mm512_set1_pd(2), i_param_115))))));
                  }
                  if(! _mm512_kortestz(t_28, t_28)) {
                    error->one(FLERR,"else no expr");
                  }
                }
              }
            }
          }
          __m512d i_fun_acc_adj_351 = _mm512_setzero_pd();
          __mmask8 t_29 = _mm512_kand(t_7, _mm512_cmple_pd_mask(i_r_13, i_v_17));
          __mmask8 t_30 = _mm512_kandn(t_29, t_7);
          if (! _mm512_kortestz(t_29, t_29)) {
            i_v_14 = _mm512_mask_blend_pd(t_29, i_v_14, _mm512_set1_pd(1));
          }
          if(! _mm512_kortestz(t_30, t_30)) {
            __mmask8 t_31 = _mm512_kand(t_30, _mm512_kand(_mm512_cmplt_pd_mask(i_r_13, i_v_12), _mm512_cmplt_pd_mask(i_v_17, i_r_13)));
            __mmask8 t_32 = _mm512_kandn(t_31, t_30);
            if (! _mm512_kortestz(t_31, t_31)) {
              __m512d i_mul_val_29 = _mm512_mul_pd(_mm512_mul_pd(_mm512_mul_pd(_mm512_sub_pd(i_r_13, i_param_10), _mm512_set1_pd(M_PI)), i_mul_adj_175), _mm512_recip_pd(i_param_11));
              i_v_14 = _mm512_mask_blend_pd(t_31, i_v_14, _mm512_sub_pd(i_mul_adj_175, _mm512_mul_pd(i_mul_adj_175, _mm512_sin_pd(i_mul_val_29))));
              i_fun_acc_adj_351 = _mm512_mask_add_pd(i_fun_acc_adj_351, t_31, i_fun_acc_adj_351, _mm512_sub_pd(_mm512_setzero_pd(), _mm512_mul_pd(_mm512_mul_pd(_mm512_mul_pd(_mm512_add_pd(i_mul_val_45, _mm512_mul_pd(i_v_170, i_v_292)), _mm512_set1_pd(M_PI)), _mm512_mul_pd(_mm512_cos_pd(i_mul_val_29), i_mul_adj_175)), _mm512_recip_pd(_mm512_mul_pd(_mm512_set1_pd(4), i_param_11)))));
            }
            if(! _mm512_kortestz(t_32, t_32)) {
              __mmask8 t_33 = _mm512_kand(t_32, _mm512_cmpnlt_pd_mask(i_r_13, i_v_12));
              __mmask8 t_34 = _mm512_kandn(t_33, t_32);
              if (! _mm512_kortestz(t_33, t_33)) {
                i_v_14 = _mm512_mask_blend_pd(t_33, i_v_14, _mm512_setzero_pd());
              }
              if(! _mm512_kortestz(t_34, t_34)) {
                error->one(FLERR,"else no expr");
              }
            }
          }
          i_a_5 = _mm512_mask_add_pd(i_a_5, t_7, i_a_5, _mm512_mul_pd(_mm512_add_pd(i_mul_val_45, _mm512_mul_pd(i_v_170, i_v_292)), i_v_14));
          __m512d i_fx_182 = _mm512_setzero_pd();
          __m512d i_fy_182 = _mm512_setzero_pd();
          __m512d i_fz_182 = _mm512_setzero_pd();
          __m512d i_adj_by_r_401 = _mm512_mul_pd(i_fun_acc_adj_351, i_recip_a_84);
          i_fx_178 = _mm512_mask_add_pd(i_fx_178, t_7, i_fx_178, _mm512_sub_pd(_mm512_setzero_pd(), _mm512_mul_pd(i_adj_by_r_401, i_dx_9)));
          i_fy_178 = _mm512_mask_add_pd(i_fy_178, t_7, i_fy_178, _mm512_sub_pd(_mm512_setzero_pd(), _mm512_mul_pd(i_adj_by_r_401, i_dy_9)));
          i_fz_178 = _mm512_mask_add_pd(i_fz_178, t_7, i_fz_178, _mm512_sub_pd(_mm512_setzero_pd(), _mm512_mul_pd(i_adj_by_r_401, i_dz_9)));
          i_fx_182 = _mm512_mask_add_pd(i_fx_182, t_7, i_fx_182, _mm512_mul_pd(i_adj_by_r_401, i_dx_9));
          i_fy_182 = _mm512_mask_add_pd(i_fy_182, t_7, i_fy_182, _mm512_mul_pd(i_adj_by_r_401, i_dy_9));
          i_fz_182 = _mm512_mask_add_pd(i_fz_182, t_7, i_fz_182, _mm512_mul_pd(i_adj_by_r_401, i_dz_9));
          __m512d i_mul_adj_402 = _mm512_mul_pd(i_mul_adj_175, i_v_14);
          __m512d i_adj_by_r_422 = _mm512_sub_pd(_mm512_setzero_pd(), _mm512_mul_pd(_mm512_mul_pd(_mm512_mul_pd(i_bultin_44, i_mul_adj_402), _mm512_mul_pd(i_param_40, i_param_41)), i_recip_a_84));
          i_fx_178 = _mm512_mask_add_pd(i_fx_178, t_7, i_fx_178, _mm512_sub_pd(_mm512_setzero_pd(), _mm512_mul_pd(i_adj_by_r_422, i_dx_9)));
          i_fy_178 = _mm512_mask_add_pd(i_fy_178, t_7, i_fy_178, _mm512_sub_pd(_mm512_setzero_pd(), _mm512_mul_pd(i_adj_by_r_422, i_dy_9)));
          i_fz_178 = _mm512_mask_add_pd(i_fz_178, t_7, i_fz_178, _mm512_sub_pd(_mm512_setzero_pd(), _mm512_mul_pd(i_adj_by_r_422, i_dz_9)));
          i_fx_182 = _mm512_mask_add_pd(i_fx_182, t_7, i_fx_182, _mm512_mul_pd(i_adj_by_r_422, i_dx_9));
          i_fy_182 = _mm512_mask_add_pd(i_fy_182, t_7, i_fy_182, _mm512_mul_pd(i_adj_by_r_422, i_dy_9));
          i_fz_182 = _mm512_mask_add_pd(i_fz_182, t_7, i_fz_182, _mm512_mul_pd(i_adj_by_r_422, i_dz_9));
          __m512d i_mul_adj_548 = _mm512_mul_pd(i_mul_adj_402, i_v_170);
          __m512d i_fun_acc_adj_620 = _mm512_setzero_pd();
          __mmask8 t_35 = _mm512_kand(t_7, _mm512_cmpnle_pd_mask(i_mul_val_290, (ONETYPE ? _mm512_set1_pd(this->outlined_param_3[1][1]) : _mm512_mask_i32gather_pd(_mm512_undefined_pd(), t_7, _mm512_castsi512_si256(_mm512_add_epi32(i_ty_8, _mm512_mullo_epi32(_mm512_set1_epi32(tp1), _mm512_add_epi32(i_ty_4, _mm512_mullo_epi32(_mm512_set1_epi32(tp1), _mm512_setzero_epi32()))))), &this->outlined_param_3[0][0], 8))));
          __mmask8 t_36 = _mm512_kandn(t_35, t_7);
          if (! _mm512_kortestz(t_35, t_35)) {
            __m512d i_bultin_630 = _mm512_sqrt_pd(i_mul_val_290);
            __m512d i_recip_633 = _mm512_mul_pd(_mm512_set1_pd(1), _mm512_recip_pd(i_bultin_630));
            i_fun_acc_adj_620 = _mm512_mask_add_pd(i_fun_acc_adj_620, t_35, i_fun_acc_adj_620, _mm512_sub_pd(_mm512_setzero_pd(), _mm512_mul_pd(_mm512_mul_pd(_mm512_mul_pd(_mm512_set1_pd(0.5), i_mul_adj_548), _mm512_mul_pd(i_recip_633, i_recip_633)), _mm512_recip_pd(i_bultin_630))));
          }
          if(! _mm512_kortestz(t_36, t_36)) {
            __mmask8 t_37 = _mm512_kand(t_36, _mm512_cmpnle_pd_mask(i_mul_val_290, (ONETYPE ? _mm512_set1_pd(this->outlined_param_1[1][1]) : _mm512_mask_i32gather_pd(_mm512_undefined_pd(), t_36, _mm512_castsi512_si256(_mm512_add_epi32(i_ty_8, _mm512_mullo_epi32(_mm512_set1_epi32(tp1), _mm512_add_epi32(i_ty_4, _mm512_mullo_epi32(_mm512_set1_epi32(tp1), _mm512_setzero_epi32()))))), &this->outlined_param_1[0][0], 8))));
            __mmask8 t_38 = _mm512_kandn(t_37, t_36);
            if (! _mm512_kortestz(t_37, t_37)) {
              __m512d i_v_642 = _mm512_sub_pd(_mm512_setzero_pd(), i_param_115);
              __m512d i_mul_val_644 = _mm512_mul_pd(_mm512_set1_pd(2), i_param_115);
              __m512d i_bultin_648 = _mm512_sqrt_pd(i_mul_val_290);
              i_fun_acc_adj_620 = _mm512_mask_add_pd(i_fun_acc_adj_620, t_37, i_fun_acc_adj_620, _mm512_sub_pd(_mm512_setzero_pd(), _mm512_mul_pd(_mm512_mul_pd(_mm512_mul_pd(i_mul_adj_548, i_v_642), _mm512_pow_pd(i_mul_val_290, _mm512_sub_pd(i_v_642, _mm512_set1_pd(1)))), _mm512_recip_pd(_mm512_mul_pd(i_bultin_648, i_mul_val_644)))));
              __m512d i_recip_665 = _mm512_mul_pd(_mm512_set1_pd(1), _mm512_recip_pd(i_bultin_648));
              i_fun_acc_adj_620 = _mm512_mask_add_pd(i_fun_acc_adj_620, t_37, i_fun_acc_adj_620, _mm512_sub_pd(_mm512_setzero_pd(), _mm512_mul_pd(_mm512_mul_pd(_mm512_mul_pd(_mm512_mul_pd(_mm512_sub_pd(_mm512_set1_pd(1), _mm512_mul_pd(_mm512_pow_pd(i_mul_val_290, i_v_642), _mm512_recip_pd(i_mul_val_644))), _mm512_set1_pd(0.5)), _mm512_mul_pd(i_mul_adj_548, i_recip_665)), i_recip_665), _mm512_recip_pd(i_bultin_648))));
            }
            if(! _mm512_kortestz(t_38, t_38)) {
              __mmask8 t_39 = _mm512_kand(t_38, _mm512_cmplt_pd_mask(i_mul_val_290, (ONETYPE ? _mm512_set1_pd(this->outlined_param_4[1][1]) : _mm512_mask_i32gather_pd(_mm512_undefined_pd(), t_38, _mm512_castsi512_si256(_mm512_add_epi32(i_ty_8, _mm512_mullo_epi32(_mm512_set1_epi32(tp1), _mm512_add_epi32(i_ty_4, _mm512_mullo_epi32(_mm512_set1_epi32(tp1), _mm512_setzero_epi32()))))), &this->outlined_param_4[0][0], 8))));
              __mmask8 t_40 = _mm512_kandn(t_39, t_38);
              if (! _mm512_kortestz(t_39, t_39)) {
              }
              if(! _mm512_kortestz(t_40, t_40)) {
                __mmask8 t_41 = _mm512_kand(t_40, _mm512_cmplt_pd_mask(i_mul_val_290, (ONETYPE ? _mm512_set1_pd(this->outlined_param_2[1][1]) : _mm512_mask_i32gather_pd(_mm512_undefined_pd(), t_40, _mm512_castsi512_si256(_mm512_add_epi32(i_ty_8, _mm512_mullo_epi32(_mm512_set1_epi32(tp1), _mm512_add_epi32(i_ty_4, _mm512_mullo_epi32(_mm512_set1_epi32(tp1), _mm512_setzero_epi32()))))), &this->outlined_param_2[0][0], 8))));
                __mmask8 t_42 = _mm512_kandn(t_41, t_40);
                if (! _mm512_kortestz(t_41, t_41)) {
                  i_fun_acc_adj_620 = _mm512_mask_add_pd(i_fun_acc_adj_620, t_41, i_fun_acc_adj_620, _mm512_sub_pd(_mm512_setzero_pd(), _mm512_mul_pd(_mm512_mul_pd(i_mul_adj_548, _mm512_pow_pd(i_mul_val_290, _mm512_sub_pd(i_param_115, _mm512_set1_pd(1)))), _mm512_recip_pd(_mm512_set1_pd(2)))));
                }
                if(! _mm512_kortestz(t_42, t_42)) {
                  __mmask8 t_43 = _mm512_kand(t_42, _mm512_cmple_pd_mask(i_mul_val_290, i_mul_val_290));
                  __mmask8 t_44 = _mm512_kandn(t_43, t_42);
                  if (! _mm512_kortestz(t_43, t_43)) {
                    i_fun_acc_adj_620 = _mm512_mask_add_pd(i_fun_acc_adj_620, t_43, i_fun_acc_adj_620, _mm512_mul_pd(_mm512_mul_pd(_mm512_mul_pd(_mm512_sub_pd(_mm512_setzero_pd(), _mm512_mul_pd(_mm512_set1_pd(1), _mm512_recip_pd(_mm512_mul_pd(_mm512_set1_pd(2), i_param_115)))), i_mul_adj_548), _mm512_mul_pd(i_param_115, _mm512_pow_pd(_mm512_add_pd(_mm512_set1_pd(1), _mm512_pow_pd(i_mul_val_290, i_param_115)), (ONETYPE ? _mm512_set1_pd(this->outlined_param_5[1][1]) : _mm512_mask_i32gather_pd(_mm512_undefined_pd(), t_43, _mm512_castsi512_si256(_mm512_add_epi32(i_ty_8, _mm512_mullo_epi32(_mm512_set1_epi32(tp1), _mm512_add_epi32(i_ty_4, _mm512_mullo_epi32(_mm512_set1_epi32(tp1), _mm512_setzero_epi32()))))), &this->outlined_param_5[0][0], 8))))), _mm512_pow_pd(i_mul_val_290, _mm512_sub_pd(i_param_115, _mm512_set1_pd(1)))));
                  }
                  if(! _mm512_kortestz(t_44, t_44)) {
                    error->one(FLERR,"else no expr");
                  }
                }
              }
            }
          }
          __m512d i_mul_adj_779 = _mm512_mul_pd(i_fun_acc_adj_620, i_param_46);
          __m512i i_i_783 = _mm512_setzero_epi32();
          __m512i t_45 = listnum_i_snlist_1047;
          for (;;) {
            __mmask8 t_46 = _mm512_kand(t_7, _mm512_cmplt_epi32_mask(i_i_783, t_45));
            if (_mm512_kortestz(t_46, t_46)) break;
            __m512i i_a_784 = _mm512_mask_i32gather_epi32(_mm512_undefined_epi32(), t_46, _mm512_add_epi32(_mm512_set_epi32(15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0), _mm512_slli_epi32(i_i_783, 3)), listentry_i_snlist_1047, 4);
            __m512d i_px_785 = _mm512_mask_i32gather_pd(_mm512_undefined_pd(), t_46, _mm512_castsi512_si256(_mm512_slli_epi32(i_a_784, 3)), &x[0].x, 4);
            __m512d i_py_785 = _mm512_mask_i32gather_pd(_mm512_undefined_pd(), t_46, _mm512_castsi512_si256(_mm512_slli_epi32(i_a_784, 3)), &x[0].y, 4);
            __m512d i_pz_785 = _mm512_mask_i32gather_pd(_mm512_undefined_pd(), t_46, _mm512_castsi512_si256(_mm512_slli_epi32(i_a_784, 3)), &x[0].z, 4);
            __m512i i_ty_785 = _mm512_mask_i32gather_epi32(_mm512_undefined_epi32(), t_46, _mm512_slli_epi32(i_a_784, 3), &x[0].w, 4);
            __m512d i_dx_786 = _mm512_sub_pd(i_px_4, i_px_785);
            __m512d i_dy_786 = _mm512_sub_pd(i_py_4, i_py_785);
            __m512d i_dz_786 = _mm512_sub_pd(i_pz_4, i_pz_785);
            __m512d i_rsq_786 = _mm512_add_pd(_mm512_add_pd(_mm512_mul_pd(i_dx_786, i_dx_786), _mm512_mul_pd(i_dy_786, i_dy_786)), _mm512_mul_pd(i_dz_786, i_dz_786));
            __m512d i_param_787 = (ONETYPE ? _mm512_set1_pd(this->param_R[1][1][1]) : _mm512_mask_i32gather_pd(_mm512_undefined_pd(), t_46, _mm512_castsi512_si256(_mm512_add_epi32(i_ty_785, _mm512_mullo_epi32(_mm512_set1_epi32(tp1), _mm512_add_epi32(i_ty_8, _mm512_mullo_epi32(_mm512_set1_epi32(tp1), _mm512_add_epi32(i_ty_4, _mm512_mullo_epi32(_mm512_set1_epi32(tp1), _mm512_setzero_epi32()))))))), &this->param_R[0][0][0], 8));
            __m512d i_param_788 = (ONETYPE ? _mm512_set1_pd(this->param_D[1][1][1]) : _mm512_mask_i32gather_pd(_mm512_undefined_pd(), t_46, _mm512_castsi512_si256(_mm512_add_epi32(i_ty_785, _mm512_mullo_epi32(_mm512_set1_epi32(tp1), _mm512_add_epi32(i_ty_8, _mm512_mullo_epi32(_mm512_set1_epi32(tp1), _mm512_add_epi32(i_ty_4, _mm512_mullo_epi32(_mm512_set1_epi32(tp1), _mm512_setzero_epi32()))))))), &this->param_D[0][0][0], 8));
            __m512d i_v_789 = _mm512_add_pd(i_param_787, i_param_788);
            __mmask8 t_47 = _mm512_kand(_mm512_cmpnle_pd_mask(i_rsq_786, _mm512_mul_pd(i_v_789, i_v_789)), t_46);
            if (! _mm512_kortestz(t_47, t_47)) {
              i_i_783 = _mm512_mask_add_epi32(i_i_783, t_47, i_i_783, _mm512_set1_epi32(1));
              continue;
            }
            __mmask8 t_48 = _mm512_kand(_mm512_cmpeq_epi32_mask(i_a_784, i_a_7), t_46);
            if (! _mm512_kortestz(t_48, t_48)) {
              i_i_783 = _mm512_mask_add_epi32(i_i_783, t_48, i_i_783, _mm512_set1_epi32(1));
              continue;
            }
            __m512d i_r_790 = _mm512_sqrt_pd(i_rsq_786);
            __m512d i_v_791;
            __m512d i_v_794 = _mm512_sub_pd(i_param_787, i_param_788);
            __m512d i_recip_b_819 = _mm512_mul_pd(_mm512_set1_pd(1), _mm512_recip_pd(i_r_790));
            __m512d i_cos_819 = _mm512_mul_pd(_mm512_mul_pd(_mm512_add_pd(_mm512_add_pd(_mm512_mul_pd(i_dx_786, i_dx_9), _mm512_mul_pd(i_dy_786, i_dy_9)), _mm512_mul_pd(i_dz_786, i_dz_9)), i_recip_a_84), i_recip_b_819);
            __m512d i_param_820 = (ONETYPE ? _mm512_set1_pd(this->param_gamma[1][1][1]) : _mm512_mask_i32gather_pd(_mm512_undefined_pd(), t_46, _mm512_castsi512_si256(_mm512_add_epi32(i_ty_785, _mm512_mullo_epi32(_mm512_set1_epi32(tp1), _mm512_add_epi32(i_ty_8, _mm512_mullo_epi32(_mm512_set1_epi32(tp1), _mm512_add_epi32(i_ty_4, _mm512_mullo_epi32(_mm512_set1_epi32(tp1), _mm512_setzero_epi32()))))))), &this->param_gamma[0][0][0], 8));
            __m512d i_param_821 = (ONETYPE ? _mm512_set1_pd(this->param_c[1][1][1]) : _mm512_mask_i32gather_pd(_mm512_undefined_pd(), t_46, _mm512_castsi512_si256(_mm512_add_epi32(i_ty_785, _mm512_mullo_epi32(_mm512_set1_epi32(tp1), _mm512_add_epi32(i_ty_8, _mm512_mullo_epi32(_mm512_set1_epi32(tp1), _mm512_add_epi32(i_ty_4, _mm512_mullo_epi32(_mm512_set1_epi32(tp1), _mm512_setzero_epi32()))))))), &this->param_c[0][0][0], 8));
            __m512d i_param_823 = (ONETYPE ? _mm512_set1_pd(this->param_d[1][1][1]) : _mm512_mask_i32gather_pd(_mm512_undefined_pd(), t_46, _mm512_castsi512_si256(_mm512_add_epi32(i_ty_785, _mm512_mullo_epi32(_mm512_set1_epi32(tp1), _mm512_add_epi32(i_ty_8, _mm512_mullo_epi32(_mm512_set1_epi32(tp1), _mm512_add_epi32(i_ty_4, _mm512_mullo_epi32(_mm512_set1_epi32(tp1), _mm512_setzero_epi32()))))))), &this->param_d[0][0][0], 8));
            __m512d i_v_833 = _mm512_sub_pd(i_cos_819, (ONETYPE ? _mm512_set1_pd(this->param_cos_theta_0[1][1][1]) : _mm512_mask_i32gather_pd(_mm512_undefined_pd(), t_46, _mm512_castsi512_si256(_mm512_add_epi32(i_ty_785, _mm512_mullo_epi32(_mm512_set1_epi32(tp1), _mm512_add_epi32(i_ty_8, _mm512_mullo_epi32(_mm512_set1_epi32(tp1), _mm512_add_epi32(i_ty_4, _mm512_mullo_epi32(_mm512_set1_epi32(tp1), _mm512_setzero_epi32()))))))), &this->param_cos_theta_0[0][0][0], 8)));
            __m512d i_recip_837 = _mm512_mul_pd(_mm512_set1_pd(1), _mm512_recip_pd(_mm512_add_pd(_mm512_mul_pd(i_param_823, i_param_823), _mm512_mul_pd(i_v_833, i_v_833))));
            __m512d i_mul_val_836 = _mm512_mul_pd(_mm512_mul_pd(i_param_821, i_param_821), i_recip_837);
            __m512d i_mul_val_839 = _mm512_mul_pd(_mm512_sub_pd((ONETYPE ? _mm512_set1_pd(this->outlined_param_0[1][1][1]) : _mm512_mask_i32gather_pd(_mm512_undefined_pd(), t_46, _mm512_castsi512_si256(_mm512_add_epi32(i_ty_785, _mm512_mullo_epi32(_mm512_set1_epi32(tp1), _mm512_add_epi32(i_ty_8, _mm512_mullo_epi32(_mm512_set1_epi32(tp1), _mm512_add_epi32(i_ty_4, _mm512_mullo_epi32(_mm512_set1_epi32(tp1), _mm512_setzero_epi32()))))))), &this->outlined_param_0[0][0][0], 8)), i_mul_val_836), i_param_820);
            __m512d i_param_840 = (ONETYPE ? _mm512_set1_pd(this->param_lambda_3[1][1][1]) : _mm512_mask_i32gather_pd(_mm512_undefined_pd(), t_46, _mm512_castsi512_si256(_mm512_add_epi32(i_ty_785, _mm512_mullo_epi32(_mm512_set1_epi32(tp1), _mm512_add_epi32(i_ty_8, _mm512_mullo_epi32(_mm512_set1_epi32(tp1), _mm512_add_epi32(i_ty_4, _mm512_mullo_epi32(_mm512_set1_epi32(tp1), _mm512_setzero_epi32()))))))), &this->param_lambda_3[0][0][0], 8));
            __m512d i_v_841 = _mm512_mul_pd(_mm512_mul_pd(i_param_840, i_param_840), i_param_840);
            __m512d i_v_844 = _mm512_sub_pd(i_r_13, i_r_790);
            __m512d i_bultin_847 = _mm512_exp_pd(_mm512_mul_pd(_mm512_mul_pd(i_v_841, i_v_844), _mm512_mul_pd(i_v_844, i_v_844)));
            __m512d i_fun_acc_adj_850 = _mm512_setzero_pd();
            __mmask8 t_49 = _mm512_kand(t_46, _mm512_cmple_pd_mask(i_r_790, i_v_794));
            __mmask8 t_50 = _mm512_kandn(t_49, t_46);
            if (! _mm512_kortestz(t_49, t_49)) {
              i_v_791 = _mm512_mask_blend_pd(t_49, i_v_791, _mm512_set1_pd(1));
            }
            if(! _mm512_kortestz(t_50, t_50)) {
              __mmask8 t_51 = _mm512_kand(t_50, _mm512_kand(_mm512_cmplt_pd_mask(i_r_790, i_v_789), _mm512_cmplt_pd_mask(i_v_794, i_r_790)));
              __mmask8 t_52 = _mm512_kandn(t_51, t_50);
              if (! _mm512_kortestz(t_51, t_51)) {
                __m512d i_mul_val_806 = _mm512_mul_pd(_mm512_mul_pd(_mm512_mul_pd(_mm512_sub_pd(i_r_790, i_param_787), _mm512_set1_pd(M_PI)), i_mul_adj_175), _mm512_recip_pd(i_param_788));
                i_v_791 = _mm512_mask_blend_pd(t_51, i_v_791, _mm512_sub_pd(i_mul_adj_175, _mm512_mul_pd(i_mul_adj_175, _mm512_sin_pd(i_mul_val_806))));
                i_fun_acc_adj_850 = _mm512_mask_add_pd(i_fun_acc_adj_850, t_51, i_fun_acc_adj_850, _mm512_sub_pd(_mm512_setzero_pd(), _mm512_mul_pd(_mm512_mul_pd(_mm512_mul_pd(_mm512_mul_pd(_mm512_set1_pd(M_PI), _mm512_cos_pd(i_mul_val_806)), _mm512_mul_pd(i_bultin_847, i_mul_adj_779)), i_mul_val_839), _mm512_recip_pd(_mm512_mul_pd(_mm512_set1_pd(4), i_param_788)))));
              }
              if(! _mm512_kortestz(t_52, t_52)) {
                __mmask8 t_53 = _mm512_kand(t_52, _mm512_cmpnlt_pd_mask(i_r_790, i_v_789));
                __mmask8 t_54 = _mm512_kandn(t_53, t_52);
                if (! _mm512_kortestz(t_53, t_53)) {
                  i_v_791 = _mm512_mask_blend_pd(t_53, i_v_791, _mm512_setzero_pd());
                }
                if(! _mm512_kortestz(t_54, t_54)) {
                  error->one(FLERR,"else no expr");
                }
              }
            }
            __m512d i_dx_818 = _mm512_sub_pd(i_px_8, i_px_785);
            __m512d i_dy_818 = _mm512_sub_pd(i_py_8, i_py_785);
            __m512d i_dz_818 = _mm512_sub_pd(i_pz_8, i_pz_785);
            __m512d i_adj_by_r_900 = _mm512_mul_pd(i_fun_acc_adj_850, i_recip_b_819);
            i_fx_178 = _mm512_mask_add_pd(i_fx_178, t_46, i_fx_178, _mm512_sub_pd(_mm512_setzero_pd(), _mm512_mul_pd(i_adj_by_r_900, i_dx_786)));
            i_fy_178 = _mm512_mask_add_pd(i_fy_178, t_46, i_fy_178, _mm512_sub_pd(_mm512_setzero_pd(), _mm512_mul_pd(i_adj_by_r_900, i_dy_786)));
            i_fz_178 = _mm512_mask_add_pd(i_fz_178, t_46, i_fz_178, _mm512_sub_pd(_mm512_setzero_pd(), _mm512_mul_pd(i_adj_by_r_900, i_dz_786)));
            __m512d i_adj_acos_978 = _mm512_mul_pd(_mm512_mul_pd(_mm512_mul_pd(_mm512_set1_pd(2), i_bultin_847), _mm512_mul_pd(i_mul_adj_779, i_mul_val_836)), _mm512_mul_pd(_mm512_mul_pd(i_param_820, i_recip_837), _mm512_mul_pd(i_v_791, i_v_833)));
            __m512d i_adj_by_r_979 = _mm512_mul_pd(_mm512_mul_pd(_mm512_sub_pd(i_recip_b_819, _mm512_mul_pd(i_cos_819, i_recip_a_84)), i_adj_acos_978), i_recip_a_84);
            i_fx_178 = _mm512_mask_add_pd(i_fx_178, t_46, i_fx_178, _mm512_sub_pd(_mm512_setzero_pd(), _mm512_mul_pd(i_adj_by_r_979, i_dx_9)));
            i_fy_178 = _mm512_mask_add_pd(i_fy_178, t_46, i_fy_178, _mm512_sub_pd(_mm512_setzero_pd(), _mm512_mul_pd(i_adj_by_r_979, i_dy_9)));
            i_fz_178 = _mm512_mask_add_pd(i_fz_178, t_46, i_fz_178, _mm512_sub_pd(_mm512_setzero_pd(), _mm512_mul_pd(i_adj_by_r_979, i_dz_9)));
            i_fx_182 = _mm512_mask_add_pd(i_fx_182, t_46, i_fx_182, _mm512_mul_pd(i_adj_by_r_979, i_dx_9));
            i_fy_182 = _mm512_mask_add_pd(i_fy_182, t_46, i_fy_182, _mm512_mul_pd(i_adj_by_r_979, i_dy_9));
            i_fz_182 = _mm512_mask_add_pd(i_fz_182, t_46, i_fz_182, _mm512_mul_pd(i_adj_by_r_979, i_dz_9));
            __m512d i_adj_by_r_980 = _mm512_mul_pd(_mm512_mul_pd(_mm512_sub_pd(i_recip_a_84, _mm512_mul_pd(i_cos_819, i_recip_b_819)), i_adj_acos_978), i_recip_b_819);
            i_fx_178 = _mm512_mask_add_pd(i_fx_178, t_46, i_fx_178, _mm512_sub_pd(_mm512_setzero_pd(), _mm512_mul_pd(i_adj_by_r_980, i_dx_786)));
            i_fy_178 = _mm512_mask_add_pd(i_fy_178, t_46, i_fy_178, _mm512_sub_pd(_mm512_setzero_pd(), _mm512_mul_pd(i_adj_by_r_980, i_dy_786)));
            i_fz_178 = _mm512_mask_add_pd(i_fz_178, t_46, i_fz_178, _mm512_sub_pd(_mm512_setzero_pd(), _mm512_mul_pd(i_adj_by_r_980, i_dz_786)));
            __m512d i_adj_by_r_981 = _mm512_sub_pd(_mm512_setzero_pd(), _mm512_mul_pd(_mm512_mul_pd(i_adj_acos_978, i_recip_a_84), i_recip_b_819));
            i_fx_182 = _mm512_mask_add_pd(i_fx_182, t_46, i_fx_182, _mm512_sub_pd(_mm512_setzero_pd(), _mm512_mul_pd(i_adj_by_r_981, i_dx_818)));
            i_fy_182 = _mm512_mask_add_pd(i_fy_182, t_46, i_fy_182, _mm512_sub_pd(_mm512_setzero_pd(), _mm512_mul_pd(i_adj_by_r_981, i_dy_818)));
            i_fz_182 = _mm512_mask_add_pd(i_fz_182, t_46, i_fz_182, _mm512_sub_pd(_mm512_setzero_pd(), _mm512_mul_pd(i_adj_by_r_981, i_dz_818)));
            __m512d i_adj_1008 = _mm512_mul_pd(_mm512_mul_pd(_mm512_mul_pd(_mm512_set1_pd(3), i_bultin_847), _mm512_mul_pd(i_mul_adj_779, i_mul_val_839)), _mm512_mul_pd(_mm512_mul_pd(i_v_791, i_v_841), _mm512_mul_pd(i_v_844, i_v_844)));
            __m512d i_adj_by_r_1010 = _mm512_mul_pd(i_adj_1008, i_recip_a_84);
            i_fx_178 = _mm512_mask_add_pd(i_fx_178, t_46, i_fx_178, _mm512_sub_pd(_mm512_setzero_pd(), _mm512_mul_pd(i_adj_by_r_1010, i_dx_9)));
            i_fy_178 = _mm512_mask_add_pd(i_fy_178, t_46, i_fy_178, _mm512_sub_pd(_mm512_setzero_pd(), _mm512_mul_pd(i_adj_by_r_1010, i_dy_9)));
            i_fz_178 = _mm512_mask_add_pd(i_fz_178, t_46, i_fz_178, _mm512_sub_pd(_mm512_setzero_pd(), _mm512_mul_pd(i_adj_by_r_1010, i_dz_9)));
            i_fx_182 = _mm512_mask_add_pd(i_fx_182, t_46, i_fx_182, _mm512_mul_pd(i_adj_by_r_1010, i_dx_9));
            i_fy_182 = _mm512_mask_add_pd(i_fy_182, t_46, i_fy_182, _mm512_mul_pd(i_adj_by_r_1010, i_dy_9));
            i_fz_182 = _mm512_mask_add_pd(i_fz_182, t_46, i_fz_182, _mm512_mul_pd(i_adj_by_r_1010, i_dz_9));
            __m512d i_adj_by_r_1012 = _mm512_sub_pd(_mm512_setzero_pd(), _mm512_mul_pd(i_adj_1008, i_recip_b_819));
            i_fx_178 = _mm512_mask_add_pd(i_fx_178, t_46, i_fx_178, _mm512_sub_pd(_mm512_setzero_pd(), _mm512_mul_pd(i_adj_by_r_1012, i_dx_786)));
            i_fy_178 = _mm512_mask_add_pd(i_fy_178, t_46, i_fy_178, _mm512_sub_pd(_mm512_setzero_pd(), _mm512_mul_pd(i_adj_by_r_1012, i_dy_786)));
            i_fz_178 = _mm512_mask_add_pd(i_fz_178, t_46, i_fz_178, _mm512_sub_pd(_mm512_setzero_pd(), _mm512_mul_pd(i_adj_by_r_1012, i_dz_786)));
            __m256i t_56 = _mm512_castsi512_si256(_mm512_slli_epi32(i_a_784, 3));
            __m512d t_57 = _mm512_add_pd(_mm512_add_pd(_mm512_add_pd(_mm512_mul_pd(i_adj_by_r_900, i_dx_786), _mm512_mul_pd(i_adj_by_r_980, i_dx_786)), _mm512_mul_pd(i_adj_by_r_981, i_dx_818)), _mm512_mul_pd(i_adj_by_r_1012, i_dx_786));
            __m512d t_58 = _mm512_add_pd(_mm512_add_pd(_mm512_add_pd(_mm512_mul_pd(i_adj_by_r_900, i_dy_786), _mm512_mul_pd(i_adj_by_r_980, i_dy_786)), _mm512_mul_pd(i_adj_by_r_981, i_dy_818)), _mm512_mul_pd(i_adj_by_r_1012, i_dy_786));
            __m512d t_59 = _mm512_add_pd(_mm512_add_pd(_mm512_add_pd(_mm512_mul_pd(i_adj_by_r_900, i_dz_786), _mm512_mul_pd(i_adj_by_r_980, i_dz_786)), _mm512_mul_pd(i_adj_by_r_981, i_dz_818)), _mm512_mul_pd(i_adj_by_r_1012, i_dz_786));
            __mmask8 t_55 = t_46;
            while (t_55) {
              __m512i conf0 = _mm512_maskz_conflict_epi32(t_55, _mm512_castsi256_si512(t_56));
              __m512i conf1 = _mm512_broadcastmw_epi32(t_55);
              __m512i conf2  = _mm512_and_si512(conf0, conf1);
              __mmask8 conf3 = _mm512_mask_testn_epi32_mask(t_55, conf2, conf2);
              t_55 = t_55 & (~conf3);
              _mm512_mask_i32scatter_pd(&f[0].x, conf3, t_56, _mm512_add_pd(t_57, _mm512_mask_i32gather_pd(_mm512_undefined_pd(), conf3, t_56, &f[0].x, 4)), 4);
              _mm512_mask_i32scatter_pd(&f[0].y, conf3, t_56, _mm512_add_pd(t_58, _mm512_mask_i32gather_pd(_mm512_undefined_pd(), conf3, t_56, &f[0].y, 4)), 4);
              _mm512_mask_i32scatter_pd(&f[0].z, conf3, t_56, _mm512_add_pd(t_59, _mm512_mask_i32gather_pd(_mm512_undefined_pd(), conf3, t_56, &f[0].z, 4)), 4);
            }
            i_i_783 = _mm512_add_epi32(i_i_783, _mm512_set1_epi32(1));
          }
          __m512d i_adj_by_r_1039 = _mm512_mul_pd(_mm512_mul_pd(_mm512_mul_pd(i_bultin_168, i_mul_adj_402), _mm512_mul_pd(i_param_164, i_param_165)), _mm512_mul_pd(i_recip_a_84, i_v_292));
          i_fx_178 = _mm512_mask_add_pd(i_fx_178, t_7, i_fx_178, _mm512_sub_pd(_mm512_setzero_pd(), _mm512_mul_pd(i_adj_by_r_1039, i_dx_9)));
          i_fy_178 = _mm512_mask_add_pd(i_fy_178, t_7, i_fy_178, _mm512_sub_pd(_mm512_setzero_pd(), _mm512_mul_pd(i_adj_by_r_1039, i_dy_9)));
          i_fz_178 = _mm512_mask_add_pd(i_fz_178, t_7, i_fz_178, _mm512_sub_pd(_mm512_setzero_pd(), _mm512_mul_pd(i_adj_by_r_1039, i_dz_9)));
          i_fx_182 = _mm512_mask_add_pd(i_fx_182, t_7, i_fx_182, _mm512_mul_pd(i_adj_by_r_1039, i_dx_9));
          i_fy_182 = _mm512_mask_add_pd(i_fy_182, t_7, i_fy_182, _mm512_mul_pd(i_adj_by_r_1039, i_dy_9));
          i_fz_182 = _mm512_mask_add_pd(i_fz_182, t_7, i_fz_182, _mm512_mul_pd(i_adj_by_r_1039, i_dz_9));
          __m256i t_61 = _mm512_castsi512_si256(_mm512_slli_epi32(i_a_7, 3));
          __m512d t_62 = i_fx_182;
          __m512d t_63 = i_fy_182;
          __m512d t_64 = i_fz_182;
          __mmask8 t_60 = t_7;
          while (t_60) {
            __m512i conf0 = _mm512_maskz_conflict_epi32(t_60, _mm512_castsi256_si512(t_61));
            __m512i conf1 = _mm512_broadcastmw_epi32(t_60);
            __m512i conf2  = _mm512_and_si512(conf0, conf1);
            __mmask8 conf3 = _mm512_mask_testn_epi32_mask(t_60, conf2, conf2);
            t_60 = t_60 & (~conf3);
            _mm512_mask_i32scatter_pd(&f[0].x, conf3, t_61, _mm512_add_pd(t_62, _mm512_mask_i32gather_pd(_mm512_undefined_pd(), conf3, t_61, &f[0].x, 4)), 4);
            _mm512_mask_i32scatter_pd(&f[0].y, conf3, t_61, _mm512_add_pd(t_63, _mm512_mask_i32gather_pd(_mm512_undefined_pd(), conf3, t_61, &f[0].y, 4)), 4);
            _mm512_mask_i32scatter_pd(&f[0].z, conf3, t_61, _mm512_add_pd(t_64, _mm512_mask_i32gather_pd(_mm512_undefined_pd(), conf3, t_61, &f[0].z, 4)), 4);
          }
          i_i_6 = _mm512_add_epi32(i_i_6, _mm512_set1_epi32(1));
        }
        i_a_2 = _mm512_mask_add_pd(i_a_2, t_2, i_a_2, i_a_5);
        __m256i t_66 = _mm512_castsi512_si256(_mm512_slli_epi32(i_i_3, 3));
        __m512d t_67 = i_fx_178;
        __m512d t_68 = i_fy_178;
        __m512d t_69 = i_fz_178;
        __mmask8 t_65 = t_2;
        while (t_65) {
          __m512i conf0 = _mm512_maskz_conflict_epi32(t_65, _mm512_castsi256_si512(t_66));
          __m512i conf1 = _mm512_broadcastmw_epi32(t_65);
          __m512i conf2  = _mm512_and_si512(conf0, conf1);
          __mmask8 conf3 = _mm512_mask_testn_epi32_mask(t_65, conf2, conf2);
          t_65 = t_65 & (~conf3);
          _mm512_mask_i32scatter_pd(&f[0].x, conf3, t_66, _mm512_add_pd(t_67, _mm512_mask_i32gather_pd(_mm512_undefined_pd(), conf3, t_66, &f[0].x, 4)), 4);
          _mm512_mask_i32scatter_pd(&f[0].y, conf3, t_66, _mm512_add_pd(t_68, _mm512_mask_i32gather_pd(_mm512_undefined_pd(), conf3, t_66, &f[0].y, 4)), 4);
          _mm512_mask_i32scatter_pd(&f[0].z, conf3, t_66, _mm512_add_pd(t_69, _mm512_mask_i32gather_pd(_mm512_undefined_pd(), conf3, t_66, &f[0].z, 4)), 4);
        }
      }
      oevdwl += _mm512_reduce_add_pd(_mm512_mul_pd(i_a_2, i_mul_adj_175));


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

void PairPotGenIntel::init_style() {
  if (force->newton_pair == 0)
    error->all(FLERR, "Pair style pot/gen/intel requires atom IDs");
  if (atom->tag_enable == 0)
    error->all(FLERR, "Pair style pot/gen/intel requires newton pair on");
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

void PairPotGenIntel::allocate() {
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
  memory->create(param_B, n+1, n+1, "pair:B");
  memory->create(param_lambda_1, n+1, n+1, "pair:lambda_1");
  memory->create(param_lambda_2, n+1, n+1, "pair:lambda_2");
  memory->create(param_beta, n+1, n+1, "pair:beta");
  memory->create(param_n, n+1, n+1, "pair:n");
  memory->create(param_R, n+1, n+1, n+1, "pair:R");
  memory->create(param_D, n+1, n+1, n+1, "pair:D");
  memory->create(param_lambda_3, n+1, n+1, n+1, "pair:lambda_3");
  memory->create(param_mm, n+1, n+1, n+1, "pair:mm");
  memory->create(param_gamma, n+1, n+1, n+1, "pair:gamma");
  memory->create(param_c, n+1, n+1, n+1, "pair:c");
  memory->create(param_d, n+1, n+1, n+1, "pair:d");
  memory->create(param_cos_theta_0, n+1, n+1, n+1, "pair:cos_theta_0");
  memory->create(outlined_param_0, n+1, n+1, n+1, "pair:outlined:0");
  memory->create(outlined_param_1, n+1, n+1, "pair:outlined:1");
  memory->create(outlined_param_2, n+1, n+1, "pair:outlined:2");
  memory->create(outlined_param_3, n+1, n+1, "pair:outlined:3");
  memory->create(outlined_param_4, n+1, n+1, "pair:outlined:4");
  memory->create(outlined_param_5, n+1, n+1, "pair:outlined:5");
}

/* ---------------------------------------------------------------------- */

double PairPotGenIntel::init_one(int i, int j) {
  if (setflag[i][j] == 0) error->all(FLERR, "All pair coeffs are not set");
  cutghost[i][j] = cutmax;
  cutghost[j][i] = cutmax;
  return cutmax;
}

/* ---------------------------------------------------------------------- */

void PairPotGenIntel::settings(int narg, char **arg) {
  if (narg != 1) error->all(FLERR, "Illegal pair_style command");
  cutmax = atof(arg[0]);
}

/* ---------------------------------------------------------------------- */

void PairPotGenIntel::coeff(int narg, char **arg) {
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
  for (int i_ty_1 = 1; i_ty_1 <= atom->ntypes; i_ty_1++) {
    for (int i_ty_2 = 1; i_ty_2 <= atom->ntypes; i_ty_2++) {
      for (int i_ty_3 = 1; i_ty_3 <= atom->ntypes; i_ty_3++) {
        outlined_param_0[i_ty_1][i_ty_2][i_ty_3] = 1 + param_c[i_ty_1][i_ty_2][i_ty_3] * param_c[i_ty_1][i_ty_2][i_ty_3] / ((double) (param_d[i_ty_1][i_ty_2][i_ty_3] * param_d[i_ty_1][i_ty_2][i_ty_3]));
      }
    }
  }
  for (int i_ty_1 = 1; i_ty_1 <= atom->ntypes; i_ty_1++) {
    for (int i_ty_2 = 1; i_ty_2 <= atom->ntypes; i_ty_2++) {
      outlined_param_1[i_ty_1][i_ty_2] = pow(2 * param_n[i_ty_1][i_ty_2] / ((double) 100000000), 0 - 1 / ((double) param_n[i_ty_1][i_ty_2]));
    }
  }
  for (int i_ty_1 = 1; i_ty_1 <= atom->ntypes; i_ty_1++) {
    for (int i_ty_2 = 1; i_ty_2 <= atom->ntypes; i_ty_2++) {
      outlined_param_2[i_ty_1][i_ty_2] = pow(2 * param_n[i_ty_1][i_ty_2] / ((double) 100000000), 1 / ((double) param_n[i_ty_1][i_ty_2]));
    }
  }
  for (int i_ty_1 = 1; i_ty_1 <= atom->ntypes; i_ty_1++) {
    for (int i_ty_2 = 1; i_ty_2 <= atom->ntypes; i_ty_2++) {
      outlined_param_3[i_ty_1][i_ty_2] = pow(2 * param_n[i_ty_1][i_ty_2] * 1 / ((double) 10000000000000), 0 - 1 / ((double) param_n[i_ty_1][i_ty_2]));
    }
  }
  for (int i_ty_1 = 1; i_ty_1 <= atom->ntypes; i_ty_1++) {
    for (int i_ty_2 = 1; i_ty_2 <= atom->ntypes; i_ty_2++) {
      outlined_param_4[i_ty_1][i_ty_2] = pow(2 * param_n[i_ty_1][i_ty_2] * 1 / ((double) 10000000000000), 1 / ((double) param_n[i_ty_1][i_ty_2]));
    }
  }
  for (int i_ty_1 = 1; i_ty_1 <= atom->ntypes; i_ty_1++) {
    for (int i_ty_2 = 1; i_ty_2 <= atom->ntypes; i_ty_2++) {
      outlined_param_5[i_ty_1][i_ty_2] = 0 - 1 / ((double) (2 * param_n[i_ty_1][i_ty_2])) - 1;
    }
  }
}

/* ---------------------------------------------------------------------- */

void PairPotGenIntel::file_process_line(int narg, char **arg, char **coeff_arg) {
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
        this->param_B[i_0][i_1] = atof(arg[3]);
        this->param_lambda_1[i_0][i_1] = atof(arg[4]);
        this->param_lambda_2[i_0][i_1] = atof(arg[5]);
        this->param_beta[i_0][i_1] = atof(arg[6]);
        this->param_n[i_0][i_1] = atof(arg[7]);
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
          this->param_R[i_0][i_1][i_2] = atof(arg[3]);
          this->param_D[i_0][i_1][i_2] = atof(arg[4]);
          this->param_lambda_3[i_0][i_1][i_2] = atof(arg[5]);
          this->param_mm[i_0][i_1][i_2] = atof(arg[6]);
          this->param_gamma[i_0][i_1][i_2] = atof(arg[7]);
          this->param_c[i_0][i_1][i_2] = atof(arg[8]);
          this->param_d[i_0][i_1][i_2] = atof(arg[9]);
          this->param_cos_theta_0[i_0][i_1][i_2] = atof(arg[10]);
        }
      }
    }
    return;
  }
  error->all(FLERR, "Could not process file input.");
}

/* ---------------------------------------------------------------------- */

template <class flt_t, class acc_t>
void PairPotGenIntel::pack_force_const(
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
void PairPotGenIntel::ForceConst<flt_t>::set_ntypes(
    const int ntypes,
    Memory *memory,
    const int cop
) {
}
