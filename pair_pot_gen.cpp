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

#include "pair_pot_gen.h"
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

/* ---------------------------------------------------------------------- */

PairPotGen::PairPotGen(LAMMPS *lmp) : Pair(lmp) {
  manybody_flag = 1;
  one_coeff = 1;
  single_enable = 0;
  restartinfo = 0;
  ghostneigh = 1;
}

/* ---------------------------------------------------------------------- */

PairPotGen::~PairPotGen() {
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

void PairPotGen::compute(int eflag, int vflag) {
  if (eflag || vflag) ev_setup(eflag,vflag);
  else evflag = vflag_fdotr = 0;

  double **x = atom->x;
  double **f = atom->f;
  double *q = atom->q;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  int newton_pair = force->newton_pair;
  int inum = list->inum;
  int *ilist = list->ilist;
  tagint *tag = atom->tag;
  int *numneigh = list->numneigh;
  int **firstneigh = list->firstneigh;
  double i_a_2 = 0.0;
  double i_mul_adj_175 = 1 / ((double) 2);
  for (int i_i_3 = 0; i_i_3 < nlocal; i_i_3++) {
    double i_px_4 = x[i_i_3][0];
    double i_py_4 = x[i_i_3][1];
    double i_pz_4 = x[i_i_3][2];
    int i_ty_4 = type[i_i_3];
    double i_a_5 = 0.0;
    double i_fx_178 = 0.0;
    double i_fy_178 = 0.0;
    double i_fz_178 = 0.0;
    int listnum_i_snlist_1047 = 0;
    int listentry_i_snlist_1047[neighbor->oneatom];
    for (int i_scounter_1048 = 0; i_scounter_1048 < numneigh[i_i_3]; i_scounter_1048++) {
      int i_satom_1049 = firstneigh[i_i_3][i_scounter_1048];
      double i_dx_1049 = i_px_4 - x[i_satom_1049][0];
      double i_dy_1049 = i_py_4 - x[i_satom_1049][1];
      double i_dz_1049 = i_pz_4 - x[i_satom_1049][2];
      if ((i_dx_1049 * i_dx_1049 + i_dy_1049 * i_dy_1049 + i_dz_1049 * i_dz_1049) > cutsq[i_ty_4][type[i_satom_1049]]) continue;
      listentry_i_snlist_1047[listnum_i_snlist_1047++] = i_satom_1049;
    }
    for (int i_i_6 = 0; i_i_6 < listnum_i_snlist_1047; i_i_6++) {
      int i_a_7 = listentry_i_snlist_1047[i_i_6];
      double i_px_8 = x[i_a_7][0];
      double i_py_8 = x[i_a_7][1];
      double i_pz_8 = x[i_a_7][2];
      int i_ty_8 = type[i_a_7];
      double i_dx_9 = i_px_4 - i_px_8;
      double i_dy_9 = i_py_4 - i_py_8;
      double i_dz_9 = i_pz_4 - i_pz_8;
      double i_rsq_9 = i_dx_9 * i_dx_9 + i_dy_9 * i_dy_9 + i_dz_9 * i_dz_9;
      double i_param_10 = param_R[i_ty_4][i_ty_8][i_ty_8];
      double i_param_11 = param_D[i_ty_4][i_ty_8][i_ty_8];
      double i_v_12 = i_param_10 + i_param_11;
      if (i_rsq_9 > (i_v_12 * i_v_12)) continue;
      double i_r_13 = sqrt(i_rsq_9);
      double i_v_14;
      double i_v_17 = i_param_10 - i_param_11;
      double i_param_40 = param_A[i_ty_4][i_ty_8];
      double i_param_41 = param_lambda_1[i_ty_4][i_ty_8];
      double i_bultin_44 = exp(0 - i_param_41 * i_r_13);
      double i_mul_val_45 = i_bultin_44 * i_param_40;
      double i_param_46 = param_beta[i_ty_4][i_ty_8];
      double i_recip_a_84 = 1 / ((double) i_r_13);
      double i_param_115 = param_n[i_ty_4][i_ty_8];
      double i_param_164 = param_B[i_ty_4][i_ty_8];
      double i_param_165 = param_lambda_2[i_ty_4][i_ty_8];
      double i_bultin_168 = exp(0 - i_param_165 * i_r_13);
      double i_v_170 = 0 - i_bultin_168 * i_param_164;
      double i_a_223 = 0.0;
      for (int i_i_224 = 0; i_i_224 < listnum_i_snlist_1047; i_i_224++) {
        int i_a_225 = listentry_i_snlist_1047[i_i_224];
        int i_ty_226 = type[i_a_225];
        double i_dx_227 = i_px_4 - x[i_a_225][0];
        double i_dy_227 = i_py_4 - x[i_a_225][1];
        double i_dz_227 = i_pz_4 - x[i_a_225][2];
        double i_rsq_227 = i_dx_227 * i_dx_227 + i_dy_227 * i_dy_227 + i_dz_227 * i_dz_227;
        double i_param_228 = param_R[i_ty_4][i_ty_8][i_ty_226];
        double i_param_229 = param_D[i_ty_4][i_ty_8][i_ty_226];
        double i_v_230 = i_param_228 + i_param_229;
        if (i_rsq_227 > (i_v_230 * i_v_230)) continue;
        if (i_a_225 == i_a_7) continue;
        double i_r_231 = sqrt(i_rsq_227);
        double i_v_232;
        double i_v_235 = i_param_228 - i_param_229;
        if (i_r_231 <= i_v_235) {
          i_v_232 = 1;
        } else {
          if ((i_r_231 < i_v_230) && (i_v_235 < i_r_231)) {
            i_v_232 = i_mul_adj_175 - i_mul_adj_175 * sin((i_r_231 - i_param_228) * M_PI * i_mul_adj_175 / ((double) i_param_229));
          } else {
            if (i_r_231 >= i_v_230) {
              i_v_232 = 0;
            } else {
              error->one(FLERR,"else no expr");
            }
          }
        }
        double i_param_262 = param_c[i_ty_4][i_ty_8][i_ty_226];
        double i_param_264 = param_d[i_ty_4][i_ty_8][i_ty_226];
        double i_v_274 = (i_dx_227 * i_dx_9 + i_dy_227 * i_dy_9 + i_dz_227 * i_dz_9) * i_recip_a_84 / ((double) i_r_231) - param_cos_theta_0[i_ty_4][i_ty_8][i_ty_226];
        double i_param_281 = param_lambda_3[i_ty_4][i_ty_8][i_ty_226];
        double i_v_285 = i_r_13 - i_r_231;
        i_a_223 += (outlined_param_0[i_ty_4][i_ty_8][i_ty_226] - i_param_262 * i_param_262 / ((double) (i_param_264 * i_param_264 + i_v_274 * i_v_274))) * param_gamma[i_ty_4][i_ty_8][i_ty_226] * exp(i_param_281 * i_param_281 * i_param_281 * i_v_285 * i_v_285 * i_v_285) * i_v_232;
      }
      double i_mul_val_290 = i_a_223 * i_param_46;
      double i_v_292;
      if (i_mul_val_290 > outlined_param_3[i_ty_4][i_ty_8]) {
        i_v_292 = 1 / ((double) sqrt(i_mul_val_290));
      } else {
        if (i_mul_val_290 > outlined_param_1[i_ty_4][i_ty_8]) {
          i_v_292 = (1 - pow(i_mul_val_290, (0 - i_param_115)) / ((double) (2 * i_param_115))) / ((double) sqrt(i_mul_val_290));
        } else {
          if (i_mul_val_290 < outlined_param_4[i_ty_4][i_ty_8]) {
            i_v_292 = 1;
          } else {
            if (i_mul_val_290 < outlined_param_2[i_ty_4][i_ty_8]) {
              i_v_292 = 1 - pow(i_mul_val_290, i_param_115) / ((double) (2 * i_param_115));
            } else {
              if (i_mul_val_290 <= i_mul_val_290) {
                i_v_292 = pow(1 + pow(i_mul_val_290, i_param_115), 0 - 1 / ((double) (2 * i_param_115)));
              } else {
                error->one(FLERR,"else no expr");
              }
            }
          }
        }
      }
      double i_fun_acc_adj_351 = 0.0;
      if (i_r_13 <= i_v_17) {
        i_v_14 = 1;
      } else {
        if ((i_r_13 < i_v_12) && (i_v_17 < i_r_13)) {
          double i_mul_val_29 = (i_r_13 - i_param_10) * M_PI * i_mul_adj_175 / ((double) i_param_11);
          i_v_14 = i_mul_adj_175 - i_mul_adj_175 * sin(i_mul_val_29);
          i_fun_acc_adj_351 += - (i_mul_val_45 + i_v_170 * i_v_292) * M_PI * cos(i_mul_val_29) * i_mul_adj_175 / ((double) (4 * i_param_11));
        } else {
          if (i_r_13 >= i_v_12) {
            i_v_14 = 0;
          } else {
            error->one(FLERR,"else no expr");
          }
        }
      }
      i_a_5 += (i_mul_val_45 + i_v_170 * i_v_292) * i_v_14;
      double i_fx_182 = 0.0;
      double i_fy_182 = 0.0;
      double i_fz_182 = 0.0;
      double i_adj_by_r_401 = i_fun_acc_adj_351 * i_recip_a_84;
      i_fx_178 += - i_adj_by_r_401 * i_dx_9;
      i_fy_178 += - i_adj_by_r_401 * i_dy_9;
      i_fz_178 += - i_adj_by_r_401 * i_dz_9;
      i_fx_182 += i_adj_by_r_401 * i_dx_9;
      i_fy_182 += i_adj_by_r_401 * i_dy_9;
      i_fz_182 += i_adj_by_r_401 * i_dz_9;
      double i_mul_adj_402 = i_mul_adj_175 * i_v_14;
      double i_adj_by_r_422 = - i_bultin_44 * i_mul_adj_402 * i_param_40 * i_param_41 * i_recip_a_84;
      i_fx_178 += - i_adj_by_r_422 * i_dx_9;
      i_fy_178 += - i_adj_by_r_422 * i_dy_9;
      i_fz_178 += - i_adj_by_r_422 * i_dz_9;
      i_fx_182 += i_adj_by_r_422 * i_dx_9;
      i_fy_182 += i_adj_by_r_422 * i_dy_9;
      i_fz_182 += i_adj_by_r_422 * i_dz_9;
      double i_mul_adj_548 = i_mul_adj_402 * i_v_170;
      double i_fun_acc_adj_620 = 0.0;
      if (i_mul_val_290 > outlined_param_3[i_ty_4][i_ty_8]) {
        double i_bultin_630 = sqrt(i_mul_val_290);
        double i_recip_633 = 1 / ((double) i_bultin_630);
        i_fun_acc_adj_620 += - 0.5 * i_mul_adj_548 * i_recip_633 * i_recip_633 / ((double) i_bultin_630);
      } else {
        if (i_mul_val_290 > outlined_param_1[i_ty_4][i_ty_8]) {
          double i_v_642 = 0 - i_param_115;
          double i_mul_val_644 = 2 * i_param_115;
          double i_bultin_648 = sqrt(i_mul_val_290);
          i_fun_acc_adj_620 += - i_mul_adj_548 * i_v_642 * pow(i_mul_val_290, (i_v_642 - 1)) / ((double) (i_bultin_648 * i_mul_val_644));
          double i_recip_665 = 1 / ((double) i_bultin_648);
          i_fun_acc_adj_620 += - (1 - pow(i_mul_val_290, i_v_642) / ((double) i_mul_val_644)) * 0.5 * i_mul_adj_548 * i_recip_665 * i_recip_665 / ((double) i_bultin_648);
        } else {
          if (i_mul_val_290 < outlined_param_4[i_ty_4][i_ty_8]) {
          } else {
            if (i_mul_val_290 < outlined_param_2[i_ty_4][i_ty_8]) {
              i_fun_acc_adj_620 += - i_mul_adj_548 * pow(i_mul_val_290, (i_param_115 - 1)) / ((double) 2);
            } else {
              if (i_mul_val_290 <= i_mul_val_290) {
                i_fun_acc_adj_620 += (0 - 1 / ((double) (2 * i_param_115))) * i_mul_adj_548 * i_param_115 * pow((1 + pow(i_mul_val_290, i_param_115)), outlined_param_5[i_ty_4][i_ty_8]) * pow(i_mul_val_290, (i_param_115 - 1));
              } else {
                error->one(FLERR,"else no expr");
              }
            }
          }
        }
      }
      double i_mul_adj_779 = i_fun_acc_adj_620 * i_param_46;
      for (int i_i_783 = 0; i_i_783 < listnum_i_snlist_1047; i_i_783++) {
        int i_a_784 = listentry_i_snlist_1047[i_i_783];
        double i_px_785 = x[i_a_784][0];
        double i_py_785 = x[i_a_784][1];
        double i_pz_785 = x[i_a_784][2];
        int i_ty_785 = type[i_a_784];
        double i_dx_786 = i_px_4 - i_px_785;
        double i_dy_786 = i_py_4 - i_py_785;
        double i_dz_786 = i_pz_4 - i_pz_785;
        double i_rsq_786 = i_dx_786 * i_dx_786 + i_dy_786 * i_dy_786 + i_dz_786 * i_dz_786;
        double i_param_787 = param_R[i_ty_4][i_ty_8][i_ty_785];
        double i_param_788 = param_D[i_ty_4][i_ty_8][i_ty_785];
        double i_v_789 = i_param_787 + i_param_788;
        if (i_rsq_786 > (i_v_789 * i_v_789)) continue;
        if (i_a_784 == i_a_7) continue;
        double i_r_790 = sqrt(i_rsq_786);
        double i_v_791;
        double i_v_794 = i_param_787 - i_param_788;
        double i_recip_b_819 = 1 / ((double) i_r_790);
        double i_cos_819 = (i_dx_786 * i_dx_9 + i_dy_786 * i_dy_9 + i_dz_786 * i_dz_9) * i_recip_a_84 * i_recip_b_819;
        double i_param_820 = param_gamma[i_ty_4][i_ty_8][i_ty_785];
        double i_param_821 = param_c[i_ty_4][i_ty_8][i_ty_785];
        double i_param_823 = param_d[i_ty_4][i_ty_8][i_ty_785];
        double i_v_833 = i_cos_819 - param_cos_theta_0[i_ty_4][i_ty_8][i_ty_785];
        double i_recip_837 = 1 / ((double) (i_param_823 * i_param_823 + i_v_833 * i_v_833));
        double i_mul_val_836 = i_param_821 * i_param_821 * i_recip_837;
        double i_mul_val_839 = (outlined_param_0[i_ty_4][i_ty_8][i_ty_785] - i_mul_val_836) * i_param_820;
        double i_param_840 = param_lambda_3[i_ty_4][i_ty_8][i_ty_785];
        double i_v_841 = i_param_840 * i_param_840 * i_param_840;
        double i_v_844 = i_r_13 - i_r_790;
        double i_bultin_847 = exp(i_v_841 * i_v_844 * i_v_844 * i_v_844);
        double i_fun_acc_adj_850 = 0.0;
        if (i_r_790 <= i_v_794) {
          i_v_791 = 1;
        } else {
          if ((i_r_790 < i_v_789) && (i_v_794 < i_r_790)) {
            double i_mul_val_806 = (i_r_790 - i_param_787) * M_PI * i_mul_adj_175 / ((double) i_param_788);
            i_v_791 = i_mul_adj_175 - i_mul_adj_175 * sin(i_mul_val_806);
            i_fun_acc_adj_850 += - M_PI * cos(i_mul_val_806) * i_bultin_847 * i_mul_adj_779 * i_mul_val_839 / ((double) (4 * i_param_788));
          } else {
            if (i_r_790 >= i_v_789) {
              i_v_791 = 0;
            } else {
              error->one(FLERR,"else no expr");
            }
          }
        }
        double i_dx_818 = i_px_8 - i_px_785;
        double i_dy_818 = i_py_8 - i_py_785;
        double i_dz_818 = i_pz_8 - i_pz_785;
        double i_adj_by_r_900 = i_fun_acc_adj_850 * i_recip_b_819;
        i_fx_178 += - i_adj_by_r_900 * i_dx_786;
        i_fy_178 += - i_adj_by_r_900 * i_dy_786;
        i_fz_178 += - i_adj_by_r_900 * i_dz_786;
        double i_adj_acos_978 = 2 * i_bultin_847 * i_mul_adj_779 * i_mul_val_836 * i_param_820 * i_recip_837 * i_v_791 * i_v_833;
        double i_adj_by_r_979 = (i_recip_b_819 - i_cos_819 * i_recip_a_84) * i_adj_acos_978 * i_recip_a_84;
        i_fx_178 += - i_adj_by_r_979 * i_dx_9;
        i_fy_178 += - i_adj_by_r_979 * i_dy_9;
        i_fz_178 += - i_adj_by_r_979 * i_dz_9;
        i_fx_182 += i_adj_by_r_979 * i_dx_9;
        i_fy_182 += i_adj_by_r_979 * i_dy_9;
        i_fz_182 += i_adj_by_r_979 * i_dz_9;
        double i_adj_by_r_980 = (i_recip_a_84 - i_cos_819 * i_recip_b_819) * i_adj_acos_978 * i_recip_b_819;
        i_fx_178 += - i_adj_by_r_980 * i_dx_786;
        i_fy_178 += - i_adj_by_r_980 * i_dy_786;
        i_fz_178 += - i_adj_by_r_980 * i_dz_786;
        double i_adj_by_r_981 = - i_adj_acos_978 * i_recip_a_84 * i_recip_b_819;
        i_fx_182 += - i_adj_by_r_981 * i_dx_818;
        i_fy_182 += - i_adj_by_r_981 * i_dy_818;
        i_fz_182 += - i_adj_by_r_981 * i_dz_818;
        double i_adj_1008 = 3 * i_bultin_847 * i_mul_adj_779 * i_mul_val_839 * i_v_791 * i_v_841 * i_v_844 * i_v_844;
        double i_adj_by_r_1010 = i_adj_1008 * i_recip_a_84;
        i_fx_178 += - i_adj_by_r_1010 * i_dx_9;
        i_fy_178 += - i_adj_by_r_1010 * i_dy_9;
        i_fz_178 += - i_adj_by_r_1010 * i_dz_9;
        i_fx_182 += i_adj_by_r_1010 * i_dx_9;
        i_fy_182 += i_adj_by_r_1010 * i_dy_9;
        i_fz_182 += i_adj_by_r_1010 * i_dz_9;
        double i_adj_by_r_1012 = - i_adj_1008 * i_recip_b_819;
        i_fx_178 += - i_adj_by_r_1012 * i_dx_786;
        i_fy_178 += - i_adj_by_r_1012 * i_dy_786;
        i_fz_178 += - i_adj_by_r_1012 * i_dz_786;
        f[i_a_784][0] += i_adj_by_r_900 * i_dx_786 + i_adj_by_r_980 * i_dx_786 + i_adj_by_r_981 * i_dx_818 + i_adj_by_r_1012 * i_dx_786;
        f[i_a_784][1] += i_adj_by_r_900 * i_dy_786 + i_adj_by_r_980 * i_dy_786 + i_adj_by_r_981 * i_dy_818 + i_adj_by_r_1012 * i_dy_786;
        f[i_a_784][2] += i_adj_by_r_900 * i_dz_786 + i_adj_by_r_980 * i_dz_786 + i_adj_by_r_981 * i_dz_818 + i_adj_by_r_1012 * i_dz_786;
      }
      double i_adj_by_r_1039 = i_bultin_168 * i_mul_adj_402 * i_param_164 * i_param_165 * i_recip_a_84 * i_v_292;
      i_fx_178 += - i_adj_by_r_1039 * i_dx_9;
      i_fy_178 += - i_adj_by_r_1039 * i_dy_9;
      i_fz_178 += - i_adj_by_r_1039 * i_dz_9;
      i_fx_182 += i_adj_by_r_1039 * i_dx_9;
      i_fy_182 += i_adj_by_r_1039 * i_dy_9;
      i_fz_182 += i_adj_by_r_1039 * i_dz_9;
      f[i_a_7][0] += i_fx_182;
      f[i_a_7][1] += i_fy_182;
      f[i_a_7][2] += i_fz_182;
    }
    i_a_2 += i_a_5;
    f[i_i_3][0] += i_fx_178;
    f[i_i_3][1] += i_fy_178;
    f[i_i_3][2] += i_fz_178;
  }
  eng_vdwl += i_a_2 * i_mul_adj_175;

  if (vflag_fdotr) virial_fdotr_compute();
}

/* ---------------------------------------------------------------------- */

void PairPotGen::init_style() {
  if (force->newton_pair == 0)
    error->all(FLERR, "Pair style pot/gen requires atom IDs");
  if (atom->tag_enable == 0)
    error->all(FLERR, "Pair style pot/gen requires newton pair on");
  int irequest = neighbor->request(this, instance_me);
  neighbor->requests[irequest]->half = 0;
  neighbor->requests[irequest]->full = 1;
  neighbor->requests[irequest]->ghost = 1;
}

/* ---------------------------------------------------------------------- */

void PairPotGen::allocate() {
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

double PairPotGen::init_one(int i, int j) {
  if (setflag[i][j] == 0) error->all(FLERR, "All pair coeffs are not set");
  cutghost[i][j] = cutmax;
  cutghost[j][i] = cutmax;
  return cutmax;
}

/* ---------------------------------------------------------------------- */

void PairPotGen::settings(int narg, char **arg) {
  if (narg != 1) error->all(FLERR, "Illegal pair_style command");
  cutmax = atof(arg[0]);
}

/* ---------------------------------------------------------------------- */

void PairPotGen::coeff(int narg, char **arg) {
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

void PairPotGen::file_process_line(int narg, char **arg, char **coeff_arg) {
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
