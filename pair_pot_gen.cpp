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
  }
}

/* ---------------------------------------------------------------------- */

void PairPotGen::compute(int eflag, int vflag) {
  feenableexcept(FE_INVALID | FE_DIVBYZERO | FE_OVERFLOW);
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
  int *numneigh = list->numneigh;
  int **firstneigh = list->firstneigh;
  double i_a_2 = 0.0;
  double i_mul_adj_124 = 1 / ((double) 2);
  for (int i_i_3 = 0; i_i_3 < nlocal; i_i_3++) {
    double i_px_4 = x[i_i_3][0];
    double i_py_4 = x[i_i_3][1];
    double i_pz_4 = x[i_i_3][2];
    int i_ty_4 = type[i_i_3];
    double i_a_5 = 0.0;
    double i_fx_127 = 0.0;
    double i_fy_127 = 0.0;
    double i_fz_127 = 0.0;
    for (int i_i_6 = 0; i_i_6 < numneigh[i_i_3]; i_i_6++) {
      int i_a_7 = firstneigh[i_i_3][i_i_6];
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
      double i_param_36 = param_A[i_ty_4][i_ty_8];
      double i_param_37 = param_lambda_1[i_ty_4][i_ty_8];
      double i_bultin_40 = exp(0 - i_param_37 * i_r_13);
      double i_param_42 = param_beta[i_ty_4][i_ty_8];
      double i_param_105 = param_n[i_ty_4][i_ty_8];
      double i_v_110 = 0 - 1 / ((double) (2 * i_param_105));
      double i_param_113 = param_B[i_ty_4][i_ty_8];
      double i_param_114 = param_lambda_2[i_ty_4][i_ty_8];
      double i_bultin_117 = exp(0 - i_param_114 * i_r_13);
      double i_v_119 = 0 - i_param_113 * i_bultin_117;
      double i_a_168 = 0.0;
      for (int i_i_169 = 0; i_i_169 < numneigh[i_i_3]; i_i_169++) {
        int i_a_170 = firstneigh[i_i_3][i_i_169];
        int i_ty_171 = type[i_a_170];
        double i_dx_172 = i_px_4 - x[i_a_170][0];
        double i_dy_172 = i_py_4 - x[i_a_170][1];
        double i_dz_172 = i_pz_4 - x[i_a_170][2];
        double i_rsq_172 = i_dx_172 * i_dx_172 + i_dy_172 * i_dy_172 + i_dz_172 * i_dz_172;
        double i_param_173 = param_R[i_ty_4][i_ty_8][i_ty_171];
        double i_param_174 = param_D[i_ty_4][i_ty_8][i_ty_171];
        double i_v_175 = i_param_173 + i_param_174;
        if (i_rsq_172 > (i_v_175 * i_v_175)) continue;
        if (i_a_170 == i_a_7) continue;
        double i_r_176 = sqrt(i_rsq_172);
        double i_v_177;
        double i_v_180 = i_param_173 - i_param_174;
        if (i_r_176 <= i_v_180) {
          i_v_177 = 1;
        } else {
          if ((i_r_176 < i_v_175) && (i_v_180 < i_r_176)) {
            i_v_177 = i_mul_adj_124 - sin((i_r_176 - i_param_173) * M_PI / ((double) (2 * i_param_174))) / ((double) 2);
          } else {
            if (i_r_176 >= i_v_175) {
              i_v_177 = 0;
            } else {
              error->one(FLERR,"else no expr");
              error->one(FLERR,"else no expr");
              error->one(FLERR,"else no expr");
              error->one(FLERR,"else no expr");
              error->one(FLERR,"else no expr");
            }
          }
        }
        double i_param_203 = param_c[i_ty_4][i_ty_8][i_ty_171];
        double i_v_204 = i_param_203 * i_param_203;
        double i_param_205 = param_d[i_ty_4][i_ty_8][i_ty_171];
        double i_v_206 = i_param_205 * i_param_205;
        double i_v_214 = (i_dx_9 * i_dx_172 + i_dy_9 * i_dy_172 + i_dz_9 * i_dz_172) / ((double) (i_r_13 * i_r_176)) - param_cos_theta_0[i_ty_4][i_ty_8][i_ty_171];
        double i_param_220 = param_lambda_3[i_ty_4][i_ty_8][i_ty_171];
        double i_v_224 = i_r_13 - i_r_176;
        i_a_168 += i_v_177 * param_gamma[i_ty_4][i_ty_8][i_ty_171] * (1 + i_v_204 / ((double) i_v_206) - i_v_204 / ((double) (i_v_206 + i_v_214 * i_v_214))) * exp(i_param_220 * i_param_220 * i_param_220 * i_v_224 * i_v_224 * i_v_224);
      }
      double i_mul_val_229 = i_param_42 * i_a_168;
      double i_v_232 = 1 + pow(i_mul_val_229, i_param_105);
      double i_v_236 = pow(i_v_232, i_v_110);
      double i_v_246 = i_param_36 * i_bultin_40 + i_v_236 * i_v_119;
      double i_fun_acc_adj_249 = 0.0;
      if (i_r_13 <= i_v_17) {
        i_v_14 = 1;
      } else {
        if ((i_r_13 < i_v_12) && (i_v_17 < i_r_13)) {
          double i_mul_val_28 = (i_r_13 - i_param_10) * M_PI / ((double) (2 * i_param_11));
          i_v_14 = i_mul_adj_124 - sin(i_mul_val_28) / ((double) 2);
          i_fun_acc_adj_249 += - M_PI * cos(i_mul_val_28) * i_mul_adj_124 * i_v_246 / ((double) (4 * i_param_11));
        } else {
          if (i_r_13 >= i_v_12) {
            i_v_14 = 0;
          } else {
            error->one(FLERR,"else no expr");
            error->one(FLERR,"else no expr");
            error->one(FLERR,"else no expr");
          }
        }
      }
      i_a_5 += i_v_14 * i_v_246;
      double i_fx_131 = 0.0;
      double i_fy_131 = 0.0;
      double i_fz_131 = 0.0;
      double i_adj_by_r_289 = i_fun_acc_adj_249 / ((double) i_r_13);
      i_fx_127 += - i_adj_by_r_289 * i_dx_9;
      i_fy_127 += - i_adj_by_r_289 * i_dy_9;
      i_fz_127 += - i_adj_by_r_289 * i_dz_9;
      i_fx_131 += i_adj_by_r_289 * i_dx_9;
      i_fy_131 += i_adj_by_r_289 * i_dy_9;
      i_fz_131 += i_adj_by_r_289 * i_dz_9;
      double i_mul_adj_290 = i_mul_adj_124 * i_v_14;
      double i_adj_by_r_310 = - i_bultin_40 * i_mul_adj_290 * i_param_36 * i_param_37 / ((double) i_r_13);
      i_fx_127 += - i_adj_by_r_310 * i_dx_9;
      i_fy_127 += - i_adj_by_r_310 * i_dy_9;
      i_fz_127 += - i_adj_by_r_310 * i_dz_9;
      i_fx_131 += i_adj_by_r_310 * i_dx_9;
      i_fy_131 += i_adj_by_r_310 * i_dy_9;
      i_fz_131 += i_adj_by_r_310 * i_dz_9;
      double i_mul_adj_540 = i_mul_adj_290 * i_v_119 * pow(i_v_232, (i_v_110 - 1)) * i_v_110 * pow(i_mul_val_229, (i_param_105 - 1)) * i_param_105 * i_param_42;
      for (int i_i_544 = 0; i_i_544 < numneigh[i_i_3]; i_i_544++) {
        int i_a_545 = firstneigh[i_i_3][i_i_544];
        double i_px_546 = x[i_a_545][0];
        double i_py_546 = x[i_a_545][1];
        double i_pz_546 = x[i_a_545][2];
        int i_ty_546 = type[i_a_545];
        double i_dx_547 = i_px_4 - i_px_546;
        double i_dy_547 = i_py_4 - i_py_546;
        double i_dz_547 = i_pz_4 - i_pz_546;
        double i_rsq_547 = i_dx_547 * i_dx_547 + i_dy_547 * i_dy_547 + i_dz_547 * i_dz_547;
        double i_param_548 = param_R[i_ty_4][i_ty_8][i_ty_546];
        double i_param_549 = param_D[i_ty_4][i_ty_8][i_ty_546];
        double i_v_550 = i_param_548 + i_param_549;
        if (i_rsq_547 > (i_v_550 * i_v_550)) continue;
        if (i_a_545 == i_a_7) continue;
        double i_r_551 = sqrt(i_rsq_547);
        double i_v_552;
        double i_v_555 = i_param_548 - i_param_549;
        double i_cos_576 = (i_dx_9 * i_dx_547 + i_dy_9 * i_dy_547 + i_dz_9 * i_dz_547) / ((double) (i_r_13 * i_r_551));
        double i_param_577 = param_gamma[i_ty_4][i_ty_8][i_ty_546];
        double i_param_578 = param_c[i_ty_4][i_ty_8][i_ty_546];
        double i_v_579 = i_param_578 * i_param_578;
        double i_param_580 = param_d[i_ty_4][i_ty_8][i_ty_546];
        double i_v_581 = i_param_580 * i_param_580;
        double i_v_589 = i_cos_576 - param_cos_theta_0[i_ty_4][i_ty_8][i_ty_546];
        double i_v_591 = i_v_581 + i_v_589 * i_v_589;
        double i_mul_val_594 = i_param_577 * (1 + i_v_579 / ((double) i_v_581) - i_v_579 / ((double) i_v_591));
        double i_param_595 = param_lambda_3[i_ty_4][i_ty_8][i_ty_546];
        double i_v_596 = i_param_595 * i_param_595 * i_param_595;
        double i_v_599 = i_r_13 - i_r_551;
        double i_bultin_602 = exp(i_v_596 * i_v_599 * i_v_599 * i_v_599);
        double i_fun_acc_adj_605 = 0.0;
        if (i_r_551 <= i_v_555) {
          i_v_552 = 1;
        } else {
          if ((i_r_551 < i_v_550) && (i_v_555 < i_r_551)) {
            double i_mul_val_566 = (i_r_551 - i_param_548) * M_PI / ((double) (2 * i_param_549));
            i_v_552 = i_mul_adj_124 - sin(i_mul_val_566) / ((double) 2);
            i_fun_acc_adj_605 += - M_PI * cos(i_mul_val_566) * i_bultin_602 * i_mul_adj_540 * i_mul_val_594 / ((double) (4 * i_param_549));
          } else {
            if (i_r_551 >= i_v_550) {
              i_v_552 = 0;
            } else {
              error->one(FLERR,"else no expr");
              error->one(FLERR,"else no expr");
            }
          }
        }
        double i_dx_575 = i_px_8 - i_px_546;
        double i_dy_575 = i_py_8 - i_py_546;
        double i_dz_575 = i_pz_8 - i_pz_546;
        double i_adj_by_r_645 = i_fun_acc_adj_605 / ((double) i_r_551);
        i_fx_127 += - i_adj_by_r_645 * i_dx_547;
        i_fy_127 += - i_adj_by_r_645 * i_dy_547;
        i_fz_127 += - i_adj_by_r_645 * i_dz_547;
        double i_adj_acos_719 = 2 * i_bultin_602 * i_mul_adj_540 * i_param_577 * i_v_552 * i_v_579 * i_v_589 / ((double) (i_v_591 * i_v_591));
        double i_adj_by_r_720 = (1 / ((double) i_r_551) - i_cos_576 / ((double) i_r_13)) * i_adj_acos_719 / ((double) i_r_13);
        i_fx_127 += - i_adj_by_r_720 * i_dx_9;
        i_fy_127 += - i_adj_by_r_720 * i_dy_9;
        i_fz_127 += - i_adj_by_r_720 * i_dz_9;
        i_fx_131 += i_adj_by_r_720 * i_dx_9;
        i_fy_131 += i_adj_by_r_720 * i_dy_9;
        i_fz_131 += i_adj_by_r_720 * i_dz_9;
        double i_adj_by_r_721 = (1 / ((double) i_r_13) - i_cos_576 / ((double) i_r_551)) * i_adj_acos_719 / ((double) i_r_551);
        i_fx_127 += - i_adj_by_r_721 * i_dx_547;
        i_fy_127 += - i_adj_by_r_721 * i_dy_547;
        i_fz_127 += - i_adj_by_r_721 * i_dz_547;
        double i_adj_by_r_722 = - i_adj_acos_719 / ((double) (i_r_13 * i_r_551));
        i_fx_131 += - i_adj_by_r_722 * i_dx_575;
        i_fy_131 += - i_adj_by_r_722 * i_dy_575;
        i_fz_131 += - i_adj_by_r_722 * i_dz_575;
        double i_adj_749 = i_mul_adj_540 * i_v_552 * i_mul_val_594 * i_bultin_602 * i_v_596 * i_v_599 * i_v_599 * 3;
        double i_adj_by_r_751 = i_adj_749 / ((double) i_r_13);
        i_fx_127 += - i_adj_by_r_751 * i_dx_9;
        i_fy_127 += - i_adj_by_r_751 * i_dy_9;
        i_fz_127 += - i_adj_by_r_751 * i_dz_9;
        i_fx_131 += i_adj_by_r_751 * i_dx_9;
        i_fy_131 += i_adj_by_r_751 * i_dy_9;
        i_fz_131 += i_adj_by_r_751 * i_dz_9;
        double i_adj_by_r_753 = - i_adj_749 / ((double) i_r_551);
        i_fx_127 += - i_adj_by_r_753 * i_dx_547;
        i_fy_127 += - i_adj_by_r_753 * i_dy_547;
        i_fz_127 += - i_adj_by_r_753 * i_dz_547;
        f[i_a_545][0] += i_adj_by_r_645 * i_dx_547 + i_adj_by_r_721 * i_dx_547 + i_adj_by_r_722 * i_dx_575 + i_adj_by_r_753 * i_dx_547;
        f[i_a_545][1] += i_adj_by_r_645 * i_dy_547 + i_adj_by_r_721 * i_dy_547 + i_adj_by_r_722 * i_dy_575 + i_adj_by_r_753 * i_dy_547;
        f[i_a_545][2] += i_adj_by_r_645 * i_dz_547 + i_adj_by_r_721 * i_dz_547 + i_adj_by_r_722 * i_dz_575 + i_adj_by_r_753 * i_dz_547;
      }
      double i_adj_by_r_780 = i_bultin_117 * i_mul_adj_290 * i_param_113 * i_param_114 * i_v_236 / ((double) i_r_13);
      i_fx_127 += - i_adj_by_r_780 * i_dx_9;
      i_fy_127 += - i_adj_by_r_780 * i_dy_9;
      i_fz_127 += - i_adj_by_r_780 * i_dz_9;
      i_fx_131 += i_adj_by_r_780 * i_dx_9;
      i_fy_131 += i_adj_by_r_780 * i_dy_9;
      i_fz_131 += i_adj_by_r_780 * i_dz_9;
      f[i_a_7][0] += i_fx_131;
      f[i_a_7][1] += i_fy_131;
      f[i_a_7][2] += i_fz_131;
    }
    i_a_2 += i_a_5;
    f[i_i_3][0] += i_fx_127;
    f[i_i_3][1] += i_fy_127;
    f[i_i_3][2] += i_fz_127;
  }
  eng_vdwl += i_a_2 / ((double) 2);

  if (vflag_fdotr) virial_fdotr_compute();
  fesetenv(FE_DFL_ENV);
}

/* ---------------------------------------------------------------------- */

void PairPotGen::init_style() {
  if (force->newton_pair == 0)
    error->all(FLERR, "Pair style pot/gen requires atom IDs");
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
