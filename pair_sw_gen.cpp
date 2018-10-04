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

#include "pair_sw_gen.h"
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

PairSwGen::PairSwGen(LAMMPS *lmp) : Pair(lmp) {
  manybody_flag = 1;
  one_coeff = 1;
  single_enable = 0;
  restartinfo = 0;
  ghostneigh = 1;
}

/* ---------------------------------------------------------------------- */

PairSwGen::~PairSwGen() {
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

void PairSwGen::compute(int eflag, int vflag) {
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
  double i_a_98 = 0.0;
  for (int i_i_3 = 0; i_i_3 < nlocal; i_i_3++) {
    double i_px_4 = x[i_i_3][0];
    double i_py_4 = x[i_i_3][1];
    double i_pz_4 = x[i_i_3][2];
    int i_ty_4 = type[i_i_3];
    double i_fx_4 = 0.0;
    double i_fy_4 = 0.0;
    double i_fz_4 = 0.0;
    double i_a_5 = 0.0;
    int listnum_i_snlist_227 = 0;
    int listentry_i_snlist_227[neighbor->oneatom];
    for (int i_scounter_228 = 0; i_scounter_228 < numneigh[i_i_3]; i_scounter_228++) {
      int i_satom_229 = firstneigh[i_i_3][i_scounter_228];
      double i_dx_229 = i_px_4 - x[i_satom_229][0];
      double i_dy_229 = i_py_4 - x[i_satom_229][1];
      double i_dz_229 = i_pz_4 - x[i_satom_229][2];
      if ((i_dx_229 * i_dx_229 + i_dy_229 * i_dy_229 + i_dz_229 * i_dz_229) > cutsq[i_ty_4][type[i_satom_229]]) continue;
      listentry_i_snlist_227[listnum_i_snlist_227++] = i_satom_229;
    }
    for (int i_i_6 = 0; i_i_6 < listnum_i_snlist_227; i_i_6++) {
      int i_a_7 = listentry_i_snlist_227[i_i_6];
      int i = i_i_3;
      int j = i_a_7;
      tagint itag = tag[i];
      tagint jtag = tag[j];
      if (itag > jtag) {
        if ((itag+jtag) % 2 == 0) continue;
      } else if (itag < jtag) {
        if ((itag+jtag) % 2 == 1) continue;
      } else {
        if (x[j][2] <  x[i][2]) continue;
        if (x[j][2] == x[i][2] && x[j][1] <  x[i][1]) continue;
        if (x[j][2] == x[i][2] && x[j][1] == x[i][1] && x[j][0] < x[i][0]) continue;
      }
      int i_ty_8 = type[i_a_7];
      double i_dx_9 = i_px_4 - x[i_a_7][0];
      double i_dy_9 = i_py_4 - x[i_a_7][1];
      double i_dz_9 = i_pz_4 - x[i_a_7][2];
      double i_rsq_9 = i_dx_9 * i_dx_9 + i_dy_9 * i_dy_9 + i_dz_9 * i_dz_9;
      double i_param_11 = param_sigma[i_ty_4][i_ty_8];
      double i_mul_val_12 = param_a[i_ty_4][i_ty_8] * i_param_11;
      if (i_rsq_9 > (i_mul_val_12 * i_mul_val_12)) continue;
      double i_r_13 = sqrt(i_rsq_9);
      double i_param_15 = param_A[i_ty_4][i_ty_8];
      double i_param_16 = param_epsilon[i_ty_4][i_ty_8];
      double i_param_17 = param_B[i_ty_4][i_ty_8];
      double i_recip_20 = 1 / ((double) i_r_13);
      double i_mul_val_19 = i_param_11 * i_recip_20;
      double i_param_21 = param_p[i_ty_4][i_ty_8];
      double i_param_27 = param_q[i_ty_4][i_ty_8];
      double i_recip_36 = 1 / ((double) (i_r_13 - i_mul_val_12));
      double i_bultin_37 = exp(i_param_11 * i_recip_36);
      double i_mul_adj_40 = i_bultin_37 * i_param_15 * i_param_16;
      double i_builtin_adj_81 = (i_param_17 * pow(i_mul_val_19, i_param_21) - pow(i_mul_val_19, i_param_27)) * i_bultin_37 * i_param_15 * i_param_16;
      double i_adj_by_r_97 = (i_mul_adj_40 * i_param_11 * i_param_27 * i_recip_20 * i_recip_20 * pow(i_mul_val_19, (i_param_27 - 1)) - i_mul_adj_40 * i_param_11 * i_param_17 * i_param_21 * i_recip_20 * i_recip_20 * pow(i_mul_val_19, (i_param_21 - 1)) - i_builtin_adj_81 * i_param_11 * i_recip_36 * i_recip_36) * i_recip_20;
      i_fx_4 += - i_adj_by_r_97 * i_dx_9;
      i_fy_4 += - i_adj_by_r_97 * i_dy_9;
      i_fz_4 += - i_adj_by_r_97 * i_dz_9;
      i_a_5 += i_builtin_adj_81;
      f[i_a_7][0] += i_adj_by_r_97 * i_dx_9;
      f[i_a_7][1] += i_adj_by_r_97 * i_dy_9;
      f[i_a_7][2] += i_adj_by_r_97 * i_dz_9;
    }
    i_a_2 += i_a_5;
    f[i_i_3][0] += i_fx_4;
    f[i_i_3][1] += i_fy_4;
    f[i_i_3][2] += i_fz_4;
    double i_fx_100 = 0.0;
    double i_fy_100 = 0.0;
    double i_fz_100 = 0.0;
    double i_a_101 = 0.0;
    for (int i_i_102 = 0; i_i_102 < listnum_i_snlist_227; i_i_102++) {
      int i_a_103 = listentry_i_snlist_227[i_i_102];
      double i_px_104 = x[i_a_103][0];
      double i_py_104 = x[i_a_103][1];
      double i_pz_104 = x[i_a_103][2];
      int i_ty_104 = type[i_a_103];
      double i_dx_105 = i_px_4 - i_px_104;
      double i_dy_105 = i_py_4 - i_py_104;
      double i_dz_105 = i_pz_4 - i_pz_104;
      double i_rsq_105 = i_dx_105 * i_dx_105 + i_dy_105 * i_dy_105 + i_dz_105 * i_dz_105;
      double i_param_107 = param_sigma[i_ty_4][i_ty_104];
      double i_mul_val_108 = param_a[i_ty_4][i_ty_104] * i_param_107;
      if (i_rsq_105 > (i_mul_val_108 * i_mul_val_108)) continue;
      double i_fx_104 = 0.0;
      double i_fy_104 = 0.0;
      double i_fz_104 = 0.0;
      double i_a_109 = 0.0;
      double i_r_117 = sqrt(i_rsq_105);
      double i_recip_a_124 = 1 / ((double) i_r_117);
      double i_param_127 = param_epsilon[i_ty_4][i_ty_104];
      double i_param_132 = param_gamma[i_ty_4][i_ty_104];
      double i_recip_139 = 1 / ((double) (i_r_117 - i_mul_val_108));
      double i_bultin_140 = exp(i_param_107 * i_param_132 * i_recip_139);
      for (int i_i_110 = 1 + i_i_102; i_i_110 < listnum_i_snlist_227; i_i_110++) {
        int i_a_111 = listentry_i_snlist_227[i_i_110];
        double i_px_112 = x[i_a_111][0];
        double i_py_112 = x[i_a_111][1];
        double i_pz_112 = x[i_a_111][2];
        int i_ty_112 = type[i_a_111];
        double i_dx_113 = i_px_4 - i_px_112;
        double i_dy_113 = i_py_4 - i_py_112;
        double i_dz_113 = i_pz_4 - i_pz_112;
        double i_rsq_113 = i_dx_113 * i_dx_113 + i_dy_113 * i_dy_113 + i_dz_113 * i_dz_113;
        double i_param_115 = param_sigma[i_ty_4][i_ty_112];
        double i_mul_val_116 = param_a[i_ty_4][i_ty_112] * i_param_115;
        if (i_rsq_113 > (i_mul_val_116 * i_mul_val_116)) continue;
        double i_r_119 = sqrt(i_rsq_113);
        double i_dx_123 = i_px_104 - i_px_112;
        double i_dy_123 = i_py_104 - i_py_112;
        double i_dz_123 = i_pz_104 - i_pz_112;
        double i_recip_b_124 = 1 / ((double) i_r_119);
        double i_cos_124 = (i_dx_105 * i_dx_113 + i_dy_105 * i_dy_113 + i_dz_105 * i_dz_113) * i_recip_a_124 * i_recip_b_124;
        double i_param_126 = param_lambda[i_ty_4][i_ty_104][i_ty_112];
        double i_v_130 = i_cos_124 - param_cos_theta0[i_ty_4][i_ty_104][i_ty_112];
        double i_v_131 = i_v_130 * i_v_130;
        double i_param_141 = param_gamma[i_ty_4][i_ty_112];
        double i_recip_148 = 1 / ((double) (i_r_119 - i_mul_val_116));
        double i_bultin_149 = exp(i_param_115 * i_param_141 * i_recip_148);
        double i_builtin_adj_198 = i_bultin_140 * i_bultin_149 * i_param_126 * i_param_127 * i_v_131;
        double i_adj_by_r_216 = - i_bultin_140 * i_bultin_149 * i_param_107 * i_param_126 * i_param_127 * i_param_132 * i_recip_139 * i_recip_139 * i_recip_a_124 * i_v_131;
        i_fx_100 += - i_adj_by_r_216 * i_dx_105;
        i_fy_100 += - i_adj_by_r_216 * i_dy_105;
        i_fz_100 += - i_adj_by_r_216 * i_dz_105;
        i_fx_104 += i_adj_by_r_216 * i_dx_105;
        i_fy_104 += i_adj_by_r_216 * i_dy_105;
        i_fz_104 += i_adj_by_r_216 * i_dz_105;
        double i_adj_by_r_218 = - i_builtin_adj_198 * i_param_115 * i_param_141 * i_recip_148 * i_recip_148 * i_recip_b_124;
        i_fx_100 += - i_adj_by_r_218 * i_dx_113;
        i_fy_100 += - i_adj_by_r_218 * i_dy_113;
        i_fz_100 += - i_adj_by_r_218 * i_dz_113;
        double i_adj_acos_223 = 2 * i_bultin_140 * i_bultin_149 * i_param_126 * i_param_127 * i_v_130;
        double i_adj_by_r_224 = (i_recip_b_124 - i_cos_124 * i_recip_a_124) * i_adj_acos_223 * i_recip_a_124;
        i_fx_100 += - i_adj_by_r_224 * i_dx_105;
        i_fy_100 += - i_adj_by_r_224 * i_dy_105;
        i_fz_100 += - i_adj_by_r_224 * i_dz_105;
        i_fx_104 += i_adj_by_r_224 * i_dx_105;
        i_fy_104 += i_adj_by_r_224 * i_dy_105;
        i_fz_104 += i_adj_by_r_224 * i_dz_105;
        double i_adj_by_r_225 = (i_recip_a_124 - i_cos_124 * i_recip_b_124) * i_adj_acos_223 * i_recip_b_124;
        i_fx_100 += - i_adj_by_r_225 * i_dx_113;
        i_fy_100 += - i_adj_by_r_225 * i_dy_113;
        i_fz_100 += - i_adj_by_r_225 * i_dz_113;
        double i_adj_by_r_226 = - i_adj_acos_223 * i_recip_a_124 * i_recip_b_124;
        i_fx_104 += - i_adj_by_r_226 * i_dx_123;
        i_fy_104 += - i_adj_by_r_226 * i_dy_123;
        i_fz_104 += - i_adj_by_r_226 * i_dz_123;
        i_a_109 += i_builtin_adj_198;
        f[i_a_111][0] += i_adj_by_r_218 * i_dx_113 + i_adj_by_r_225 * i_dx_113 + i_adj_by_r_226 * i_dx_123;
        f[i_a_111][1] += i_adj_by_r_218 * i_dy_113 + i_adj_by_r_225 * i_dy_113 + i_adj_by_r_226 * i_dy_123;
        f[i_a_111][2] += i_adj_by_r_218 * i_dz_113 + i_adj_by_r_225 * i_dz_113 + i_adj_by_r_226 * i_dz_123;
      }
      i_a_101 += i_a_109;
      f[i_a_103][0] += i_fx_104;
      f[i_a_103][1] += i_fy_104;
      f[i_a_103][2] += i_fz_104;
    }
    i_a_98 += i_a_101;
    f[i_i_3][0] += i_fx_100;
    f[i_i_3][1] += i_fy_100;
    f[i_i_3][2] += i_fz_100;
  }
  eng_vdwl += i_a_2;
  eng_vdwl += i_a_98;

  if (vflag_fdotr) virial_fdotr_compute();
}

/* ---------------------------------------------------------------------- */

void PairSwGen::init_style() {
  if (force->newton_pair == 0)
    error->all(FLERR, "Pair style sw/gen requires atom IDs");
  if (atom->tag_enable == 0)
    error->all(FLERR, "Pair style sw/gen requires newton pair on");
  int irequest = neighbor->request(this, instance_me);
  neighbor->requests[irequest]->half = 0;
  neighbor->requests[irequest]->full = 1;
  neighbor->requests[irequest]->ghost = 1;
}

/* ---------------------------------------------------------------------- */

void PairSwGen::allocate() {
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

double PairSwGen::init_one(int i, int j) {
  if (setflag[i][j] == 0) error->all(FLERR, "All pair coeffs are not set");
  cutghost[i][j] = cutmax;
  cutghost[j][i] = cutmax;
  return cutmax;
}

/* ---------------------------------------------------------------------- */

void PairSwGen::settings(int narg, char **arg) {
  if (narg != 1) error->all(FLERR, "Illegal pair_style command");
  cutmax = atof(arg[0]);
}

/* ---------------------------------------------------------------------- */

void PairSwGen::coeff(int narg, char **arg) {
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

void PairSwGen::file_process_line(int narg, char **arg, char **coeff_arg) {
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
