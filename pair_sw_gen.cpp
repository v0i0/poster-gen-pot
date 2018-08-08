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
  double i_mul_adj_36 = 1 / ((double) 2);
  double i_a_124 = 0.0;
  for (int i_i_3 = 0; i_i_3 < nlocal; i_i_3++) {
    double i_px_4 = x[i_i_3][0];
    double i_py_4 = x[i_i_3][1];
    double i_pz_4 = x[i_i_3][2];
    int i_ty_4 = type[i_i_3];
    double i_a_5 = 0.0;
    double i_fx_39 = 0.0;
    double i_fy_39 = 0.0;
    double i_fz_39 = 0.0;
    double i_a_127 = 0.0;
    double i_fx_176 = 0.0;
    double i_fy_176 = 0.0;
    double i_fz_176 = 0.0;
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
      double i_param_11 = param_sigma[i_ty_4][i_ty_8];
      double i_mul_val_12 = param_a[i_ty_4][i_ty_8] * i_param_11;
      if (i_rsq_9 > (i_mul_val_12 * i_mul_val_12)) continue;
      double i_r_13 = sqrt(i_rsq_9);
      double i_param_14 = param_A[i_ty_4][i_ty_8];
      double i_param_15 = param_epsilon[i_ty_4][i_ty_8];
      double i_param_16 = param_B[i_ty_4][i_ty_8];
      double i_mul_val_18 = i_param_11 / ((double) i_r_13);
      double i_param_19 = param_p[i_ty_4][i_ty_8];
      double i_param_24 = param_q[i_ty_4][i_ty_8];
      double i_v_26 = i_param_16 * pow(i_mul_val_18, i_param_19) - pow(i_mul_val_18, i_param_24);
      double i_v_31 = i_r_13 - i_mul_val_12;
      double i_bultin_33 = exp(i_param_11 / ((double) i_v_31));
      i_a_5 += i_param_14 * i_param_15 * i_v_26 * i_bultin_33;
      double i_mul_adj_72 = i_mul_adj_36 * i_param_14 * i_param_15 * i_bultin_33;
      double i_adj_by_r_121 = (i_mul_adj_72 * i_param_11 * i_param_24 * pow(i_mul_val_18, (i_param_24 - 1)) / ((double) (i_r_13 * i_r_13)) - i_mul_adj_72 * i_param_11 * i_param_16 * i_param_19 * pow(i_mul_val_18, (i_param_19 - 1)) / ((double) (i_r_13 * i_r_13)) - i_bultin_33 * i_mul_adj_36 * i_param_11 * i_param_14 * i_param_15 * i_v_26 / ((double) (i_v_31 * i_v_31))) / ((double) i_r_13);
      i_fx_39 += - i_adj_by_r_121 * i_dx_9;
      i_fy_39 += - i_adj_by_r_121 * i_dy_9;
      i_fz_39 += - i_adj_by_r_121 * i_dz_9;
      f[i_a_7][0] += i_adj_by_r_121 * i_dx_9;
      f[i_a_7][1] += i_adj_by_r_121 * i_dy_9;
      f[i_a_7][2] += i_adj_by_r_121 * i_dz_9;
      double i_a_135 = 0.0;
      double i_fx_180 = 0.0;
      double i_fy_180 = 0.0;
      double i_fz_180 = 0.0;
      for (int i_i_136 = 0; i_i_136 < numneigh[i_i_3]; i_i_136++) {
        int i_a_137 = firstneigh[i_i_3][i_i_136];
        double i_px_138 = x[i_a_137][0];
        double i_py_138 = x[i_a_137][1];
        double i_pz_138 = x[i_a_137][2];
        int i_ty_138 = type[i_a_137];
        double i_dx_139 = i_px_4 - i_px_138;
        double i_dy_139 = i_py_4 - i_py_138;
        double i_dz_139 = i_pz_4 - i_pz_138;
        double i_rsq_139 = i_dx_139 * i_dx_139 + i_dy_139 * i_dy_139 + i_dz_139 * i_dz_139;
        double i_param_141 = param_sigma[i_ty_4][i_ty_138];
        double i_mul_val_142 = param_a[i_ty_4][i_ty_138] * i_param_141;
        if (i_rsq_139 > (i_mul_val_142 * i_mul_val_142)) continue;
        if (i_a_137 == i_a_7) continue;
        double i_r_144 = sqrt(i_rsq_139);
        double i_cos_148 = (i_dx_9 * i_dx_139 + i_dy_9 * i_dy_139 + i_dz_9 * i_dz_139) / ((double) (i_r_13 * i_r_144));
        double i_param_149 = param_lambda[i_ty_4][i_ty_8][i_ty_138];
        double i_v_153 = i_cos_148 - param_cos_theta0[i_ty_4][i_ty_8][i_ty_138];
        double i_v_154 = i_v_153 * i_v_153;
        double i_param_155 = param_gamma[i_ty_4][i_ty_8];
        double i_bultin_162 = exp(i_param_11 * i_param_155 / ((double) i_v_31));
        double i_param_163 = param_gamma[i_ty_4][i_ty_138];
        double i_v_168 = i_r_144 - i_mul_val_142;
        double i_bultin_170 = exp(i_param_141 * i_param_163 / ((double) i_v_168));
        i_a_135 += i_param_149 * i_param_15 * i_v_154 * i_bultin_162 * i_bultin_170;
        double i_dx_199 = i_px_8 - i_px_138;
        double i_dy_199 = i_py_8 - i_py_138;
        double i_dz_199 = i_pz_8 - i_pz_138;
        double i_adj_by_r_285 = - i_bultin_162 * i_bultin_170 * i_mul_adj_36 * i_param_11 * i_param_149 * i_param_15 * i_param_155 * i_v_154 / ((double) (i_r_13 * i_v_31 * i_v_31));
        i_fx_176 += - i_adj_by_r_285 * i_dx_9;
        i_fy_176 += - i_adj_by_r_285 * i_dy_9;
        i_fz_176 += - i_adj_by_r_285 * i_dz_9;
        i_fx_180 += i_adj_by_r_285 * i_dx_9;
        i_fy_180 += i_adj_by_r_285 * i_dy_9;
        i_fz_180 += i_adj_by_r_285 * i_dz_9;
        double i_adj_by_r_287 = - i_bultin_162 * i_bultin_170 * i_mul_adj_36 * i_param_141 * i_param_149 * i_param_15 * i_param_163 * i_v_154 / ((double) (i_r_144 * i_v_168 * i_v_168));
        i_fx_176 += - i_adj_by_r_287 * i_dx_139;
        i_fy_176 += - i_adj_by_r_287 * i_dy_139;
        i_fz_176 += - i_adj_by_r_287 * i_dz_139;
        double i_adj_acos_292 = 2 * i_bultin_162 * i_bultin_170 * i_mul_adj_36 * i_param_149 * i_param_15 * i_v_153;
        double i_adj_by_r_293 = (1 / ((double) i_r_144) - i_cos_148 / ((double) i_r_13)) * i_adj_acos_292 / ((double) i_r_13);
        i_fx_176 += - i_adj_by_r_293 * i_dx_9;
        i_fy_176 += - i_adj_by_r_293 * i_dy_9;
        i_fz_176 += - i_adj_by_r_293 * i_dz_9;
        i_fx_180 += i_adj_by_r_293 * i_dx_9;
        i_fy_180 += i_adj_by_r_293 * i_dy_9;
        i_fz_180 += i_adj_by_r_293 * i_dz_9;
        double i_adj_by_r_294 = (1 / ((double) i_r_13) - i_cos_148 / ((double) i_r_144)) * i_adj_acos_292 / ((double) i_r_144);
        i_fx_176 += - i_adj_by_r_294 * i_dx_139;
        i_fy_176 += - i_adj_by_r_294 * i_dy_139;
        i_fz_176 += - i_adj_by_r_294 * i_dz_139;
        double i_adj_by_r_295 = - i_adj_acos_292 / ((double) (i_r_13 * i_r_144));
        i_fx_180 += - i_adj_by_r_295 * i_dx_199;
        i_fy_180 += - i_adj_by_r_295 * i_dy_199;
        i_fz_180 += - i_adj_by_r_295 * i_dz_199;
        f[i_a_137][0] += i_adj_by_r_287 * i_dx_139 + i_adj_by_r_294 * i_dx_139 + i_adj_by_r_295 * i_dx_199;
        f[i_a_137][1] += i_adj_by_r_287 * i_dy_139 + i_adj_by_r_294 * i_dy_139 + i_adj_by_r_295 * i_dy_199;
        f[i_a_137][2] += i_adj_by_r_287 * i_dz_139 + i_adj_by_r_294 * i_dz_139 + i_adj_by_r_295 * i_dz_199;
      }
      i_a_127 += i_a_135;
      f[i_a_7][0] += i_fx_180;
      f[i_a_7][1] += i_fy_180;
      f[i_a_7][2] += i_fz_180;
    }
    i_a_2 += i_a_5;
    f[i_i_3][0] += i_fx_39;
    f[i_i_3][1] += i_fy_39;
    f[i_i_3][2] += i_fz_39;
    i_a_124 += i_a_127;
    f[i_i_3][0] += i_fx_176;
    f[i_i_3][1] += i_fy_176;
    f[i_i_3][2] += i_fz_176;
  }
  eng_vdwl += i_a_2 / ((double) 2);
  eng_vdwl += i_a_124 / ((double) 2);

  if (vflag_fdotr) virial_fdotr_compute();
  fesetenv(FE_DFL_ENV);
}

/* ---------------------------------------------------------------------- */

void PairSwGen::init_style() {
  if (force->newton_pair == 0)
    error->all(FLERR, "Pair style sw/gen requires atom IDs");
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
