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

#ifdef PAIR_CLASS

PairStyle(pot/gen,PairPotGen)

#else

#ifndef LMP_PAIR_POT_GEN_H
#define LMP_PAIR_POT_GEN_H LMP_PAIR_POT_GEN_H

#include "pair.h"

namespace LAMMPS_NS {

class PairPotGen : public Pair {
 public:
  PairPotGen(class LAMMPS *);
  ~PairPotGen();
  void compute(int, int);
  void settings(int, char**);
  void coeff(int, char**);
  double init_one(int, int);
  void init_style();
  virtual void allocate();
  virtual void file_process_line(int narg, char **arg, char **coeff_arg);

  double cutmax;
  int* type_map;
  double **param_A;
  double **param_B;
  double **param_lambda_1;
  double **param_lambda_2;
  double **param_beta;
  double **param_n;
  double ***param_R;
  double ***param_D;
  double ***param_lambda_3;
  double ***param_mm;
  double ***param_gamma;
  double ***param_c;
  double ***param_d;
  double ***param_cos_theta_0;

};

}  // namespace LAMMPS_NS

#endif  // LMP_PAIR_POT_GEN_H

#endif  // PAIR_CLASS
