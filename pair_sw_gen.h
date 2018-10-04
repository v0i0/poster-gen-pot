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

PairStyle(sw/gen,PairSwGen)

#else

#ifndef LMP_PAIR_SW_GEN_H
#define LMP_PAIR_SW_GEN_H LMP_PAIR_SW_GEN_H

#include "pair.h"

namespace LAMMPS_NS {

class PairSwGen : public Pair {
 public:
  PairSwGen(class LAMMPS *);
  ~PairSwGen();
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
  double **param_a;
  double **param_B;
  double **param_p;
  double **param_q;
  double **param_epsilon;
  double **param_sigma;
  double **param_gamma;
  double ***param_lambda;
  double ***param_cos_theta0;


};

}  // namespace LAMMPS_NS

#endif  // LMP_PAIR_SW_GEN_H

#endif  // PAIR_CLASS
