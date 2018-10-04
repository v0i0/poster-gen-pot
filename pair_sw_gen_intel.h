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

PairStyle(sw/gen/intel,PairSwGenIntel)

#else

#ifndef LMP_PAIR_SW_GEN_INTEL_H
#define LMP_PAIR_SW_GEN_INTEL_H LMP_PAIR_SW_GEN_INTEL_H

#include "pair.h"

namespace LAMMPS_NS {

class FixIntel;

template<class flt_t, class acc_t>
class IntelBuffers;

class PairSwGenIntel : public Pair {
 public:

PairSwGenIntel(class LAMMPS *);
~PairSwGenIntel();
  void compute(int, int);
  void settings(int, char**);
  void coeff(int, char**);
  double init_one(int, int);
  void init_style();
  virtual void allocate();
  virtual void file_process_line(int narg, char **arg, char **coeff_arg);

  FixIntel* fix;
  int onetype;
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


  template <class flt_t> class ForceConst;

  template <class flt_t, class acc_t>
  void compute(int eflag, int vflag, IntelBuffers<flt_t,acc_t> *buffers,
               const ForceConst<flt_t> &fc);
  template <int ONETYPE, int EFLAG, class flt_t, class acc_t>
  void eval(const int offload, const int vflag,
            IntelBuffers<flt_t,acc_t> * buffers, const ForceConst<flt_t> &fc,
            const int astart, const int aend);

  template <class flt_t, class acc_t>
  void pack_force_const(ForceConst<flt_t> &fc,
                        IntelBuffers<flt_t, acc_t> *buffers);

  // ----------------------------------------------------------------------

  template <class flt_t>
  struct ForceConst {
    ForceConst() : _ntypes(0)  {}
    ~ForceConst() { set_ntypes(0, NULL, _cop); }

    void set_ntypes(const int ntypes, Memory *memory, const int cop);

   private:
    int _ntypes, _cop;
    Memory *_memory;
  };
  ForceConst<float> force_const_single;
  ForceConst<double> force_const_double;
};

}  // namespace LAMMPS_NS

#endif  // LMP_PAIR_SW_GEN_INTEL_H

#endif  // PAIR_CLASS
