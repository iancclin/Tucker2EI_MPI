/******************************************************************************
 * Copyright (c) 2021-2022.                                                   *
 *  The Regents of the University of Michigan and Tucker2EI developers.       *
 *                                                                            *
 *  This file is part of the Tucker2EI code.                                  *
 *                                                                            *
 *  Tucker2EI is free software: you can redistribute it and/or modify         *
 *    it under the terms of the Lesser GNU General Public License as          *
 *    published by the Free Software Foundation, either version 3 of          *
 *    the License, or (at your option) any later version.                     *
 *                                                                            *
 *  Tucker2EI is distributed in the hope that it will be useful, but          *
 *    WITHOUT ANY WARRANTY; without even the implied warranty                 *
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.                    *
 *    See the Lesser GNU General Public License for more details.             *
 *                                                                            *
 *  You should have received a copy of the GNU Lesser General Public          *
 *    License at the top level of Tucker2EI distribution. If not, see         *
 *    <https://www.gnu.org/licenses/>.                                        *
 ******************************************************************************/

/*
 * @author Ian C. Lin.
 */

#include "DataReader.h"
#include <algorithm>
#include <iostream>
#include <memory>
#include "FE.h"
#include "ConvolutionComputer.h"
#include "FE3D.h"
#include "FourIndexIntegrator.h"
#include "TwoIndexIntegrator.h"
#include <mpi.h>
#include <complex>
#include "BlasWrapper.h"

int
main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);

  // set up meta data
  bool             is_dryrun = false;
  int              n_wfn     = 92;
  std::vector<int> wfn_input_order;
  for (int i = 0; i < n_wfn; ++i)
    wfn_input_order.push_back(i);

  unsigned int num_ele_x = 26, num_ele_y = 26, num_ele_z = 52;

  double      tensor_decomp_tol = 1.0e-6;
  std::string omega_filename    = "omega_k35_5e9";
  std::string alpha_filename    = "alpha_k35_5e9";
  double      a_square          = 7.5817691832389684e-07;

  std::string output_prefix = "H2_HF_OrbSelf" + std::to_string(n_wfn) + "_";

  // start computation
  std::string file_name_x = "x_coord.dat";
  std::string file_name_y = "y_coord.dat";
  std::string file_name_z = "z_coord.dat";

  Tucker2EI::FE3D fe(file_name_x,
                     file_name_y,
                     file_name_z,
                     num_ele_x,
                     num_ele_y,
                     num_ele_z,
                     Tucker2EI::PT_10,
                     Tucker2EI::PT_10,
                     Tucker2EI::PT_10);

  Tucker2EI::FE3D fe_conv(file_name_x,
                          file_name_y,
                          file_name_z,
                          num_ele_x,
                          num_ele_y,
                          num_ele_z,
                          Tucker2EI::PT_14,
                          Tucker2EI::PT_14,
                          Tucker2EI::PT_14);

  int world_size, world_rank;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  if (is_dryrun == true)
  {
    return 0;
  }

  std::vector<int> four_idx_list(world_size + 1, 0);
  int              mod_n_fouridx = n_wfn % world_size;
  int              div_n_fouridx = n_wfn / world_size;
  for (int i_rank = 0; i_rank < world_size; ++i_rank)
  {
    four_idx_list[i_rank + 1] =
      four_idx_list[i_rank] + div_n_fouridx + (i_rank < mod_n_fouridx ? 1 : 0);
  }

  Tucker2EI::ConvolutionComputer convolution_computer(omega_filename,
                                                      alpha_filename,
                                                      a_square);
  Tucker2EI::FourIndexIntegrator four_index_integrator(fe,
                                                       fe_conv,
                                                       convolution_computer,
                                                       tensor_decomp_tol);

  // the four-idx integrals are stored in reduced form (only store and
  // compute unique terms, symmetric entries will be recovered later)
  std::vector<double> fidx_int(n_wfn, 0.0);

  for (int i = four_idx_list[world_rank]; i < four_idx_list[world_rank + 1];
       ++i)
  {
    fidx_int[i] =
      four_index_integrator.Integrate(std::array<int, 4>{i, i, i, i});
  }

  MPI_Allreduce(MPI_IN_PLACE,
                fidx_int.data(),
                fidx_int.size(),
                MPI_DOUBLE,
                MPI_SUM,
                MPI_COMM_WORLD);

  if (world_rank == 0)
  {
    std::string fidx_reduced_filename = output_prefix + "self_four_idx.txt";

    FILE *cfout = fopen(fidx_reduced_filename.c_str(), "w");
    for (int i = 0; i < fidx_int.size(); ++i)
    {
      fprintf(cfout, "wfn %d self = %.16e\n", i, fidx_int[i]);
    }
    fclose(cfout);
  }

  MPI_Finalize();
  return 0;
}
