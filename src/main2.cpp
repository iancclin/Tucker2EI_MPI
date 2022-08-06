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
#include <numeric>
#include "BlasWrapper.h"
#include "Parameters.h"

int
main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);

  int world_size, world_rank;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  std::string           input_param_filename(argv[1]);
  Tucker2EI::Parameters parameters(input_param_filename);
  if (world_rank == 0)
    parameters.Print();

  std::vector<int> wfn_input_order = parameters.GetInputOrder();
  std::vector<int> printout_nwfn   = parameters.GetPrintoutWfns();
  int              n_wfn           = wfn_input_order.size();

  unsigned int num_ele_x = parameters.GetNumEleX();
  unsigned int num_ele_y = parameters.GetNumEleY();
  unsigned int num_ele_z = parameters.GetNumEleZ();

  double      tensor_decomp_tol = parameters.GetTensorTol();
  std::string omega_filename    = parameters.GetOmegaFilename();
  std::string alpha_filename    = parameters.GetAlphaFilename();
  double      a_square          = parameters.GetASquare();

  // x, y, z, charge
  std::vector<std::array<double, 4>> atom = parameters.GetAtom();

  std::string output_prefix =
    parameters.GetOutputPrefix() + std::to_string(n_wfn) + "_";



  // start computation
  std::string file_name_x = parameters.GetXCoordFilename();
  std::string file_name_y = parameters.GetYCoordFilename();
  std::string file_name_z = parameters.GetZCoordFilename();

  Tucker2EI::FE3D fe(file_name_x,
                     file_name_y,
                     file_name_z,
                     num_ele_x,
                     num_ele_y,
                     num_ele_z,
                     Tucker2EI::QuadPoints(parameters.GetNumFeQuadX()),
                     Tucker2EI::QuadPoints(parameters.GetNumFeQuadY()),
                     Tucker2EI::QuadPoints(parameters.GetNumFeQuadZ()));

  Tucker2EI::FE3D fe_conv(file_name_x,
                          file_name_y,
                          file_name_z,
                          num_ele_x,
                          num_ele_y,
                          num_ele_z,
                          Tucker2EI::QuadPoints(parameters.GetNumFeConvQuadX()),
                          Tucker2EI::QuadPoints(parameters.GetNumFeConvQuadY()),
                          Tucker2EI::QuadPoints(
                            parameters.GetNumFeConvQuadZ()));


  // compute 2-electron integrals
  // two_idx stored the idx to be computed, normal_two_dix stored two_idx maps
  // to 0, 1, 2, ... order for insertion to final matrix
  std::vector<std::array<int, 2>> two_idx, normal_two_idx;
  for (int j = 0; j < n_wfn; ++j)
  {
    for (int i = j; i < n_wfn; ++i)
    {
      int idx_i = wfn_input_order[i], idx_j = wfn_input_order[j];
      two_idx.emplace_back(std::array<int, 2>{idx_i, idx_j});
      normal_two_idx.emplace_back(std::array<int, 2>{i, j});
    }
  }

  std::vector<int> two_idx_list(world_size + 1, 0);
  int              mod_n_twoidx = two_idx.size() % world_size;
  int              div_n_twoidx = two_idx.size() / world_size;
  for (int i_rank = 0; i_rank < world_size; ++i_rank)
  {
    two_idx_list[i_rank + 1] =
      two_idx_list[i_rank] + div_n_twoidx + (i_rank < mod_n_twoidx ? 1 : 0);
  }

  std::vector<std::array<int, 4>> four_idx, normal_four_idx;
  for (int j = 0; j < two_idx.size(); ++j)
  {
    for (int i = j; i < two_idx.size(); ++i)
    {
      four_idx.emplace_back(std::array<int, 4>{
        two_idx[i][0], two_idx[i][1], two_idx[j][0], two_idx[j][1]});
      normal_four_idx.emplace_back(std::array<int, 4>{normal_two_idx[i][0],
                                                      normal_two_idx[i][1],
                                                      normal_two_idx[j][0],
                                                      normal_two_idx[j][1]});
    }
  }



  std::vector<int> four_idx_list(world_size + 1, 0);
  int              mod_n_fouridx = four_idx.size() % world_size;
  int              div_n_fouridx = four_idx.size() / world_size;
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
  std::vector<double> fidx_int_reduced(four_idx.size(), 0.0);

  for (int i = four_idx_list[world_rank]; i < four_idx_list[world_rank + 1];
       ++i)
  {
    fidx_int_reduced[i] = four_index_integrator.Integrate(four_idx[i]);
  }

  MPI_Allreduce(MPI_IN_PLACE,
                fidx_int_reduced.data(),
                fidx_int_reduced.size(),
                MPI_DOUBLE,
                MPI_SUM,
                MPI_COMM_WORLD);

  std::vector<std::vector<std::vector<std::vector<double>>>> fidx_int(
    n_wfn,
    std::vector<std::vector<std::vector<double>>>(
      n_wfn,
      std::vector<std::vector<double>>(n_wfn,
                                       std::vector<double>(n_wfn, 0.0))));
  for (int i_fidx = 0; i_fidx < four_idx.size(); ++i_fidx)
  {
    int    i             = normal_four_idx[i_fidx][0];
    int    j             = normal_four_idx[i_fidx][1];
    int    k             = normal_four_idx[i_fidx][2];
    int    l             = normal_four_idx[i_fidx][3];
    double fidx_int_val  = fidx_int_reduced[i_fidx];
    fidx_int[i][j][k][l] = fidx_int_val;
    fidx_int[k][l][i][j] = fidx_int_val;
    fidx_int[j][i][l][k] = fidx_int_val;
    fidx_int[l][k][j][i] = fidx_int_val;
    fidx_int[j][i][k][l] = fidx_int_val;
    fidx_int[l][k][i][j] = fidx_int_val;
    fidx_int[i][j][l][k] = fidx_int_val;
    fidx_int[k][l][j][i] = fidx_int_val;
  }

  if (world_rank == 0)
  {
    std::string fidx_reduced_filename = output_prefix + "four_idx_reduced.txt";

    FILE *cfout = fopen(fidx_reduced_filename.c_str(), "w");
    fprintf(cfout, "four index:\n");
    fprintf(cfout, "number wavefunctions: %d\n", n_wfn);
    fprintf(cfout, "used wfn and order: ");
    for (int i = 0; i < wfn_input_order.size(); ++i)
    {
      fprintf(cfout, "%d, ", wfn_input_order[i]);
    }
    fprintf(cfout, "\n\n");
    for (int i = 0; i < fidx_int_reduced.size(); ++i)
    {
      fprintf(cfout,
              "fidx([%3d][%3d][%3d][%3d])[%d][%d][%d][%d]\t=\t%.16e\n",
              normal_four_idx[i][0],
              normal_four_idx[i][1],
              normal_four_idx[i][2],
              normal_four_idx[i][3],
              four_idx[i][0],
              four_idx[i][1],
              four_idx[i][2],
              four_idx[i][3],
              fidx_int_reduced[i]);
    }
    fclose(cfout);

    for (int i_printout = 0; i_printout < printout_nwfn.size(); ++i_printout)
    {
      std::string prefix =
        output_prefix + "orb" + std::to_string(printout_nwfn[i_printout]) + "_";
      std::string fidx_filename = prefix + "four_idx.txt";

      cfout = fopen(fidx_filename.c_str(), "w");
      fprintf(cfout, "%d\n", printout_nwfn[i_printout]);
      for (int i = 0; i < printout_nwfn[i_printout]; ++i)
      {
        for (int j = 0; j < printout_nwfn[i_printout]; ++j)
        {
          for (int k = 0; k < printout_nwfn[i_printout]; ++k)
          {
            for (int l = 0; l < printout_nwfn[i_printout]; ++l)
            {
              fprintf(cfout, "%.16e ", fidx_int[i][j][k][l]);
            }
          }
        }
        fprintf(cfout, "\n");
      }
      fclose(cfout);
    }
  }


  MPI_Finalize();
  return 0;
}
