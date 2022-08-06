/******************************************************************************
 * Copyright (c) 2022.                                                        *
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

#include <string>
#include <Tucker.hpp>
#include <mpi.h>
#include "BlasWrapper.h"
#include "DataReader.h"
#include "Parameters.h"
#include "FE3D.h"
#include "ConvolutionComputer.h"
#include "FourIndexIntegrator.h"
#include "FourIndexGenerator.h"
#include "TwoIndexIntegrator.h"
#include <chrono>

using namespace std::chrono;

int
main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);

  int world_size, world_rank;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);


  std::string           input_param_filename(argv[1]);
  Tucker2EI::Parameters parameters(input_param_filename);
  //  if (world_rank == 0)
  //    parameters.Print();

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

  double t_start, t_end;

  if (world_rank == 0)
  {
    std::string meta_filename = output_prefix + "metadata.txt";
    FILE       *cfout         = fopen(meta_filename.c_str(), "w");
    fprintf(cfout, "number wavefunctions: %d\n", n_wfn);
    fprintf(cfout, std::string("omega file: " + omega_filename).c_str(), n_wfn);
    fprintf(cfout, "\n");
    fprintf(cfout, std::string("alpha file: " + alpha_filename).c_str(), n_wfn);
    fprintf(cfout, "\n");
    fprintf(cfout, "A square: %.16e\n", a_square);
    fprintf(cfout, "tensor decomposition tolerance: %.4e\n", tensor_decomp_tol);

    fprintf(cfout, "atom position: \n");
    for (int atom_i = 0; atom_i < atom.size(); ++atom_i)
    {
      fprintf(cfout,
              "atom %03d (%.10e): %.10e, %.10e, %.10e\n",
              atom_i,
              atom[atom_i][3],
              atom[atom_i][0],
              atom[atom_i][1],
              atom[atom_i][2]);
    }
    fprintf(cfout, "\n\n");

    fprintf(cfout,
            "wfn number elements: (%d, %d, %d)\n",
            fe.fe_x.GetNumberElements(),
            fe.fe_y.GetNumberElements(),
            fe.fe_z.GetNumberElements());
    fprintf(cfout,
            "wfn fe order: (%d, %d, %d)\n",
            fe.fe_x.GetNumberNodesPerElement() - 1,
            fe.fe_y.GetNumberNodesPerElement() - 1,
            fe.fe_z.GetNumberNodesPerElement() - 1);
    fprintf(cfout,
            "wfn quad: (%d, %d, %d)\n\n",
            fe.fe_x.GetNumberQuadPointsPerElement(),
            fe.fe_y.GetNumberQuadPointsPerElement(),
            fe.fe_z.GetNumberQuadPointsPerElement());
    fprintf(cfout, "number wavefunctions: %d\n", n_wfn);
    fprintf(cfout, "used wfn and order: ");
    for (int i = 0; i < wfn_input_order.size(); ++i)
    {
      fprintf(cfout, "%d, ", wfn_input_order[i]);
    }
    fprintf(cfout, "\n");
    fprintf(cfout, "printed out at which number of wfn: ");
    for (int i = 0; i < printout_nwfn.size(); ++i)
    {
      fprintf(cfout, "%d, ", printout_nwfn[i]);
    }
    fprintf(cfout, "\n\n");

    fprintf(cfout, "FE nodes coord x:\n");
    for (int i = 0; i < fe.fe_x.GetNumberNodes(); ++i)
    {
      fprintf(cfout, "%.16e ", fe.fe_x.GetNodalCoord()[i]);
    }
    fprintf(cfout, "\n");
    fprintf(cfout, "FE nodes coord y:\n");
    for (int i = 0; i < fe.fe_y.GetNumberNodes(); ++i)
    {
      fprintf(cfout, "%.16e ", fe.fe_y.GetNodalCoord()[i]);
    }
    fprintf(cfout, "\n");
    fprintf(cfout, "FE nodes coord z:\n");
    for (int i = 0; i < fe.fe_z.GetNumberNodes(); ++i)
    {
      fprintf(cfout, "%.16e ", fe.fe_z.GetNodalCoord()[i]);
    }
    fprintf(cfout, "\n");

    fprintf(cfout, "FE conv nodes coord x:\n");
    for (int i = 0; i < fe_conv.fe_x.GetNumberNodes(); ++i)
    {
      fprintf(cfout, "%.16e ", fe_conv.fe_x.GetNodalCoord()[i]);
    }
    fprintf(cfout, "\n");
    fprintf(cfout, "FE conv nodes coord y:\n");
    for (int i = 0; i < fe_conv.fe_y.GetNumberNodes(); ++i)
    {
      fprintf(cfout, "%.16e ", fe_conv.fe_y.GetNodalCoord()[i]);
    }
    fprintf(cfout, "\n");
    fprintf(cfout, "FE conv nodes coord z:\n");
    for (int i = 0; i < fe_conv.fe_z.GetNumberNodes(); ++i)
    {
      fprintf(cfout, "%.16e ", fe_conv.fe_z.GetNodalCoord()[i]);
    }
    fprintf(cfout, "\n");

    fprintf(cfout, "FE quad coord x:\n");
    for (int i = 0; i < fe.fe_x.GetNumberQuadPoints(); ++i)
    {
      fprintf(cfout, "%.16e ", fe.fe_x.GetQuadCoord()[i]);
    }
    fprintf(cfout, "\n");
    fprintf(cfout, "FE quad coord y:\n");
    for (int i = 0; i < fe.fe_y.GetNumberQuadPoints(); ++i)
    {
      fprintf(cfout, "%.16e ", fe.fe_y.GetQuadCoord()[i]);
    }
    fprintf(cfout, "\n");
    fprintf(cfout, "FE quad coord z:\n");
    for (int i = 0; i < fe.fe_z.GetNumberQuadPoints(); ++i)
    {
      fprintf(cfout, "%.16e ", fe.fe_z.GetQuadCoord()[i]);
    }
    fprintf(cfout, "\n");

    fprintf(cfout, "FE conv quad coord x:\n");
    for (int i = 0; i < fe_conv.fe_x.GetNumberQuadPoints(); ++i)
    {
      fprintf(cfout, "%.16e ", fe_conv.fe_x.GetQuadCoord()[i]);
    }
    fprintf(cfout, "\n");
    fprintf(cfout, "FE conv quad coord y:\n");
    for (int i = 0; i < fe_conv.fe_y.GetNumberQuadPoints(); ++i)
    {
      fprintf(cfout, "%.16e ", fe_conv.fe_y.GetQuadCoord()[i]);
    }
    fprintf(cfout, "\n");
    fprintf(cfout, "FE conv quad coord z:\n");
    for (int i = 0; i < fe_conv.fe_z.GetNumberQuadPoints(); ++i)
    {
      fprintf(cfout, "%.16e ", fe_conv.fe_z.GetQuadCoord()[i]);
    }
    fprintf(cfout, "\n");

    fprintf(cfout, "quadx=[");
    for (int i = 0; i < fe.fe_x.GetNumberQuadPoints() - 1; ++i)
    {
      fprintf(cfout, "%.16e, ", fe.fe_x.GetQuadCoord()[i]);
    }
    fprintf(cfout, "%.16e];\n", fe.fe_x.GetQuadCoord().back());

    fprintf(cfout, "quady=[");
    for (int i = 0; i < fe.fe_y.GetNumberQuadPoints() - 1; ++i)
    {
      fprintf(cfout, "%.16e, ", fe.fe_y.GetQuadCoord()[i]);
    }
    fprintf(cfout, "%.16e];\n", fe.fe_y.GetQuadCoord().back());

    fprintf(cfout, "quadz=[");
    for (int i = 0; i < fe.fe_z.GetNumberQuadPoints() - 1; ++i)
    {
      fprintf(cfout, "%.16e, ", fe.fe_z.GetQuadCoord()[i]);
    }
    fprintf(cfout, "%.16e];\n", fe.fe_z.GetQuadCoord().back());

    fprintf(cfout, "convquadx=[");
    for (int i = 0; i < fe_conv.fe_x.GetNumberQuadPoints() - 1; ++i)
    {
      fprintf(cfout, "%.16e, ", fe_conv.fe_x.GetQuadCoord()[i]);
    }
    fprintf(cfout, "%.16e];\n", fe_conv.fe_x.GetQuadCoord().back());

    fprintf(cfout, "convquady=[");
    for (int i = 0; i < fe_conv.fe_y.GetNumberQuadPoints() - 1; ++i)
    {
      fprintf(cfout, "%.16e, ", fe_conv.fe_y.GetQuadCoord()[i]);
    }
    fprintf(cfout, "%.16e];\n", fe_conv.fe_y.GetQuadCoord().back());

    fprintf(cfout, "convquadz=[");
    for (int i = 0; i < fe_conv.fe_z.GetNumberQuadPoints() - 1; ++i)
    {
      fprintf(cfout, "%.16e, ", fe_conv.fe_z.GetQuadCoord()[i]);
    }
    fprintf(cfout, "%.16e];\n", fe_conv.fe_z.GetQuadCoord().back());


    fclose(cfout);
  }

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

  Tucker2EI::TwoIndexIntegrator two_index_integrator(fe, atom);

  // the two terms are stored in reduced form (only store and compute unique
  // terms, symmetric entries will be recovered later)
  std::vector<double> kinetic_reduced(two_idx.size(), 0.0);
  std::vector<double> external_reduced(two_idx.size(), 0.0);
  std::vector<double> oneele_reduced(two_idx.size(), 0.0);
  std::vector<double> overlap_reduced(two_idx.size(), 0.0);

  MPI_Barrier(MPI_COMM_WORLD);
  t_start = MPI_Wtime();
  for (int i = two_idx_list[world_rank]; i < two_idx_list[world_rank + 1]; ++i)
  {
    external_reduced[i] =
      two_index_integrator.ExternalIntegrator(two_idx[i][0], two_idx[i][1]);
    kinetic_reduced[i] =
      two_index_integrator.KineticIntegrator(two_idx[i][0], two_idx[i][1]);
    oneele_reduced[i] = 0.5 * kinetic_reduced[i] - external_reduced[i];
    overlap_reduced[i] =
      two_index_integrator.OverlapIntegrator(two_idx[i][0], two_idx[i][1]);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  t_end = MPI_Wtime();
  printf("timer: one electron temrs computation time: %.8fs\n", t_end -
                                                                  t_start);

  MPI_Barrier(MPI_COMM_WORLD);
  t_start = MPI_Wtime();
  MPI_Allreduce(MPI_IN_PLACE,
                kinetic_reduced.data(),
                kinetic_reduced.size(),
                MPI_DOUBLE,
                MPI_SUM,
                MPI_COMM_WORLD);

  MPI_Allreduce(MPI_IN_PLACE,
                external_reduced.data(),
                external_reduced.size(),
                MPI_DOUBLE,
                MPI_SUM,
                MPI_COMM_WORLD);

  MPI_Allreduce(MPI_IN_PLACE,
                oneele_reduced.data(),
                oneele_reduced.size(),
                MPI_DOUBLE,
                MPI_SUM,
                MPI_COMM_WORLD);

  MPI_Allreduce(MPI_IN_PLACE,
                overlap_reduced.data(),
                overlap_reduced.size(),
                MPI_DOUBLE,
                MPI_SUM,
                MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);
  t_end = MPI_Wtime();
  printf("timer: one electron terms communication time: %.8fs\n", t_end -
                                                                  t_start);

  std::vector<std::vector<double>> kinetic(n_wfn,
                                           std::vector<double>(n_wfn, 0.0));
  std::vector<std::vector<double>> external(n_wfn,
                                            std::vector<double>(n_wfn, 0.0));
  std::vector<std::vector<double>> one_ei(n_wfn,
                                          std::vector<double>(n_wfn, 0.0));
  std::vector<std::vector<double>> overlap_mat(n_wfn,
                                               std::vector<double>(n_wfn, 0.0));

  MPI_Barrier(MPI_COMM_WORLD);
  t_start = MPI_Wtime();
  for (int i_twoidx = 0; i_twoidx < two_idx.size(); ++i_twoidx)
  {
    int    i   = normal_two_idx[i_twoidx][0];
    int    j   = normal_two_idx[i_twoidx][1];
    double kin = kinetic_reduced[i_twoidx], ext = external_reduced[i_twoidx];
    double ovr        = overlap_reduced[i_twoidx];
    kinetic[i][j]     = kin;
    kinetic[j][i]     = kin;
    external[i][j]    = ext;
    external[j][i]    = ext;
    one_ei[i][j]      = 0.5 * kin - ext;
    one_ei[j][i]      = one_ei[i][j];
    overlap_mat[i][j] = ovr;
    overlap_mat[j][i] = ovr;
  }

  if (world_rank == 0)
  {
    std::string kinetic_reduced_filename =
      output_prefix + "kinetic_reduced.txt";
    std::string external_reduced_filename =
      output_prefix + "external_reduced.txt";
    std::string oneele_reduced_filename = output_prefix + "oneele_reduced.txt";
    std::string ovrlap_reduced_filename = output_prefix + "overlap_reduced.txt";

    FILE *cfout = fopen(kinetic_reduced_filename.c_str(), "w");
    fprintf(cfout, "kinetic:\n");
    fprintf(cfout, "number wavefunctions: %d\n", n_wfn);
    fprintf(cfout, "used wfn and order: ");
    for (int i = 0; i < wfn_input_order.size(); ++i)
    {
      fprintf(cfout, "%d, ", wfn_input_order[i]);
    }
    fprintf(cfout, "\n\n");
    for (int i = 0; i < kinetic_reduced.size(); ++i)
    {
      fprintf(cfout,
              "kinetic([%3d][%3d])[%3d][%3d]\t=\t%.16e\n",
              normal_two_idx[i][0],
              normal_two_idx[i][1],
              two_idx[i][0],
              two_idx[i][1],
              kinetic_reduced[i]);
    }
    fclose(cfout);

    cfout = fopen(external_reduced_filename.c_str(), "w");
    fprintf(cfout, "external:\n");
    fprintf(cfout, "number wavefunctions: %d\n", n_wfn);
    fprintf(cfout, "used wfn and order: ");
    for (int i = 0; i < wfn_input_order.size(); ++i)
    {
      fprintf(cfout, "%d, ", wfn_input_order[i]);
    }
    fprintf(cfout, "\n\n");
    for (int i = 0; i < external_reduced.size(); ++i)
    {
      fprintf(cfout,
              "external([%3d][%3d])[%d][%d]\t=\t%.16e\n",
              normal_two_idx[i][0],
              normal_two_idx[i][1],
              two_idx[i][0],
              two_idx[i][1],
              external_reduced[i]);
    }
    fclose(cfout);

    cfout = fopen(oneele_reduced_filename.c_str(), "w");
    fprintf(cfout, "one ele terms:\n");
    fprintf(cfout, "number wavefunctions: %d\n", n_wfn);
    fprintf(cfout, "used wfn and order: ");
    for (int i = 0; i < wfn_input_order.size(); ++i)
    {
      fprintf(cfout, "%d, ", wfn_input_order[i]);
    }
    fprintf(cfout, "\n\n");
    for (int i = 0; i < oneele_reduced.size(); ++i)
    {
      fprintf(cfout,
              "h([%3d][%3d])[%d][%d]\t=\t%.16e\n",
              normal_two_idx[i][0],
              normal_two_idx[i][1],
              two_idx[i][0],
              two_idx[i][1],
              oneele_reduced[i]);
    }
    fclose(cfout);

    cfout = fopen(ovrlap_reduced_filename.c_str(), "w");
    fprintf(cfout, "overlap:\n");
    fprintf(cfout, "number wavefunctions: %d\n", n_wfn);
    fprintf(cfout, "used wfn and order: ");
    for (int i = 0; i < wfn_input_order.size(); ++i)
    {
      fprintf(cfout, "%d, ", wfn_input_order[i]);
    }
    fprintf(cfout, "\n\n");
    for (int i = 0; i < kinetic_reduced.size(); ++i)
    {
      fprintf(cfout,
              "overlap([%3d][%3d])[%3d][%3d]\t=\t%.16e\n",
              normal_two_idx[i][0],
              normal_two_idx[i][1],
              two_idx[i][0],
              two_idx[i][1],
              overlap_reduced[i]);
    }
    fclose(cfout);

    for (int i_printout = 0; i_printout < printout_nwfn.size(); ++i_printout)
    {
      std::string prefix =
        output_prefix + "orb" + std::to_string(printout_nwfn[i_printout]) + "_";
      std::string kinetic_filename      = prefix + "kinetic.txt";
      std::string external_filename     = prefix + "external.txt";
      std::string one_electron_filename = prefix + "one_electron.txt";
      std::string overlap_filename      = prefix + "overlap.txt";

      cfout = fopen(kinetic_filename.c_str(), "w");
      fprintf(cfout, "kinetic: \n");
      fprintf(cfout, "%d\n", printout_nwfn[i_printout]);
      for (int i = 0; i < printout_nwfn[i_printout]; ++i)
      {
        for (int j = 0; j < printout_nwfn[i_printout]; ++j)
        {
          fprintf(cfout, "%.16e ", kinetic[i][j]);
        }
        fprintf(cfout, "\n");
      }
      fclose(cfout);

      cfout = fopen(external_filename.c_str(), "w");
      fprintf(cfout, "external: \n");
      fprintf(cfout, "%d\n", printout_nwfn[i_printout]);
      for (int i = 0; i < printout_nwfn[i_printout]; ++i)
      {
        for (int j = 0; j < printout_nwfn[i_printout]; ++j)
        {
          fprintf(cfout, "%.16e ", external[i][j]);
        }
        fprintf(cfout, "\n");
      }
      fclose(cfout);

      cfout = fopen(one_electron_filename.c_str(), "w");
      fprintf(cfout, "%d\n", printout_nwfn[i_printout]);
      for (int i = 0; i < printout_nwfn[i_printout]; ++i)
      {
        for (int j = 0; j < printout_nwfn[i_printout]; ++j)
        {
          fprintf(cfout, "%.16e ", one_ei[i][j]);
        }
        fprintf(cfout, "\n");
      }
      fclose(cfout);

      cfout = fopen(overlap_filename.c_str(), "w");
      fprintf(cfout, "%d\n", printout_nwfn[i_printout]);
      for (int i = 0; i < printout_nwfn[i_printout]; ++i)
      {
        for (int j = 0; j < printout_nwfn[i_printout]; ++j)
        {
          fprintf(cfout, "%.16e ", overlap_mat[i][j]);
        }
        fprintf(cfout, "\n");
      }
      fclose(cfout);
    }
  }
  MPI_Barrier(MPI_COMM_WORLD);
  t_end = MPI_Wtime();
  printf("timer: one electron terms printout time: %.8fs\n", t_end -
                                                                    t_start);


  Tucker2EI::ConvolutionComputer convolution_computer(omega_filename,
                                                      alpha_filename,
                                                      a_square);

  Tucker2EI::FEMap fe_map(fe, MPI_COMM_WORLD);
  Tucker2EI::FEMap fe_conv_map(fe_conv, MPI_COMM_WORLD);

  Tucker2EI::FourIndexGenerator four_index_generator(fe_map,
                                                     fe_conv_map,
                                                     convolution_computer,
                                                     parameters.GetTensorTol());


  std::vector<double> fidx_int_reduced;
  MPI_Barrier(MPI_COMM_WORLD);
  t_start = MPI_Wtime();
  four_index_generator.Calculate(wfn_input_order, fidx_int_reduced);
  MPI_Barrier(MPI_COMM_WORLD);
  t_end = MPI_Wtime();
  printf("timer: two electron terms computation time: %.8fs\n", t_end -
                                                                    t_start);



  MPI_Barrier(MPI_COMM_WORLD);
  t_start = MPI_Wtime();
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
  MPI_Barrier(MPI_COMM_WORLD);
  t_end = MPI_Wtime();
  printf("timer: two electron terms communication time: %.8fs\n", t_end -
                                                                    t_start);

  MPI_Finalize();
}