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
#include "FE.h"
#include "FE3D.h"
#include <algorithm>
#include <iostream>
#include <mpi.h>
#include <numeric>

int
main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);

  // set up meta data
  int          n_wfn           = 80;
  std::string  output_filename = "H2_r2exp_80.exp";
  unsigned int num_ele_x = 26, num_ele_y = 26, num_ele_z = 40;

  // x, y, z, charge
  std::vector<std::array<double, 4>> atom;
  atom.emplace_back(std::array<double, 4>{0.0, 0.0, -0.699198615750, 1.0});
  atom.emplace_back(std::array<double, 4>{0.0, 0.0, 0.699198615750, 1.0});

  // start computation
  int mpi_rank, mpi_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

  std::vector<int> owned_wfn_idx(mpi_size + 1, 0);
  int              mod_n_wfn = n_wfn % mpi_size, div_n_wfn = n_wfn / mpi_size;
  for (int i_rank = 0; i_rank < mpi_size; ++i_rank)
  {
    owned_wfn_idx[i_rank + 1] =
      owned_wfn_idx[i_rank] + div_n_wfn + (i_rank < mod_n_wfn ? 1 : 0);
  }

  int owned_wfn_start = owned_wfn_idx[mpi_rank];
  int owned_wfn_end   = owned_wfn_idx[mpi_rank + 1];

  std::string file_name_x = "x_coord.dat";
  std::string file_name_y = "y_coord.dat";
  std::string file_name_z = "z_coord.dat";

  // start computation
  Tucker2EI::FE3D fe(file_name_x,
                     file_name_y,
                     file_name_z,
                     num_ele_x,
                     num_ele_y,
                     num_ele_z,
                     Tucker2EI::PT_10,
                     Tucker2EI::PT_10,
                     Tucker2EI::PT_10);

  double                total_charge  = 0.0;
  std::array<double, 3> charge_center = {0.0, 0.0, 0.0};
  for (int i_atom = 0; i_atom < atom.size(); ++i_atom)
  {
    double charge = atom[i_atom][3];
    total_charge += charge;
    for (int dim = 0; dim < 3; ++dim)
    {
      charge_center[dim] += charge * atom[i_atom][dim];
    }
  }
  for (int dim = 0; dim < 3; ++dim)
  {
    charge_center[dim] /= total_charge;
  }

  FILE *fileID = nullptr;

  if (mpi_rank == 0)
  {
    fileID = fopen(output_filename.c_str(), "w");
    fprintf(fileID, "rsq exp value with atoms\n");
    fprintf(fileID,
            "fe nodes: (%d, %d, %d)\n",
            fe.fe_x.GetNumberNodes(),
            fe.fe_y.GetNumberNodes(),
            fe.fe_z.GetNumberNodes());
    fprintf(fileID,
            "fe order: (%d, %d, %d)\n",
            fe.fe_x.GetNumberNodesPerElement() - 1,
            fe.fe_y.GetNumberNodesPerElement() - 1,
            fe.fe_z.GetNumberNodesPerElement() - 1);

    fprintf(fileID, "atoms coordinates (x, y, z, charge)\n");
    for (int i_atom = 0; i_atom < atom.size(); ++i_atom)
    {
      fprintf(fileID,
              "atom %3d charge %.4f:  (%.10e, %.10e, %.10e)\n",
              i_atom,
              atom[i_atom][0],
              atom[i_atom][1],
              atom[i_atom][2],
              atom[i_atom][3]);
    }
    fprintf(fileID,
            "charge center (x, y, z, total charge): (%.10e, %.10e, %.10e, %.4f)"
            "\n\n",
            charge_center[0],
            charge_center[1],
            charge_center[2],
            total_charge);
  }

  const auto &node_x = fe.fe_x.GetNodalCoord();
  const auto &node_y = fe.fe_y.GetNodalCoord();
  const auto &node_z = fe.fe_z.GetNodalCoord();

  std::vector<double> wfn_r2_exp(n_wfn, 0.0);
  for (int i_wfn = owned_wfn_start; i_wfn < owned_wfn_end; ++i_wfn)
  {
    Tucker::Tensor *wfn;
    std::string     wfn_filename = "waveFunc_" + std::to_string(i_wfn) + ".dat";

    Tucker2EI::utils::TensorReader(wfn_filename,
                                   fe.fe_x.GetNumberNodes(),
                                   fe.fe_y.GetNumberNodes(),
                                   fe.fe_z.GetNumberNodes(),
                                   wfn);
    double *wfn_ptr = wfn->data();
    for (int k = 0; k < fe.fe_z.GetNumberNodes(); ++k)
    {
      for (int j = 0; j < fe.fe_y.GetNumberNodes(); ++j)
      {
        for (int i = 0; i < fe.fe_x.GetNumberNodes(); ++i)
        {
          std::vector<double> x_atom(atom.size(), 0);
          std::vector<double> y_atom(atom.size(), 0);
          std::vector<double> z_atom(atom.size(), 0);
          for (int i_atom = 0; i_atom < atom.size(); ++i_atom)
          {
            x_atom[i_atom] = node_x[i] - atom[i_atom][0];
            y_atom[i_atom] = node_y[j] - atom[i_atom][1];
            z_atom[i_atom] = node_z[k] - atom[i_atom][2];
          }
          double rsq = 0.0;
          for (int i_atom = 0; i_atom < atom.size(); ++i_atom)
          {
            rsq = x_atom[i_atom] * x_atom[i_atom] +
                  y_atom[i_atom] * y_atom[i_atom] +
                  z_atom[i_atom] * z_atom[i_atom];
          }
          double wfn_val = *(wfn_ptr);
          *(wfn_ptr)     = wfn_val * wfn_val * rsq;
          wfn_ptr++;


          //          double x       = node_x[i] - charge_center[0];
          //          double y       = node_y[j] - charge_center[1];
          //          double z       = node_z[k] - charge_center[2];
          //          double rsq     = x * x + y * y + z * z;
          //          double wfn_val = *(wfn_ptr);
          //          *(wfn_ptr)     = wfn_val * wfn_val * rsq;
          //          wfn_ptr++;
        }
      }
    }

    wfn_r2_exp[i_wfn] = fe.Compute3DIntegralFromNode(wfn);
    Tucker::MemoryManager::safe_delete(wfn);

    printf("proc %d, wfn %d, <r^{2}>_{\\psi_{%d}}: %.10e\n",
           mpi_rank,
           i_wfn,
           i_wfn,
           wfn_r2_exp[i_wfn]);
  }

  MPI_Allreduce(MPI_IN_PLACE,
                wfn_r2_exp.data(),
                wfn_r2_exp.size(),
                MPI_DOUBLE,
                MPI_SUM,
                MPI_COMM_WORLD);

  if (mpi_rank == 0)
  {
    for (int i_wfn = 0; i_wfn < n_wfn; ++i_wfn)
    {
      fprintf(fileID,
              "wfn %d, <r^{2}>_{\\psi_{%d}}: %.10e\n",
              i_wfn,
              i_wfn,
              wfn_r2_exp[i_wfn]);
    }
    fprintf(fileID, "\n\n");

    std::vector<unsigned> wfn_idx_order(n_wfn, 0);
    std::iota(wfn_idx_order.begin(), wfn_idx_order.end(), 0);
    std::stable_sort(wfn_idx_order.begin(),
                     wfn_idx_order.end(),
                     [&wfn_r2_exp](unsigned idx_1, unsigned idx_2) {
                       return wfn_r2_exp[idx_1] < wfn_r2_exp[idx_2];
                     });

    fprintf(fileID, "wfn order sorted by r^{2} exp value: \n{");
    for (int i = 0; i < n_wfn - 1; ++i)
    {
      fprintf(fileID, "%d, ", wfn_idx_order[i]);
    }
    fprintf(fileID, "%d}\n\n", wfn_idx_order.back());

    fprintf(fileID, "sorted wfn printed out with value\n");
    for (int i = 0; i < n_wfn; ++i)
    {
      fprintf(fileID,
              "wfn [%d]: %.10e\n",
              wfn_idx_order[i],
              wfn_r2_exp[wfn_idx_order[i]]);
    }

    fclose(fileID);
  }

  MPI_Finalize();
  return 0;
}
