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

#include <cmath>
#include "BasisSmoothScreener.h"
#include "DataReader.h"

int
main()
{
  // set up meta data
  int n_wfn = 80;

  // x, y, z, charge
  std::vector<std::array<double, 4>> atom;
  atom.emplace_back(std::array<double, 4>{0.0, 0.0, -0.699198615750, 1.0});
  atom.emplace_back(std::array<double, 4>{0.0, 0.0, 0.699198615750, 1.0});

  std::string file_name_x = "x_coord.dat";
  std::string file_name_y = "y_coord.dat";
  std::string file_name_z = "z_coord.dat";

  unsigned int num_ele_x = 26, num_ele_y = 26, num_ele_z = 40;

  double spread_t = 0.8, cutoff = 4.0;

  Tucker2EI::FE3D fe(file_name_x,
                     file_name_y,
                     file_name_z,
                     num_ele_x,
                     num_ele_y,
                     num_ele_z,
                     Tucker2EI::PT_10,
                     Tucker2EI::PT_10,
                     Tucker2EI::PT_10);

  Tucker2EI::BasisSmoothScreener basis_smooth_screener(fe,
                                                       atom,
                                                       spread_t,
                                                       cutoff);

  Tucker::SizeArray sz(3);
  sz[0] = fe.fe_x.GetNumberNodes(), sz[1] = fe.fe_y.GetNumberNodes(),
  sz[2] = fe.fe_z.GetNumberNodes();


  Tucker::Tensor *tensor = Tucker::MemoryManager::safe_new<Tucker::Tensor>(sz);
  Tucker::Tensor *tensor_screen =
    Tucker::MemoryManager::safe_new<Tucker::Tensor>(sz);
  Tucker::Tensor *tensor_screen_sq =
    Tucker::MemoryManager::safe_new<Tucker::Tensor>(sz);

  FILE *original_wfn, *cutoff_wfn, *cutoff_scaled_wfn;
  original_wfn      = fopen("original_wfn.dat", "w");
  cutoff_wfn        = fopen("cutoff_wfn.dat", "w");
  cutoff_scaled_wfn = fopen("cutoff_scaled_wfn.dat", "w");

  int x_0_id = (sz[0] - 1) / 2;
  int y_0_id = (sz[1] - 1) / 2;
  for (int i = 0; i < n_wfn; ++i)
  {
    printf("processing wfn %d...\n", i);
    tensor->initialize();
    tensor_screen->initialize();
    tensor_screen_sq->initialize();
    Tucker2EI::utils::TensorReader("waveFunc_" + std::to_string(i) + ".dat",
                                   fe.fe_x.GetNumberNodes(),
                                   fe.fe_y.GetNumberNodes(),
                                   fe.fe_z.GetNumberNodes(),
                                   tensor);

    double *basis_smooth_screener_ptr =
      basis_smooth_screener.GetScreener()->data();
    double *tensor_ptr           = tensor->data();
    double *tensor_screen_ptr    = tensor_screen->data();
    double *tensor_screen_sq_ptr = tensor_screen_sq->data();
    for (int j = 0; j < tensor->getNumElements(); ++j)
    {
      double val              = basis_smooth_screener_ptr[j] * tensor_ptr[j];
      tensor_screen_ptr[j]    = val;
      tensor_screen_sq_ptr[j] = val * val;
    }

    double norm = fe.Compute3DIntegralFromNode(tensor_screen_sq);
    norm        = std::sqrt(norm);
    printf("norm %d: %.6f\n", i, norm);
    for (int j = 0; j < sz[2]; ++j)
    {
      fprintf(original_wfn,
              "%.16e, ",
              *(tensor_ptr + x_0_id + y_0_id * sz[0] + j * sz[0] * sz[1]));
      fprintf(cutoff_wfn,
              "%.16e, ",
              *(tensor_screen_ptr + x_0_id + y_0_id * sz[0] +
                j * sz[0] * sz[1]));
    }

    for (int j = 0; j < tensor->getNumElements(); ++j)
    {
      tensor_screen_ptr[j] /= norm;
    }
    for (int j = 0; j < sz[2]; ++j)
    {
      fprintf(cutoff_scaled_wfn,
              "%.16e, ",
              *(tensor_screen_ptr + x_0_id + y_0_id * sz[0] +
                j * sz[0] * sz[1]));
    }
    Tucker2EI::utils::TensorPrint("cutoff_waveFunc_" + std::to_string(i) +
                                    ".dat",
                                  tensor_screen);
    fprintf(original_wfn, "\n");
    fprintf(cutoff_wfn, "\n");
    fprintf(cutoff_scaled_wfn, "\n");
  }

  fclose(original_wfn);
  fclose(cutoff_wfn);
  fclose(cutoff_scaled_wfn);

  Tucker::MemoryManager::safe_delete(tensor);

  return 0;
}