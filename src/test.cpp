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

/**
 * This file tries to compute the hartree potential of hydrogen and compare
 * Results are:
 *
 * (all values are squared)
 * error: 7.7868027635927956e-03
 * sum(analytic): 2.1096968469494343e+05
 * sum (kernel): 2.1102665411556582e+05
 * relative error: 3.6909581463575236e-08
 *
 * integral hartree analytic (energy norm): 3.8018412004613142e+03
 * integral hartree kernel (energy norm): 3.8018041402646054e+03
 * integral squared error (error's energy norm): 5.5673535458737287e-06
 * field squared relative error: 1.4643835058650498e-09
 * hartree energy (analytic): 6.2489504937367679e-01
 * hartree energy (kernel): 6.2495181580973647e-01
 */

#include "DataReader.h"
#include <algorithm>
#include <iostream>
#include <vector>
#include <memory>
#include "FE.h"
#include "ConvolutionComputer.h"
#include <Tucker.hpp>
#include <cmath>
#include <numeric>

namespace
{
  void
  ReconstructTensor(Tucker::Tensor  *core,
                    Tucker::Matrix  *factor_mat_x,
                    Tucker::Matrix  *factor_mat_y,
                    Tucker::Matrix  *factor_mat_z,
                    Tucker::Tensor *&tensor)
  {
    Tucker::Tensor *temp;
    temp   = core;
    tensor = Tucker::ttm(temp, 0, factor_mat_x);
    temp   = tensor;
    tensor = Tucker::ttm(temp, 1, factor_mat_y);
    Tucker::MemoryManager::safe_delete(temp);
    temp   = tensor;
    tensor = Tucker::ttm(temp, 2, factor_mat_z);
    Tucker::MemoryManager::safe_delete(temp);
  }
} // namespace

int
main(int argc, char **argv)
{
  std::string   file_name_x = "x_coord.dat";
  std::string   file_name_y = "y_coord.dat";
  std::string   file_name_z = "z_coord.dat";
  Tucker2EI::FE fe_x(file_name_x, 26, Tucker2EI::PT_4);
  Tucker2EI::FE fe_y(file_name_x, 26, Tucker2EI::PT_4);
  Tucker2EI::FE fe_z(file_name_z, 33, Tucker2EI::PT_4);
  Tucker2EI::FE fe_conv_x(file_name_x, 26, Tucker2EI::PT_8);
  Tucker2EI::FE fe_conv_y(file_name_y, 26, Tucker2EI::PT_8);
  Tucker2EI::FE fe_conv_z(file_name_z, 33, Tucker2EI::PT_8);
  std::array<const Tucker2EI::FE *, 3> fe      = {&fe_x, &fe_y, &fe_z};
  std::array<const Tucker2EI::FE *, 3> fe_conv = {&fe_conv_x,
                                                  &fe_conv_y,
                                                  &fe_conv_z};

  //    printf("xquad=[");
  //    for (auto i : fe_x.GetQuadCoord()) printf("%.16f ", i);
  //    printf("];\n");
  //
  //    printf("yquad=[");
  //    for (auto i : fe_y.GetQuadCoord()) printf("%.16f ", i);
  //    printf("];\n");
  //
  //    printf("zquad=[");
  //    for (auto i : fe_z.GetQuadCoord()) printf("%.16f ", i);
  //    printf("];\n");
  //
  //    printf("xquadElectro=[");
  //    for (auto i : fe_conv_x.GetQuadCoord()) printf("%.16f ", i);
  //    printf("];\n");
  //
  //    printf("yquadElectro=[");
  //    for (auto i : fe_conv_y.GetQuadCoord()) printf("%.16f ", i);
  //    printf("];\n");
  //
  //    printf("zquadElectro=[");
  //    for (auto i : fe_conv_z.GetQuadCoord()) printf("%.16f ", i);
  //    printf("];\n");
  //
  //    return 0;

  const auto &x_quad_coord = fe_x.GetQuadCoord();
  const auto &y_quad_coord = fe_y.GetQuadCoord();
  const auto &z_quad_coord = fe_z.GetQuadCoord();

  int quad_size_x = x_quad_coord.size();
  int quad_size_y = y_quad_coord.size();
  int quad_size_z = z_quad_coord.size();

  const auto &x_node_coord = fe_x.GetNodalCoord();
  const auto &y_node_coord = fe_y.GetNodalCoord();
  const auto &z_node_coord = fe_z.GetNodalCoord();

  int node_size_x = x_node_coord.size();
  int node_size_y = y_node_coord.size();
  int node_size_z = z_node_coord.size();

  Tucker::SizeArray quad_size(3);
  quad_size[0] = quad_size_x, quad_size[1] = quad_size_y,
  quad_size[2] = quad_size_z;

  Tucker::SizeArray node_size(3);
  node_size[0] = node_size_x, node_size[1] = node_size_y,
  node_size[2] = node_size_z;

  // Compute analytical hartree at quad
  auto hartree_computer =
    [](const double x, const double y, const double z) -> double {
    double r = std::sqrt(x * x + y * y + z * z);
    return (1.0 / r) - (1.0 + 1.0 / r) * std::exp(-2.0 * r);
  };

  Tucker::Tensor *hartree_quad =
    Tucker::MemoryManager::safe_new<Tucker::Tensor>(quad_size);
  hartree_quad->initialize();
  double *hartree_quad_ptr = hartree_quad->data();
  for (int k = 0, cnt = 0; k < quad_size_z; ++k)
  {
    for (int j = 0; j < quad_size_y; ++j)
    {
      for (int i = 0; i < quad_size_x; ++i)
      {
        hartree_quad_ptr[cnt++] =
          hartree_computer(x_quad_coord[i], y_quad_coord[j], z_quad_coord[k]);
      }
    }
  }


  // Compute analytical rho at quad
  auto rho_computer =
    [](const double x, const double y, const double z) -> double {
    double r = std::sqrt(x * x + y * y + z * z);
    return (1.0 / M_PI) * std::exp(-2.0 * r);
  };

  Tucker::Tensor *rho_node =
    Tucker::MemoryManager::safe_new<Tucker::Tensor>(node_size);
  rho_node->initialize();
  double *rho_node_ptr = rho_node->data();
  for (int k = 0, cnt = 0; k < node_size_z; ++k)
  {
    for (int j = 0; j < node_size_y; ++j)
    {
      for (int i = 0; i < node_size_x; ++i)
      {
        rho_node_ptr[cnt++] =
          rho_computer(x_node_coord[i], y_node_coord[j], z_node_coord[k]);
      }
    }
  }


  // Compute hartree using kernel expansion
  std::string                    file_name_omega = "omega_k35_7e7";
  std::string                    file_name_alpha = "alpha_k35_7e7";
  double                         a_square        = 6.013251619273229e-05;
  Tucker2EI::ConvolutionComputer convolution_computer(file_name_omega,
                                                      file_name_alpha,
                                                      a_square);
  Tucker::Tensor                *conv_quad =
    Tucker::MemoryManager::safe_new<Tucker::Tensor>(quad_size);
  conv_quad->initialize();
  double tol = 1.0e-6;
  convolution_computer.ComputeQuadConvolutionFromNode(
    fe, fe_conv, tol, rho_node, conv_quad);

  double *conv_quad_ptr = conv_quad->data();
  double  error = 0.0, sum_analytic = 0.0, sum_kernel = 0.0;
  for (int i = 0; i < conv_quad->getNumElements(); ++i)
  {
    double diff = conv_quad_ptr[i] - hartree_quad_ptr[i];
    error += diff * diff;
    sum_analytic += hartree_quad_ptr[i] * hartree_quad_ptr[i];
    sum_kernel += conv_quad_ptr[i] * conv_quad_ptr[i];
  }

  printf("(all values are squared)\n"
         "error: %.16e\n"
         "sum(analytic): %.16e\n"
         "sum (kernel): %.16e\n"
         "relative error: %.16e\n",
         error,
         sum_analytic,
         sum_kernel,
         error / sum_analytic);
  std::cout << std::endl;
  double integral_analytic = 0.0, integral_kernel = 0.0, integral_error = 0.0;
  double hartree_energy_kernel = 0.0, hartree_energy_analytic = 0.0;
  ;

  const auto &jw_x = fe_x.GetJacobTimesWeightQuadValues();
  const auto &jw_y = fe_y.GetJacobTimesWeightQuadValues();
  const auto &jw_z = fe_z.GetJacobTimesWeightQuadValues();
  for (int k = 0, cnt = 0; k < quad_size_z; ++k)
  {
    for (int j = 0; j < quad_size_y; ++j)
    {
      for (int i = 0; i < quad_size_x; ++i)
      {
        double jw_3d = jw_x[i] * jw_y[j] * jw_z[k];
        integral_kernel += conv_quad_ptr[cnt] * jw_3d;
        integral_analytic += hartree_quad_ptr[cnt] * jw_3d;
        double diff = conv_quad_ptr[cnt] - hartree_quad_ptr[cnt];
        diff        = diff * diff;
        integral_error += diff * jw_3d;
        double rho =
          rho_computer(x_quad_coord[i], y_quad_coord[j], z_quad_coord[k]);
        hartree_energy_kernel += rho * conv_quad_ptr[cnt] * jw_3d;
        hartree_energy_analytic += rho * hartree_quad_ptr[cnt] * jw_3d;
        cnt++;
      }
    }
  }
  printf("integral hartree analytic (energy norm): %.16e\n"
         "integral hartree kernel (energy norm): %.16e\n"
         "integral squared error (error's energy norm): %.16e\n"
         "field squared relative error: %.16e\n"
         "hartree energy (analytic): %.16e\n"
         "hartree energy (kernel): %.16e\n",
         integral_analytic,
         integral_kernel,
         integral_error,
         integral_error / integral_analytic,
         hartree_energy_analytic,
         hartree_energy_kernel);

  //    std::ofstream fout("out.txt");
  //    for (int k = 0, cnt = 0; k < quad_size_z; ++k)
  //      {
  //        for (int j = 0; j < quad_size_y; ++j)
  //          {
  //            for (int i = 0; i < quad_size_x; ++i)
  //              {
  //                fout << "(" << x_quad_coord[i] << ", " << y_quad_coord[j]
  //                     << "," << z_quad_coord[k] << "): " <<
  //                     conv_quad_ptr[cnt]
  //                     << "\t" << hartree_quad_ptr[cnt] << std::endl;
  //                cnt++;
  //              }
  //          }
  //      }

  return 0;
}
