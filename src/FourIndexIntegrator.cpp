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

#include "FourIndexIntegrator.h"
#include "DataReader.h"
#include "BlasWrapper.h"

#include <chrono>
#include <cmath>
using namespace std::chrono;

Tucker2EI::FourIndexIntegrator::FourIndexIntegrator(
  const Tucker2EI::FE3D                &fe,
  const Tucker2EI::FE3D                &fe_conv,
  const Tucker2EI::ConvolutionComputer &conv_comp,
  double                                decomp_tol)
  : Integrator(fe)
  , fe_conv_(fe_conv)
  , conv_comp_(conv_comp)
  , decomp_tol_(decomp_tol)
{}

double
Tucker2EI::FourIndexIntegrator::Integrate(std::array<int, 4> f_idx)
{
  // read in and compute wfn_k * wfn_l on nodes
  auto            start = high_resolution_clock::now();
  Tucker::Tensor *wfn;
  ReadWfn(f_idx[2], wfn);
  Tucker::Tensor *wfn_prod;
  ReadWfn(f_idx[3], wfn_prod);
  auto end = high_resolution_clock::now();
  printf("read in: %.4f\n",
         double(duration_cast<milliseconds>(end - start).count()) / 1.0e3);

  double *wfn_prod_ptr = wfn_prod->data();
  double *wfn_ptr      = wfn->data();
  start                = high_resolution_clock::now();
  for (int i = 0; i < wfn_prod->getNumElements(); ++i)
  {
    wfn_prod_ptr[i] *= wfn_ptr[i];
  }
  Tucker::MemoryManager::safe_delete(wfn);
  end = high_resolution_clock::now();
  printf("nodal product %.4f\n",
         double(duration_cast<milliseconds>(end - start).count()) / 1.0e3);

  // compute convolution integral on quadrature points
  start = high_resolution_clock::now();
  Tucker::Tensor *conv_wfn_quad =
    Tucker::MemoryManager::safe_new<Tucker::Tensor>(num_quad_);
  conv_wfn_quad->initialize();
  conv_comp_.ComputeQuadConvolutionFromNode(
    fe_.fe, fe_conv_.fe, decomp_tol_, wfn_prod, conv_wfn_quad);
  end = high_resolution_clock::now();
  printf("1d conv: %.4f\n",
         double(duration_cast<milliseconds>(end - start).count()) / 1.0e3);

  Tucker::MemoryManager::safe_delete(wfn_prod);


  //  start = high_resolution_clock::now();
  //  conv_comp_.ComputeQuadConvolutionFromNodeOnNodeInplace(
  //      fe_.fe, fe_conv_.fe, decomp_tol_, wfn_prod);
  //  std::vector<double> cellwfn =
  //  std::move(fe_.Convert3DMeshToCellwiseVector(wfn_prod));
  //  Tucker::MemoryManager::safe_delete(wfn_prod);
  //  std::vector<double>
  //  cellquad(fe_.GetNum3DEle()*fe_.GetNumQuadPointsPer3Dele(), 0.0);
  //    const auto &A = fe_.GetElementalShapeFunctionAtQuadPoints();
  //  Tucker2EI::blas_wrapper::Dgemm(fe_.GetNumQuadPointsPer3Dele(),
  //                                 fe_.GetNumNodesPer3Dele(),
  //                                 fe_.GetNum3DEle(),
  //                                 A.data(),
  //                                 cellwfn.data(),
  //                                 cellquad.data());
  //  end = high_resolution_clock::now();
  //  printf("1d conv: %.4f\n",
  //  double(duration_cast<milliseconds>(end-start).count())/1.0e3);

  //  int num_quad_per_cell_x = fe_.fe_x.GetNumberQuadPointsPerElement();
  //  int num_quad_per_cell_y = fe_.fe_y.GetNumberQuadPointsPerElement();
  //  int num_quad_per_cell_z = fe_.fe_z.GetNumberQuadPointsPerElement();
  //
  //  int num_quad_x = fe_.fe_x.GetNumberQuadPoints();
  //  int num_quad_y = fe_.fe_y.GetNumberQuadPoints();
  //  int num_quad_z = fe_.fe_z.GetNumberQuadPoints();
  //
  //  int num_ele_x = fe_.fe_x.GetNumberElements();
  //  int num_ele_y = fe_.fe_y.GetNumberElements();
  //  int num_ele_z = fe_.fe_z.GetNumberElements();

  //  double error = 0.0, denom = 0.0;
  //  double *tempd1 = conv_wfn_quad->data();
  //  for (int ele_k = 0, cnt = 0; ele_k < num_ele_z; ++ele_k) {
  //    for (int ele_j = 0; ele_j < num_ele_y; ++ele_j) {
  //      for (int ele_i = 0; ele_i < num_ele_x; ++ele_i) {
  //        for (int node_k = 0; node_k < num_quad_per_cell_z; ++node_k) {
  //          for (int node_j = 0; node_j < num_quad_per_cell_y; ++node_j) {
  //            for (int node_i = 0; node_i < num_quad_per_cell_x; ++node_i) {
  //              int ii = ele_i * num_quad_per_cell_x + node_i;
  //              int jj = ele_j * num_quad_per_cell_y + node_j;
  //              int kk = ele_k * num_quad_per_cell_z + node_k;
  //              int idx = ii + jj * num_quad_x + kk * num_quad_x * num_quad_y;
  //              double diff = cellquad[cnt] - tempd1[idx];
  //              error += diff*diff;
  //              denom += tempd1[idx]*tempd1[idx];
  //              cnt++;
  //            }
  //          }
  //        }
  //      }
  //    }
  //  }
  //  printf("conv quad error: %.4e, rel error %.4e\n", std::sqrt(error),
  //  std::sqrt(error/denom));
  //
  //  for (int ele_k = 0, cnt = 0; ele_k < num_ele_z; ++ele_k) {
  //    for (int ele_j = 0; ele_j < num_ele_y; ++ele_j) {
  //      for (int ele_i = 0; ele_i < num_ele_x; ++ele_i) {
  //        for (int node_k = 0; node_k < num_quad_per_cell_z; ++node_k) {
  //          for (int node_j = 0; node_j < num_quad_per_cell_y; ++node_j) {
  //            for (int node_i = 0; node_i < num_quad_per_cell_x; ++node_i) {
  //              int ii = ele_i * num_quad_per_cell_x + node_i;
  //              int jj = ele_j * num_quad_per_cell_y + node_j;
  //              int kk = ele_k * num_quad_per_cell_z + node_k;
  //              int idx = ii + jj * num_quad_x + kk * num_quad_x * num_quad_y;
  //              tempd1[idx] = cellquad[cnt];
  ////              double diff = cellquad[cnt] - tempd1[idx];
  ////              error += diff*diff;
  ////              denom += tempd1[idx]*tempd1[idx];
  //              cnt++;
  //            }
  //          }
  //        }
  //      }
  //    }
  //  }

  //  Tucker::Tensor *conv_wfn_quad2 =
  //      Tucker::MemoryManager::safe_new<Tucker::Tensor>(num_quad_);
  //  conv_wfn_quad->initialize();
  //
  //  Tucker::MemoryManager::safe_delete(wfn_prod);

  // compte wfn_i*wfn_j on nodes
  start = high_resolution_clock::now();
  ReadWfn(f_idx[0], wfn);
  ReadWfn(f_idx[1], wfn_prod);
  end = high_resolution_clock::now();
  printf("read in 2: %.4f\n",
         double(duration_cast<milliseconds>(end - start).count()) / 1.0e3);
  start        = high_resolution_clock::now();
  wfn_prod_ptr = wfn_prod->data();
  wfn_ptr      = wfn->data();
  for (int i = 0; i < wfn_prod->getNumElements(); ++i)
  {
    wfn_prod_ptr[i] *= wfn_ptr[i];
  }
  Tucker::MemoryManager::safe_delete(wfn);
  end = high_resolution_clock::now();
  printf("prod 2: %.4f\n",
         double(duration_cast<milliseconds>(end - start).count()) / 1.0e3);

  // interpolate wfn_i*wfn_j onto quad points
  start = high_resolution_clock::now();
  Tucker::Tensor *wfn_ij_conv_wfn_kl_quad =
    Tucker::MemoryManager::safe_new<Tucker::Tensor>(num_quad_);
  wfn_ij_conv_wfn_kl_quad->initialize();
  fe_.InterpolateNode2Quad(wfn_prod, wfn_ij_conv_wfn_kl_quad);
  Tucker::MemoryManager::safe_delete(wfn_prod);
  end = high_resolution_clock::now();
  printf("node 2 quad: %.4f\n",
         double(duration_cast<milliseconds>(end - start).count()) / 1.0e3);

  // compute wfn_ij * conv_wfn_kl and integrate over the field
  start                               = high_resolution_clock::now();
  double *wfn_ij_conv_wfn_kl_quad_ptr = wfn_ij_conv_wfn_kl_quad->data();
  double *conv_wfn_kl_quad_ptr        = conv_wfn_quad->data();
  for (int i = 0; i < wfn_ij_conv_wfn_kl_quad->getNumElements(); ++i)
  {
    wfn_ij_conv_wfn_kl_quad_ptr[i] *= conv_wfn_kl_quad_ptr[i];
  }
  Tucker::MemoryManager::safe_delete(conv_wfn_quad);
  end = high_resolution_clock::now();
  printf("quad product: %.4f\n",
         double(duration_cast<milliseconds>(end - start).count()) / 1.0e3);

  start         = high_resolution_clock::now();
  double result = fe_.Compute3DIntegralFromQuad(wfn_ij_conv_wfn_kl_quad);
  end           = high_resolution_clock::now();
  printf("integrate from quad: %.4f\n",
         double(duration_cast<milliseconds>(end - start).count()) / 1.0e3);
  Tucker::MemoryManager::safe_delete(wfn_ij_conv_wfn_kl_quad);

  return result;
}