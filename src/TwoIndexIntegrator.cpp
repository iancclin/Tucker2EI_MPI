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

#include <cmath>
#include "TwoIndexIntegrator.h"
#include "BlasWrapper.h"
#include "ExceptionHandler.h"

Tucker2EI::TwoIndexIntegrator::TwoIndexIntegrator(
  const Tucker2EI::FE3D   &fe,
  const std::vector<Atom> &nuclei_coord)
  : Integrator(fe)
  , nuclei_coord_(nuclei_coord)
{}

double
Tucker2EI::TwoIndexIntegrator::ExternalIntegrator(int idx_i, int idx_j)
{
  Tucker::Tensor *wfn;
  ReadWfn(idx_i, wfn);
  Tucker::Tensor *wfn_prod;
  ReadWfn(idx_j, wfn_prod);

  double *wfn_prod_ptr = wfn_prod->data();
  double *wfn_ptr      = wfn->data();
  for (int i = 0; i < wfn_prod->getNumElements(); ++i)
  {
    wfn_prod_ptr[i] *= wfn_ptr[i];
  }
  Tucker::MemoryManager::safe_delete(wfn);

  Tucker::Tensor *wfn_prod_quad =
    Tucker::MemoryManager::safe_new<Tucker::Tensor>(num_quad_);
  wfn_prod_quad->initialize();

  fe_.InterpolateNode2Quad(wfn_prod, wfn_prod_quad);

  Tucker::MemoryManager::safe_delete(wfn_prod);

  const auto &x_quad = fe_.fe_x.GetQuadCoord();
  const auto &y_quad = fe_.fe_y.GetQuadCoord();
  const auto &z_quad = fe_.fe_z.GetQuadCoord();

  Tucker::Tensor *wfn_prod_by_r =
    Tucker::MemoryManager::safe_new<Tucker::Tensor>(num_quad_);
  double *wfn_prod_by_r_ptr = wfn_prod_by_r->data();
  double *wfn_prod_quad_ptr = wfn_prod_quad->data();

  auto r_computer =
    [](const double x, const double y, const double z, const Atom &atom) {
      double x_square = (x - atom[0]) * (x - atom[0]);
      double y_square = (y - atom[1]) * (y - atom[1]);
      double z_square = (z - atom[2]) * (z - atom[2]);
      return std::sqrt(x_square + y_square + z_square);
    };

  double result = 0.0;
  for (int i_atom = 0; i_atom < nuclei_coord_.size(); ++i_atom)
  {
    wfn_prod_by_r->initialize();
    std::copy(wfn_prod_quad_ptr,
              wfn_prod_quad_ptr + wfn_prod_quad->getNumElements(),
              wfn_prod_by_r_ptr);

    double charge = nuclei_coord_[i_atom][3];

    for (int k = 0, cnt = 0; k < num_quad_[2]; ++k)
    {
      for (int j = 0; j < num_quad_[1]; ++j)
      {
        for (int i = 0; i < num_quad_[0]; ++i)
        {
          wfn_prod_by_r_ptr[cnt++] *=
            charge /
            r_computer(x_quad[i], y_quad[j], z_quad[k], nuclei_coord_[i_atom]);
        }
      }
    }
    result += fe_.Compute3DIntegralFromQuad(wfn_prod_by_r);
  }
  Tucker::MemoryManager::safe_delete(wfn_prod_by_r);
  Tucker::MemoryManager::safe_delete(wfn_prod_quad);
  return result;
}

double
Tucker2EI::TwoIndexIntegrator::OverlapIntegrator(int idx_i, int idx_j)
{
  Tucker::Tensor *wfn;
  ReadWfn(idx_i, wfn);
  Tucker::Tensor *wfn_prod;
  ReadWfn(idx_j, wfn_prod);

  double *wfn_prod_ptr = wfn_prod->data();
  double *wfn_ptr      = wfn->data();
  for (int i = 0; i < wfn_prod->getNumElements(); ++i)
  {
    wfn_prod_ptr[i] *= wfn_ptr[i];
  }
  Tucker::MemoryManager::safe_delete(wfn);

  double overlap = fe_.Compute3DIntegralFromNode(wfn_prod);

  Tucker::MemoryManager::safe_delete(wfn_prod);

  return overlap;
}

double
Tucker2EI::TwoIndexIntegrator::KineticIntegrator(int idx_i, int idx_j)
{
  Tucker::Tensor *wfn_j;
  ReadWfn(idx_j, wfn_j);
  Tucker::Tensor *wfn_i;
  ReadWfn(idx_i, wfn_i);

  double result = KineticIntegrator(wfn_i, wfn_j);

  Tucker::MemoryManager::safe_delete(wfn_j);
  Tucker::MemoryManager::safe_delete(wfn_i);


  return result;
}

double
Tucker2EI::TwoIndexIntegrator::KineticIntegrator(Tucker::Tensor *wfn_i,
                                                 Tucker::Tensor *wfn_j)
{
  unsigned int num_ele_x     = fe_.fe_x.GetNumberElements();
  unsigned int num_ele_y     = fe_.fe_y.GetNumberElements();
  unsigned int num_ele_z     = fe_.fe_z.GetNumberElements();
  unsigned int num_nodes_3du = fe_.GetNumNodesPer3Dele();

  Tucker::Tensor *dNidNj_wfn_j =
    Tucker::MemoryManager::safe_new<Tucker::Tensor>(num_node_);
  dNidNj_wfn_j->initialize();

  for (int k_ele = 0, ele = 0; k_ele < num_ele_z; ++k_ele)
  {
    for (int j_ele = 0; j_ele < num_ele_y; ++j_ele)
    {
      for (int i_ele = 0; i_ele < num_ele_x; ++i_ele, ++ele)
      {
        auto dNidNj = fe_.GetElementalDNiDNj(i_ele, j_ele, k_ele);

        std::vector<double> ele_nodal_val =
          fe_.ExtractElementalVector(i_ele, j_ele, k_ele, wfn_j->data());
        std::vector<double> result(num_nodes_3du, 0.0);
        Tucker2EI::blas_wrapper::Dgemvn(num_nodes_3du,
                                        num_nodes_3du,
                                        dNidNj.data(),
                                        ele_nodal_val.data(),
                                        result.data());
        fe_.WriteElementalVector(
          i_ele, j_ele, k_ele, result, dNidNj_wfn_j->data());
      }
    }
  }

  double *wfn_i_ptr        = wfn_i->data();
  double *dNidNj_wfn_j_ptr = dNidNj_wfn_j->data();

  double result       = 0.0;
  int    num_nodes_3d = wfn_i->getNumElements();
  for (int i = 0; i < num_nodes_3d; ++i)
  {
    result += wfn_i_ptr[i] * dNidNj_wfn_j_ptr[i];
  }
  Tucker::MemoryManager::safe_delete(dNidNj_wfn_j);

  return result;
}
