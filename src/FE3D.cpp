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

#include <chrono>
#include "FE3D.h"
#include "BlasWrapper.h"
#include "ExceptionHandler.h"
Tucker2EI::FE3D::FE3D(const std::string    &file_name_x,
                      const std::string    &file_name_y,
                      const std::string    &file_name_z,
                      unsigned int          num_ele_x,
                      unsigned int          num_ele_y,
                      unsigned int          num_ele_z,
                      Tucker2EI::QuadPoints quad_points_x,
                      Tucker2EI::QuadPoints quad_points_y,
                      Tucker2EI::QuadPoints quad_points_z)
  : fe_x(file_name_x, num_ele_x, quad_points_x)
  , fe_y(file_name_y, num_ele_y, quad_points_y)
  , fe_z(file_name_z, num_ele_z, quad_points_z)
  , num_3d_ele_(num_ele_x * num_ele_y * num_ele_z)
{
  fe[0] = &fe_x;
  fe[1] = &fe_y;
  fe[2] = &fe_z;

  num_nodes_per_3dele_ = fe_x.GetNumberNodesPerElement() *
                         fe_y.GetNumberNodesPerElement() *
                         fe_z.GetNumberNodesPerElement();

  num_quad_points_per_3dele_ = fe_x.GetNumberQuadPointsPerElement() *
                               fe_y.GetNumberQuadPointsPerElement() *
                               fe_z.GetNumberQuadPointsPerElement();

  elemental_shape_function_at_quad_points_.assign(num_quad_points_per_3dele_ *
                                                    num_nodes_per_3dele_,
                                                  0.0);

  ComputeShapeFunctionAtQuadPoints(elemental_shape_function_at_quad_points_);

  for (int dim = 0; dim < 3; ++dim)
  {
    elemental_dNIdNJ_comp_[dim].assign(num_nodes_per_3dele_ *
                                         num_nodes_per_3dele_,
                                       0.0);
    jacob_comp_[dim].assign(num_3d_ele_, 0.0);
  }
  ComputedNIdNJComponents(elemental_dNIdNJ_comp_, jacob_comp_);
}

// this part should be optimized, especially the part copying data back to quad
// tensor
void
Tucker2EI::FE3D::InterpolateNode2Quad(const Tucker::Tensor *nodal_val,
                                      Tucker::Tensor       *quad_val) const
{
  TUCKER2EI_ASSERTWITHSTRING((nodal_val->size(0) == fe_x.GetNumberNodes()) ||
                               (nodal_val->size(1) == fe_y.GetNumberNodes()) ||
                               (nodal_val->size(2) == fe_z.GetNumberNodes()),
                             "The size nodal tensor is not consistent with "
                             "the FE objects.");
  TUCKER2EI_ASSERTWITHSTRING(
    (quad_val->size(0) == fe_x.GetNumberQuadPoints()) ||
      (quad_val->size(1) == fe_y.GetNumberQuadPoints()) ||
      (quad_val->size(2) == fe_z.GetNumberQuadPoints()),
    "The size quad tensor is not consistent with "
    "the FE objects.");

  std::vector<double> elemental_nodal_vals(num_nodes_per_3dele_, 0.0);
  std::vector<double> elemental_quad_vals(num_quad_points_per_3dele_, 0.0);
  const double       *node_val_ptr = nodal_val->data();
  double             *quad_val_ptr = quad_val->data();

  for (int i_ele = 0; i_ele < fe_x.GetNumberElements(); ++i_ele)
  {
    for (int j_ele = 0; j_ele < fe_y.GetNumberElements(); ++j_ele)
    {
      for (int k_ele = 0; k_ele < fe_z.GetNumberElements(); ++k_ele)
      {
        std::vector<double> v_temp_node(num_nodes_per_3dele_, 0.0),
          v_temp_quad(num_quad_points_per_3dele_, 0.0);
        ExtractElementalVector(i_ele,
                               j_ele,
                               k_ele,
                               fe_x.GetNumberNodesPerElement(),
                               fe_y.GetNumberNodesPerElement(),
                               fe_z.GetNumberNodesPerElement(),
                               node_val_ptr,
                               fe_x.GetNumberNodes(),
                               fe_y.GetNumberNodes(),
                               fe_z.GetNumberNodes(),
                               v_temp_node);

        Tucker2EI::blas_wrapper::Dgemvn(
          num_quad_points_per_3dele_,
          num_nodes_per_3dele_,
          elemental_shape_function_at_quad_points_.data(),
          v_temp_node.data(),
          v_temp_quad.data());

        WriteElementalQuadVector(i_ele,
                                 j_ele,
                                 k_ele,
                                 fe_x.GetNumberQuadPointsPerElement(),
                                 fe_y.GetNumberQuadPointsPerElement(),
                                 fe_z.GetNumberQuadPointsPerElement(),
                                 v_temp_quad,
                                 fe_x.GetNumberQuadPoints(),
                                 fe_y.GetNumberQuadPoints(),
                                 fe_z.GetNumberQuadPoints(),
                                 quad_val_ptr);
      }
    }
  }
}

double
Tucker2EI::FE3D::Compute3DIntegralFromNode(
  const Tucker::Tensor *nodal_val) const
{
  TUCKER2EI_ASSERTWITHSTRING((nodal_val->size(0) == fe_x.GetNumberNodes()) ||
                               (nodal_val->size(1) == fe_y.GetNumberNodes()) ||
                               (nodal_val->size(2) == fe_z.GetNumberNodes()),
                             "The size nodal tensor is not consistent with "
                             "the FE objects.");

  Tucker::SizeArray quad_size(3);
  for (int dim = 0; dim < 3; ++dim)
  {
    quad_size[dim] = fe[dim]->GetNumberQuadPoints();
  }

  Tucker::Tensor *quad_val =
    Tucker::MemoryManager::safe_new<Tucker::Tensor>(quad_size);
  quad_val->initialize();

  InterpolateNode2Quad(nodal_val, quad_val);

  double result = Compute3DIntegralFromQuad(quad_val);

  Tucker::MemoryManager::safe_delete(quad_val);

  return result;
}

double
Tucker2EI::FE3D::Compute3DIntegralFromQuad(const Tucker::Tensor *quad_val) const
{
  TUCKER2EI_ASSERTWITHSTRING(
    (quad_val->size(0) == fe_x.GetNumberQuadPoints()) ||
      (quad_val->size(1) == fe_y.GetNumberQuadPoints()) ||
      (quad_val->size(2) == fe_z.GetNumberQuadPoints()),
    "The size quad tensor is not consistent with "
    "the FE objects.");

  double        result       = 0.0;
  const auto   &jw_x         = fe_x.GetJacobTimesWeightQuadValues();
  const auto   &jw_y         = fe_y.GetJacobTimesWeightQuadValues();
  const auto   &jw_z         = fe_z.GetJacobTimesWeightQuadValues();
  const double *quad_val_ptr = quad_val->data();
  for (int k = 0, cnt = 0; k < fe_z.GetNumberQuadPoints(); ++k)
  {
    for (int j = 0; j < fe_y.GetNumberQuadPoints(); ++j)
    {
      for (int i = 0; i < fe_x.GetNumberQuadPoints(); ++i)
      {
        double jw_3d = jw_x[i] * jw_y[j] * jw_z[k];
        result += quad_val_ptr[cnt] * jw_3d;
        cnt++;
      }
    }
  }
  return result;
}

std::vector<double>
Tucker2EI::FE3D::ExtractElementalVector(int           ele_id_x,
                                        int           ele_id_y,
                                        int           ele_id_z,
                                        const double *nodal_tensor) const
{
  std::vector<double> result(num_nodes_per_3dele_, 0.0);
  ExtractElementalVector(ele_id_x,
                         ele_id_y,
                         ele_id_z,
                         fe_x.GetNumberNodesPerElement(),
                         fe_y.GetNumberNodesPerElement(),
                         fe_z.GetNumberNodesPerElement(),
                         nodal_tensor,
                         fe_x.GetNumberNodes(),
                         fe_y.GetNumberNodes(),
                         fe_z.GetNumberNodes(),
                         result);
  return result;
}

void
Tucker2EI::FE3D::WriteElementalVector(int                        ele_id_x,
                                      int                        ele_id_y,
                                      int                        ele_id_z,
                                      const std::vector<double> &val,
                                      double *nodal_tensor) const
{
  WriteElementalVector(ele_id_x,
                       ele_id_y,
                       ele_id_z,
                       fe_x.GetNumberNodesPerElement(),
                       fe_y.GetNumberNodesPerElement(),
                       fe_z.GetNumberNodesPerElement(),
                       val,
                       fe_x.GetNumberNodes(),
                       fe_y.GetNumberNodes(),
                       fe_z.GetNumberNodes(),
                       nodal_tensor);
}

void
Tucker2EI::FE3D::ComputeShapeFunctionAtQuadPoints(
  std::vector<double> &elemental_shape_function_at_quad_points) const
{
  const auto &shape_func_at_quad_x =
    fe_x.GetElementalShapeFunctionAtQuadPoints();
  const auto &shape_func_at_quad_y =
    fe_y.GetElementalShapeFunctionAtQuadPoints();
  const auto &shape_func_at_quad_z =
    fe_z.GetElementalShapeFunctionAtQuadPoints();

  int elemental_num_nodes_x = fe_x.GetNumberNodesPerElement();
  int elemental_num_nodes_y = fe_y.GetNumberNodesPerElement();
  int elemental_num_nodes_z = fe_z.GetNumberNodesPerElement();

  int elemental_num_quad_x = fe_x.GetNumberQuadPointsPerElement();
  int elemental_num_quad_y = fe_y.GetNumberQuadPointsPerElement();
  int elemental_num_quad_z = fe_z.GetNumberQuadPointsPerElement();

  for (int k_node = 0, cnt = 0; k_node < elemental_num_nodes_z; ++k_node)
  {
    for (int j_node = 0; j_node < elemental_num_nodes_y; ++j_node)
    {
      for (int i_node = 0; i_node < elemental_num_nodes_x; ++i_node)
      {
        for (int k_quad = 0; k_quad < elemental_num_quad_z; ++k_quad)
        {
          for (int j_quad = 0; j_quad < elemental_num_quad_y; ++j_quad)
          {
            for (int i_quad = 0; i_quad < elemental_num_quad_x; ++i_quad)
            {
              double x_val =
                shape_func_at_quad_x[i_node * elemental_num_quad_x + i_quad];
              double y_val =
                shape_func_at_quad_y[j_node * elemental_num_quad_y + j_quad];
              double z_val =
                shape_func_at_quad_z[k_node * elemental_num_quad_z + k_quad];
              elemental_shape_function_at_quad_points[cnt++] =
                x_val * y_val * z_val;
            }
          }
        }
      }
    }
  }
}

void
Tucker2EI::FE3D::ComputedNIdNJComponents(
  std::array<std::vector<double>, 3> &elemental_dNIdNJ_comp,
  std::array<std::vector<double>, 3> &jacob_comp) const
{
  int num_ele_nodes_x = fe_x.GetNumberNodesPerElement();
  int num_ele_nodes_y = fe_y.GetNumberNodesPerElement();
  int num_ele_nodes_z = fe_z.GetNumberNodesPerElement();

  const auto &dNi_dNj_x = fe_x.GetElementalDNiDNj();
  const auto &dNi_dNj_y = fe_y.GetElementalDNiDNj();
  const auto &dNi_dNj_z = fe_z.GetElementalDNiDNj();

  const auto &Ni_Nj_x = fe_x.GetElementalNiNj();
  const auto &Ni_Nj_y = fe_y.GetElementalNiNj();
  const auto &Ni_Nj_z = fe_z.GetElementalNiNj();


  for (int kJ = 0, cnt = 0; kJ < num_ele_nodes_z; ++kJ)
  {
    for (int jJ = 0; jJ < num_ele_nodes_y; ++jJ)
    {
      for (int iJ = 0; iJ < num_ele_nodes_x; ++iJ)
      {
        for (int kI = 0; kI < num_ele_nodes_z; ++kI)
        {
          for (int jI = 0; jI < num_ele_nodes_y; ++jI)
          {
            for (int iI = 0; iI < num_ele_nodes_x; ++iI)
            {
              elemental_dNIdNJ_comp[0][cnt] =
                dNi_dNj_x[iI][iJ] * Ni_Nj_y[jI][jJ] * Ni_Nj_z[kI][kJ];
              elemental_dNIdNJ_comp[1][cnt] =
                Ni_Nj_x[iI][iJ] * dNi_dNj_y[jI][jJ] * Ni_Nj_z[kI][kJ];
              elemental_dNIdNJ_comp[2][cnt] =
                Ni_Nj_x[iI][iJ] * Ni_Nj_y[jI][jJ] * dNi_dNj_z[kI][kJ];
              cnt++;
            }
          }
        }
      }
    }
  }

  int num_ele_x = fe_x.GetNumberElements();
  int num_ele_y = fe_y.GetNumberElements();
  int num_ele_z = fe_z.GetNumberElements();

  const auto &jacob_x = fe_x.GetJacobianOnEle();
  const auto &jacob_y = fe_y.GetJacobianOnEle();
  const auto &jacob_z = fe_z.GetJacobianOnEle();

  const auto &inv_jacob_x = fe_x.GetInvjacobianOnEle();
  const auto &inv_jacob_y = fe_y.GetInvjacobianOnEle();
  const auto &inv_jacob_z = fe_z.GetInvjacobianOnEle();

  for (int k_ele = 0, ele = 0; k_ele < num_ele_z; ++k_ele)
  {
    for (int j_ele = 0; j_ele < num_ele_y; ++j_ele)
    {
      for (int i_ele = 0; i_ele < num_ele_x; ++i_ele, ++ele)
      {
        jacob_comp[0][ele] =
          inv_jacob_x[i_ele] * jacob_y[j_ele] * jacob_z[k_ele];
        jacob_comp[1][ele] =
          jacob_x[i_ele] * inv_jacob_y[j_ele] * jacob_z[k_ele];
        jacob_comp[2][ele] =
          jacob_x[i_ele] * jacob_y[j_ele] * inv_jacob_z[k_ele];
      }
    }
  }
}

void
Tucker2EI::FE3D::ExtractElementalVector(
  const int            ele_id_x,
  const int            ele_id_y,
  const int            ele_id_z,
  const int            ele_num_nodes_x,
  const int            ele_num_nodes_y,
  const int            ele_num_nodes_z,
  const double        *nodal_tensor,
  const int            tensor_size_x,
  const int            tensor_size_y,
  const int            tensor_size_z,
  std::vector<double> &elemental_vector) const
{
  elemental_vector.assign(ele_num_nodes_x * ele_num_nodes_y * ele_num_nodes_z,
                          0.0);
  int ele_i_start = ele_id_x * (ele_num_nodes_x - 1);
  int ele_j_start = ele_id_y * (ele_num_nodes_y - 1);
  int ele_k_start = ele_id_z * (ele_num_nodes_z - 1);
  for (int k_node = ele_k_start, cnt = 0;
       k_node < ele_k_start + ele_num_nodes_z;
       ++k_node)
  {
    for (int j_node = ele_j_start; j_node < ele_j_start + ele_num_nodes_y;
         ++j_node)
    {
      for (int i_node = ele_i_start; i_node < ele_i_start + ele_num_nodes_x;
           ++i_node)
      {
        int idx = i_node + j_node * tensor_size_x +
                  k_node * tensor_size_x * tensor_size_y;
        elemental_vector[cnt] = nodal_tensor[idx];
        cnt++;
      }
    }
  }
}

void
Tucker2EI::FE3D::WriteElementalVector(
  int                        ele_id_x,
  int                        ele_id_y,
  int                        ele_id_z,
  int                        ele_num_nodes_x,
  int                        ele_num_nodes_y,
  int                        ele_num_nodes_z,
  const std::vector<double> &elemental_vector,
  int                        tensor_size_x,
  int                        tensor_size_y,
  int                        tensor_size_z,
  double                    *nodal_tensor) const
{
  int ele_i_start = ele_id_x * (ele_num_nodes_x - 1);
  int ele_j_start = ele_id_y * (ele_num_nodes_y - 1);
  int ele_k_start = ele_id_z * (ele_num_nodes_z - 1);
  for (int k_node = ele_k_start, cnt = 0;
       k_node < ele_k_start + ele_num_nodes_z;
       ++k_node)
  {
    for (int j_node = ele_j_start; j_node < ele_j_start + ele_num_nodes_y;
         ++j_node)
    {
      for (int i_node = ele_i_start; i_node < ele_i_start + ele_num_nodes_x;
           ++i_node)
      {
        int idx = i_node + j_node * tensor_size_x +
                  k_node * tensor_size_x * tensor_size_y;
        nodal_tensor[idx] += elemental_vector[cnt];
        cnt++;
      }
    }
  }
}

void
Tucker2EI::FE3D::WriteElementalQuadVector(
  const int                  ele_id_x,
  const int                  ele_id_y,
  const int                  ele_id_z,
  const int                  ele_num_quads_x,
  const int                  ele_num_quads_y,
  const int                  ele_num_quads_z,
  const std::vector<double> &quad_vector,
  const int                  tensor_size_x,
  const int                  tensor_size_y,
  const int                  tensor_size_z,
  double                    *quad_tensor) const
{
  int ele_i_start = ele_id_x * ele_num_quads_x;
  int ele_j_start = ele_id_y * ele_num_quads_y;
  int ele_k_start = ele_id_z * ele_num_quads_z;
  for (int k_quad = ele_k_start, cnt = 0;
       k_quad < ele_k_start + ele_num_quads_z;
       ++k_quad)
  {
    for (int j_quad = ele_j_start; j_quad < ele_j_start + ele_num_quads_y;
         ++j_quad)
    {
      for (int i_quad = ele_i_start; i_quad < ele_i_start + ele_num_quads_x;
           ++i_quad)
      {
        int idx = i_quad + j_quad * tensor_size_x +
                  k_quad * tensor_size_x * tensor_size_y;
        quad_tensor[idx] = quad_vector[cnt];
        cnt++;
      }
    }
  }
}
std::vector<double>
Tucker2EI::FE3D::GetElementalDNiDNj(int ele_id_x,
                                    int ele_id_y,
                                    int ele_id_z) const
{
  std::vector<double> result(num_nodes_per_3dele_ * num_nodes_per_3dele_, 0.0);

  double jac_x = fe_x.GetInvjacobianOnEle()[ele_id_x] *
                 fe_y.GetJacobianOnEle()[ele_id_y] *
                 fe_z.GetJacobianOnEle()[ele_id_z];
  double jac_y = fe_x.GetJacobianOnEle()[ele_id_x] *
                 fe_y.GetInvjacobianOnEle()[ele_id_y] *
                 fe_z.GetJacobianOnEle()[ele_id_z];
  double jac_z = fe_x.GetJacobianOnEle()[ele_id_x] *
                 fe_y.GetJacobianOnEle()[ele_id_y] *
                 fe_z.GetInvjacobianOnEle()[ele_id_z];

  const auto &comp_x = elemental_dNIdNJ_comp_[0];
  const auto &comp_y = elemental_dNIdNJ_comp_[1];
  const auto &comp_z = elemental_dNIdNJ_comp_[2];

  for (int i = 0; i < result.size(); ++i)
  {
    result[i] = jac_x * comp_x[i] + jac_y * comp_y[i] + jac_z * comp_z[i];
  }

  return result;
}
unsigned int
Tucker2EI::FE3D::GetNumNodesPer3Dele() const
{
  return num_nodes_per_3dele_;
}
unsigned int
Tucker2EI::FE3D::GetNumQuadPointsPer3Dele() const
{
  return num_quad_points_per_3dele_;
}
unsigned int
Tucker2EI::FE3D::GetNum3DEle() const
{
  return num_3d_ele_;
}
const std::vector<double> &
Tucker2EI::FE3D::GetElementalShapeFunctionAtQuadPoints() const
{
  return elemental_shape_function_at_quad_points_;
}
void
Tucker2EI::FE3D::Convert3DMeshToCellwiseVector(const Tucker::Tensor *tensor,
                                               double *cellwise_vector) const
{
  int num_nodes_per_cell_x = fe_x.GetNumberNodesPerElement();
  int num_nodes_per_cell_y = fe_y.GetNumberNodesPerElement();
  int num_nodes_per_cell_z = fe_z.GetNumberNodesPerElement();

  int num_nodes_x = fe_x.GetNumberNodes();
  int num_nodes_y = fe_y.GetNumberNodes();
  int num_nodes_z = fe_z.GetNumberNodes();

  int num_ele_x = fe_x.GetNumberElements();
  int num_ele_y = fe_y.GetNumberElements();
  int num_ele_z = fe_z.GetNumberElements();

  //std::vector<double> result(num_nodes_per_3dele_ * num_3d_ele_, 0.0);
  const double       *tensor_data = tensor->data();
  for (int ele_k = 0, cnt = 0; ele_k < num_ele_z; ++ele_k)
  {
    for (int ele_j = 0; ele_j < num_ele_y; ++ele_j)
    {
      for (int ele_i = 0; ele_i < num_ele_x; ++ele_i)
      {
        for (int node_k = 0; node_k < num_nodes_per_cell_z; ++node_k)
        {
          for (int node_j = 0; node_j < num_nodes_per_cell_y; ++node_j)
          {
            for (int node_i = 0; node_i < num_nodes_per_cell_x; ++node_i)
            {
              int ii  = ele_i * (num_nodes_per_cell_x - 1) + node_i;
              int jj  = ele_j * (num_nodes_per_cell_y - 1) + node_j;
              int kk  = ele_k * (num_nodes_per_cell_z - 1) + node_k;
              int idx = ii + jj * num_nodes_x + kk * num_nodes_x * num_nodes_y;
              cellwise_vector[cnt] = tensor_data[idx];
              cnt++;
            }
          }
        }
      }
    }
  }
  //return result;
}

std::vector<double>
Tucker2EI::FE3D::Convert3DMeshToCellwiseVector(const Tucker::Tensor *tensor) const {
  std::vector<double> result(num_nodes_per_3dele_ * num_3d_ele_, 0.0);
  Convert3DMeshToCellwiseVector(tensor, result.data());
  return result;
}