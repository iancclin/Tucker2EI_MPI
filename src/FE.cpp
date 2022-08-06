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

#include "FE.h"
#include "DataReader.h"
#include "BlasWrapper.h"
#include "ExceptionHandler.h"
#include <numeric>
#include <utility>

Tucker2EI::FE::FE(const std::string &file_name,
                  unsigned int       number_elements,
                  QuadPoints         quad_points)
  : quad_rule_(quad_points)
  , number_elements_(number_elements)
{
  Tucker2EI::utils::VecReader(file_name, nodal_coord_);
  InitializeHelper(nodal_coord_, number_elements, quad_points);
}

Tucker2EI::FE::FE(std::vector<double> coord,
                  unsigned int        number_elements,
                  QuadPoints          quad_points)
  : quad_rule_(quad_points)
  , number_elements_(number_elements)
  , nodal_coord_(std::move(coord))
{
  InitializeHelper(nodal_coord_, number_elements_, quad_points);
}

void
Tucker2EI::FE::ComputeValueNodal2Quad(const std::vector<double> &nodal_val,
                                      std::vector<double>       &quad_val) const
{
  TUCKER2EI_ASSERTWITHSTRING(nodal_val.size() == number_nodes_,
                             "nodal size not consistent");
  TUCKER2EI_ASSERTWITHSTRING(quad_val.size() == number_quad_points_,
                             "quad size not consistent");
  ComputeValueNodal2Quad(nodal_val.data(), quad_val.data());
}

void
Tucker2EI::FE::ComputeValueNodal2Quad(const double *nodal_val,
                                      double       *quad_val) const
{
  for (int i = 0; i < number_elements_; ++i)
  {
    blas_wrapper::Dgemvn(number_quad_points_per_element_,
                         number_nodes_per_element_,
                         elemental_shape_function_at_quad_points_.data(),
                         nodal_val + i * (number_nodes_per_element_ - 1),
                         quad_val + i * number_quad_points_per_element_);
  }
}

double
Tucker2EI::FE::IntegrateWithNodalValues(
  const std::vector<double> &nodal_val) const
{
  TUCKER2EI_ASSERTWITHSTRING(nodal_val.size() == number_nodes_,
                             "size not consistent");

  return IntegrateWithNodalValues(nodal_val.data());
}

double
Tucker2EI::FE::IntegrateWithNodalValues(const double *nodal_val) const
{
  std::vector<double> quad_val(number_quad_points_, 0.0);
  ComputeValueNodal2Quad(nodal_val, quad_val.data());

  return IntegrateWithQuadValues(quad_val);
}

double
Tucker2EI::FE::IntegrateWithQuadValues(
  const std::vector<double> &quad_val) const
{
  TUCKER2EI_ASSERTWITHSTRING(quad_val.size() == number_quad_points_,
                             "size not consistent");

  return IntegrateWithQuadValues(quad_val.data());
}

double
Tucker2EI::FE::IntegrateWithQuadValues(const double *quad_val) const
{
  double result = 0.0;
  for (int i = 0; i < number_quad_points_; ++i)
  {
    result += quad_val[i] * jacob_times_weight_quad_values_[i];
  }
  return result;
}


void
Tucker2EI::FE::InitializeHelper(const std::vector<double> &coord,
                                unsigned int               number_elements,
                                Tucker2EI::QuadPoints      quad_points)
{
  number_nodes_                   = nodal_coord_.size();
  domain_start_                   = nodal_coord_.front();
  domain_end_                     = nodal_coord_.back();
  number_nodes_per_element_       = (number_nodes_ - 1) / number_elements + 1;
  number_quad_points_per_element_ = quad_points;
  number_quad_points_ = number_quad_points_per_element_ * number_elements_;

  // compute isoparametric coordinates for an element in [-1, 1]
  elemental_nodal_coord_.assign(number_nodes_per_element_, -1);
  double element_size =
    nodal_coord_[number_nodes_per_element_ - 1] - nodal_coord_[0];
  for (int i = 1; i < number_nodes_per_element_; ++i)
  {
    elemental_nodal_coord_[i] =
      elemental_nodal_coord_[i - 1] +
      2.0 * (nodal_coord_[i] - nodal_coord_[i - 1]) / element_size;
  }

  elemental_shape_function_at_quad_points_.assign(
    number_nodes_per_element_ * number_quad_points_per_element_, 0.0);
  ComputeElementalShapeFunctionAtQuadPoints(
    elemental_shape_function_at_quad_points_);

  quad_coord_.assign(number_quad_points_, 0.0);
  ComputeValueNodal2Quad(nodal_coord_, quad_coord_);

  jacob_times_weight_quad_values_.assign(number_quad_points_, 0.0);
  jacobian_on_ele_.assign(number_elements_, 0.0);
  ComputeJacobTimesWeightQuadValues(jacob_times_weight_quad_values_,
                                    jacobian_on_ele_);

  invjacob_times_weight_quad_values_.assign(number_quad_points_, 0.0);
  invjacobian_on_ele_.assign(number_elements_, 0.0);
  ComputeInvjacobTimesWeightQuadValues(invjacob_times_weight_quad_values_,
                                       invjacobian_on_ele_);

  elemental_Ni_Nj_.assign(number_nodes_per_element_,
                          std::vector<double>(number_nodes_per_element_, 0.0));
  ComputeElementalNINJ(elemental_Ni_Nj_);

  elemental_dNi_dNj_.assign(number_nodes_per_element_,
                            std::vector<double>(number_nodes_per_element_,
                                                0.0));
  ComputeElementaldNIdNJ(elemental_dNi_dNj_);
}

void
Tucker2EI::FE::ComputeElementalShapeFunctionAtQuadPoints(
  std::vector<double> &elemental_shape_function_at_quad_points) const
{
  TUCKER2EI_ASSERTWITHSTRING(elemental_shape_function_at_quad_points.size() ==
                               (number_nodes_per_element_ *
                                number_quad_points_per_element_),
                             "size not consistent");

  const std::vector<double> &elemental_nodal_coord = elemental_nodal_coord_;
  const std::vector<double> &elemental_quad_points = quad_rule_.GetQuadPoints();

  // computing shape function values using Legendre polynomials
  // i_node: col, i_quad: row
  for (int i_node = 0; i_node < number_nodes_per_element_; ++i_node)
  {
    for (int i_quad = 0; i_quad < number_quad_points_per_element_; ++i_quad)
    {
      double val = 1.0;
      for (int i_prod = 0; i_prod < number_nodes_per_element_; ++i_prod)
      {
        // compute \prod_{i_{prod} \neq i_{node}} -\frac{x_{i_{quad}} -
        // x_{i_{prod}}}{x_{i_{node}} - x_{i_{prod}}}
        if (i_prod != i_node)
        {
          val *=
            ((elemental_quad_points[i_quad] - elemental_nodal_coord[i_prod]) /
             (elemental_nodal_coord[i_node] - elemental_nodal_coord[i_prod]));
        }
      }

      elemental_shape_function_at_quad_points
        [i_quad + i_node * number_quad_points_per_element_] = val;
    }
  }
}

void
Tucker2EI::FE::ComputeElementalShapeFunctionDerivativeAtQuadPoints(
  std::vector<double> &elemental_shape_function_derivative_at_quad_points) const
{
  TUCKER2EI_ASSERTWITHSTRING(
    elemental_shape_function_derivative_at_quad_points.size() ==
      (number_nodes_per_element_ * number_quad_points_per_element_),
    "size not consistent");

  const std::vector<double> &elemental_nodal_coord = elemental_nodal_coord_;
  const std::vector<double> &elemental_quad_points = quad_rule_.GetQuadPoints();

  // computing derivative of shape function values
  // i_node: col, i_quad: row
  for (int i_node = 0; i_node < number_nodes_per_element_; ++i_node)
  {
    for (int i_quad = 0; i_quad < number_quad_points_per_element_; ++i_quad)
    {
      double val = 0.0;
      for (int i_sum = 0; i_sum < number_nodes_per_element_; ++i_sum)
      {
        if (i_sum != i_node)
        {
          // derivative = \frac{1}{x_{i_{node}} - x_{i_{sum}}}
          double derivative = 1.0 / (elemental_nodal_coord[i_node] -
                                     elemental_nodal_coord[i_sum]);

          // compute \frac{1}{x_{i_{node}} - x_{i_{sum}}}
          // \prod_{i_{prod} \neq i_{sum} \neq i_{node}}
          // -\frac{x_{i_{quad}} - x_{i_{prod}}}{x_{i_{node}} -
          // x_{i_{prod}}}
          for (int i_prod = 0; i_prod < number_nodes_per_element_; ++i_prod)
          {
            if (i_prod != i_node && i_prod != i_sum)
            {
              derivative *= ((elemental_quad_points[i_quad] -
                              elemental_nodal_coord[i_prod]) /
                             (elemental_nodal_coord[i_node] -
                              elemental_nodal_coord[i_prod]));
            }
          }
          elemental_shape_function_derivative_at_quad_points
            [i_quad + i_node * number_quad_points_per_element_] += derivative;
        }
      }
    }
  }
}

void
Tucker2EI::FE::ComputeJacobTimesWeightQuadValues(
  std::vector<double> &jacob_times_weight_quad_values,
  std::vector<double> &jacobian_on_ele) const
{
  TUCKER2EI_ASSERTWITHSTRING(jacob_times_weight_quad_values.size() ==
                               number_quad_points_,
                             "quad size not consistent");
  TUCKER2EI_ASSERTWITHSTRING(jacobian_on_ele.size() == number_elements_,
                             "element size not consistent");

  const std::vector<double> &weight = quad_rule_.GetQuadWeights();

  for (int i_ele = 0, i_jacob = 0; i_ele < number_elements_; ++i_ele)
  {
    int i_ele_start = i_ele * (number_nodes_per_element_ - 1);
    int i_ele_end   = (i_ele + 1) * (number_nodes_per_element_ - 1);

    double jacobian =
      0.5 * (nodal_coord_[i_ele_end] - nodal_coord_[i_ele_start]);
    jacobian_on_ele[i_ele] = jacobian;
    for (int i_quad = 0; i_quad < number_quad_points_per_element_;
         ++i_quad, ++i_jacob)
    {
      jacob_times_weight_quad_values[i_jacob] = jacobian * weight[i_quad];
    }
  }
}

void
Tucker2EI::FE::ComputeInvjacobTimesWeightQuadValues(
  std::vector<double> &invjacob_times_weight_quad_values,
  std::vector<double> &invjacobian_on_ele) const
{
  TUCKER2EI_ASSERTWITHSTRING(invjacob_times_weight_quad_values.size() ==
                               number_quad_points_,
                             "size not consistent");
  TUCKER2EI_ASSERTWITHSTRING(invjacobian_on_ele.size() == number_elements_,
                             "element size not consistent");

  const std::vector<double> &weight = quad_rule_.GetQuadWeights();

  for (int i_ele = 0, i_jacob = 0; i_ele < number_elements_; ++i_ele)
  {
    int i_ele_start = i_ele * (number_nodes_per_element_ - 1);
    int i_ele_end   = (i_ele + 1) * (number_nodes_per_element_ - 1);

    double inv_jacobian =
      2.0 / (nodal_coord_[i_ele_end] - nodal_coord_[i_ele_start]);
    invjacobian_on_ele[i_ele] = inv_jacobian;
    for (int i_quad = 0; i_quad < number_quad_points_per_element_;
         ++i_quad, ++i_jacob)
    {
      invjacob_times_weight_quad_values[i_jacob] =
        inv_jacobian * weight[i_quad];
    }
  }
}

void
Tucker2EI::FE::ComputeElementalNINJ(DoubleDVec &elemental_Ni_Nj) const
{
  const auto &weight = quad_rule_.GetQuadWeights();
  for (int i = 0; i < number_nodes_per_element_; ++i)
  {
    const double *N_i = &elemental_shape_function_at_quad_points_
                          [i * number_quad_points_per_element_];
    for (int j = 0; j < number_nodes_per_element_; ++j)
    {
      const double *N_j = &elemental_shape_function_at_quad_points_
                            [j * number_quad_points_per_element_];

      double int_Ni_Nj = 0.0;
      for (int i_quad = 0; i_quad < number_quad_points_per_element_; ++i_quad)
      {
        int_Ni_Nj += weight[i_quad] * N_i[i_quad] * N_j[i_quad];
      }

      elemental_Ni_Nj[i][j] = int_Ni_Nj;
    }
  }
}

void
Tucker2EI::FE::ComputeElementaldNIdNJ(DoubleDVec &elemental_dNi_dNj) const
{
  std::vector<double> elemental_dNi_quad(number_nodes_per_element_ *
                                           number_quad_points_per_element_,
                                         0.0);
  ComputeElementalShapeFunctionDerivativeAtQuadPoints(elemental_dNi_quad);

  const auto &weight = quad_rule_.GetQuadWeights();

  for (int i = 0; i < number_nodes_per_element_; ++i)
  {
    const double *dN_i =
      &elemental_dNi_quad[i * number_quad_points_per_element_];
    for (int j = 0; j < number_nodes_per_element_; ++j)
    {
      const double *dN_j =
        &elemental_dNi_quad[j * number_quad_points_per_element_];

      double int_dNi_dNj = 0.0;
      for (int i_quad = 0; i_quad < number_quad_points_per_element_; ++i_quad)
      {
        int_dNi_dNj += weight[i_quad] * dN_i[i_quad] * dN_j[i_quad];
      }
      elemental_dNi_dNj[i][j] = int_dNi_dNj;
    }
  }
}
unsigned int
Tucker2EI::FE::GetNumberElements() const
{
  return number_elements_;
}
unsigned int
Tucker2EI::FE::GetNumberNodes() const
{
  return number_nodes_;
}
unsigned int
Tucker2EI::FE::GetNumberNodesPerElement() const
{
  return number_nodes_per_element_;
}
unsigned int
Tucker2EI::FE::GetNumberQuadPoints() const
{
  return number_quad_points_;
}
unsigned int
Tucker2EI::FE::GetNumberQuadPointsPerElement() const
{
  return number_quad_points_per_element_;
}
double
Tucker2EI::FE::GetDomainStart() const
{
  return domain_start_;
}
double
Tucker2EI::FE::GetDomainEnd() const
{
  return domain_end_;
}
const std::vector<double> &
Tucker2EI::FE::GetNodalCoord() const
{
  return nodal_coord_;
}
// const std::vector<double> &
// Tucker2EI::FE::GetElementalNodalCoord() const
//{
//   return elemental_nodal_coord_;
// }
const std::vector<double> &
Tucker2EI::FE::GetElementalShapeFunctionAtQuadPoints() const
{
  return elemental_shape_function_at_quad_points_;
}
const std::vector<double> &
Tucker2EI::FE::GetJacobTimesWeightQuadValues() const
{
  return jacob_times_weight_quad_values_;
}
const std::vector<double> &
Tucker2EI::FE::GetInvjacobTimesWeightQuadValues() const
{
  return invjacob_times_weight_quad_values_;
}
const std::vector<double> &
Tucker2EI::FE::GetQuadCoord() const
{
  return quad_coord_;
}
const std::vector<double> &
Tucker2EI::FE::GetQuadWeightOnOneEle() const
{
  return quad_rule_.GetQuadWeights();
}
const Tucker2EI::FE::DoubleDVec &
Tucker2EI::FE::GetElementalNiNj() const
{
  return elemental_Ni_Nj_;
}
const Tucker2EI::FE::DoubleDVec &
Tucker2EI::FE::GetElementalDNiDNj() const
{
  return elemental_dNi_dNj_;
}
const std::vector<double> &
Tucker2EI::FE::GetJacobianOnEle() const
{
  return jacobian_on_ele_;
}
const std::vector<double> &
Tucker2EI::FE::GetInvjacobianOnEle() const
{
  return invjacobian_on_ele_;
}
