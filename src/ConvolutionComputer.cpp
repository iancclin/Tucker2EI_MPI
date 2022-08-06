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

#include <algorithm>
#include <string>
#include <cmath>
#include <array>
#include "BlasWrapper.h"
#include "DataReader.h"
#include "ConvolutionComputer.h"
#include "ExceptionHandler.h"

namespace
{
  // fixme: this function should not be put here, move this to Tensor
  //  wrappers later
  void
  ReconstructTensor(Tucker::Tensor  *core,
                    Tucker::Matrix  *factor_mat_x,
                    Tucker::Matrix  *factor_mat_y,
                    Tucker::Matrix  *factor_mat_z,
                    Tucker::Tensor *&tensor);
} // namespace

Tucker2EI::ConvolutionComputer::ConvolutionComputer()
  : is_empty_(true)
  , num_expansion_terms_(0)
  , a_square_(0.0)
{}

bool
Tucker2EI::ConvolutionComputer::IsEmpty() const
{
  return is_empty_;
}

Tucker2EI::ConvolutionComputer::ConvolutionComputer(
  const std::string &omega_filename,
  const std::string &alpha_filename,
  const double       a_square)
  : is_empty_(false)
  , num_expansion_terms_(0)
  , a_square_(a_square)
{
  omega_.clear();
  alpha_.clear();
  Tucker2EI::utils::VecReader(omega_filename, omega_);
  Tucker2EI::utils::VecReader(alpha_filename, alpha_);

  TUCKER2EI_ASSERTWITHSTRING(omega_.size() == alpha_.size(),
                             "omega and alpha should be in  the same size");

  num_expansion_terms_ = alpha_.size();

  double inv_a_square = 1.0 / a_square_;
  std::for_each(alpha_.begin(), alpha_.end(), [inv_a_square](double &v) {
    v *= inv_a_square;
  });

  double inv_sqrt_a_square = 1.0 / std::sqrt(a_square_);
  std::for_each(omega_.begin(), omega_.end(), [inv_sqrt_a_square](double &v) {
    v *= inv_sqrt_a_square;
  });
}

void
Tucker2EI::ConvolutionComputer::SetKernelCoeffFiles(
  const std::string &omega_filename,
  const std::string &alpha_filename,
  const double       a_square)
{
  a_square_ = a_square;
  omega_.clear();
  alpha_.clear();
  Tucker2EI::utils::VecReader(omega_filename, omega_);
  Tucker2EI::utils::VecReader(alpha_filename, alpha_);

  TUCKER2EI_ASSERTWITHSTRING(omega_.size() == alpha_.size(),
                             "omega and alpha should be in  the same size");

  num_expansion_terms_ = alpha_.size();

  double inv_a_square = 1.0 / a_square;
  std::for_each(alpha_.begin(), alpha_.end(), [inv_a_square](double &v) {
    v *= inv_a_square;
  });

  double inv_sqrt_a_square = 1.0 / std::sqrt(a_square);
  std::for_each(omega_.begin(), omega_.end(), [inv_sqrt_a_square](double &v) {
    v *= inv_sqrt_a_square;
  });
}
using namespace std::chrono;
void
Tucker2EI::ConvolutionComputer::ComputeQuadConvolutionFromNode(
  FEPtrContainer  fe,
  FEPtrContainer  fe_conv,
  double          tol,
  Tucker::Tensor *input_nodal_field,
  Tucker::Tensor *convolution_quad_field) const
{
  auto                        start = high_resolution_clock::now();
  const Tucker::TuckerTensor *input_nodal_field_tt =
    Tucker::STHOSVD(input_nodal_field, tol);
  auto end = high_resolution_clock::now();
  printf("conv::decomp: %.4f\n",
         double(duration_cast<milliseconds>(end - start).count()) / 1.0e3);


  int tucker_rank[3];
  for (int dim = 0; dim < 3; ++dim)
  {
    tucker_rank[dim] = input_nodal_field_tt->G->size(dim);
  }

  int n_fe_quad_points[3];
  for (int dim = 0; dim < 3; ++dim)
  {
    n_fe_quad_points[dim] = fe[dim]->GetNumberQuadPoints();
  }

  std::vector<std::vector<Tucker::Matrix *>> conv_mat_1d(
    3, std::vector<Tucker::Matrix *>(num_expansion_terms_));

  start = high_resolution_clock::now();
  std::vector<std::vector<double>> conv_mat_1d_temp(3);

  for (int dim = 0; dim < 3; ++dim)
  {
    for (int i_term = 0; i_term < num_expansion_terms_; ++i_term)
    {
      conv_mat_1d[dim][i_term] =
        Tucker::MemoryManager::safe_new<Tucker::Matrix>(n_fe_quad_points[dim],
                                                        tucker_rank[dim]);
      conv_mat_1d[dim][i_term]->initialize();
    }
    Compute1DConv(tucker_rank[dim],
                  *fe[dim],
                  *fe_conv[dim],
                  input_nodal_field_tt->U[dim],
                  conv_mat_1d[dim]);
  }
  end = high_resolution_clock::now();
  printf("conv::1d conv int: %.4f\n",
         double(duration_cast<milliseconds>(end - start).count()) / 1.0e3);


  // Reconstruct and reduce tensors
  start = high_resolution_clock::now();
  for (int i_term = 0; i_term < num_expansion_terms_; ++i_term)
  {
    Tucker::Tensor *tensor;
    ReconstructTensor(input_nodal_field_tt->G,
                      conv_mat_1d[0][i_term],
                      conv_mat_1d[1][i_term],
                      conv_mat_1d[2][i_term],
                      tensor);
    blas_wrapper::Daxpy(tensor->getNumElements(),
                        omega_[i_term],
                        tensor->data(),
                        convolution_quad_field->data());
    Tucker::MemoryManager::safe_delete(tensor);
  }
  end = high_resolution_clock::now();
  printf("conv::recon: %.4f\n",
         double(duration_cast<milliseconds>(end - start).count()) / 1.0e3);


  // Release resources
  for (int dim = 0; dim < 3; ++dim)
  {
    for (int i_term = 0; i_term < num_expansion_terms_; ++i_term)
    {
      Tucker::MemoryManager::safe_delete(conv_mat_1d[dim][i_term]);
    }
  }
  Tucker::MemoryManager::safe_delete(input_nodal_field_tt);
}

void
Tucker2EI::ConvolutionComputer::ComputeQuadConvolutionFromNodeOnNodeInplace(
  FEPtrContainer  fe,
  FEPtrContainer  fe_conv,
  double          tol,
  Tucker::Tensor *input_nodal_field) const
{
  auto                        start = high_resolution_clock::now();
  const Tucker::TuckerTensor *input_nodal_field_tt =
    Tucker::STHOSVD(input_nodal_field, tol);
  auto end = high_resolution_clock::now();
  printf("nconv::decomp: %.4f\n",
         double(duration_cast<milliseconds>(end - start).count()) / 1.0e3);


  int tucker_rank[3];
  for (int dim = 0; dim < 3; ++dim)
  {
    tucker_rank[dim] = input_nodal_field_tt->G->size(dim);
  }

  int n_fe_node_points[3];
  for (int dim = 0; dim < 3; ++dim)
  {
    n_fe_node_points[dim] = fe[dim]->GetNumberNodes();
  }

  std::vector<std::vector<Tucker::Matrix *>> conv_mat_1d(
    3, std::vector<Tucker::Matrix *>(num_expansion_terms_));

  start = high_resolution_clock::now();
  std::vector<std::vector<double>> conv_mat_1d_temp(3);

  for (int dim = 0; dim < 3; ++dim)
  {
    for (int i_term = 0; i_term < num_expansion_terms_; ++i_term)
    {
      conv_mat_1d[dim][i_term] =
        Tucker::MemoryManager::safe_new<Tucker::Matrix>(n_fe_node_points[dim],
                                                        tucker_rank[dim]);
      conv_mat_1d[dim][i_term]->initialize();
    }
    Compute1DConvOnNode(tucker_rank[dim],
                        *fe[dim],
                        *fe_conv[dim],
                        input_nodal_field_tt->U[dim],
                        conv_mat_1d[dim]);
  }
  end = high_resolution_clock::now();
  printf("conv::1d conv int: %.4f\n",
         double(duration_cast<milliseconds>(end - start).count()) / 1.0e3);

  // Reconstruct and reduce tensors
  start = high_resolution_clock::now();
  Tucker::SizeArray fe_node_sz(3);
  for (int dim = 0; dim < 3; ++dim)
  {
    fe_node_sz[dim] = fe[dim]->GetNumberNodes();
  }

  input_nodal_field->initialize();
  for (int i_term = 0; i_term < num_expansion_terms_; ++i_term)
  {
    Tucker::Tensor *tensor;
    ReconstructTensor(input_nodal_field_tt->G,
                      conv_mat_1d[0][i_term],
                      conv_mat_1d[1][i_term],
                      conv_mat_1d[2][i_term],
                      tensor);
    blas_wrapper::Daxpy(tensor->getNumElements(),
                        omega_[i_term],
                        tensor->data(),
                        input_nodal_field->data());
    Tucker::MemoryManager::safe_delete(tensor);
  }
  end = high_resolution_clock::now();
  printf("conv::recon: %.4f\n",
         double(duration_cast<milliseconds>(end - start).count()) / 1.0e3);


  // Release resources
  for (int dim = 0; dim < 3; ++dim)
  {
    for (int i_term = 0; i_term < num_expansion_terms_; ++i_term)
    {
      Tucker::MemoryManager::safe_delete(conv_mat_1d[dim][i_term]);
    }
  }
  Tucker::MemoryManager::safe_delete(input_nodal_field_tt);
}


void
Tucker2EI::ConvolutionComputer::Clear()
{
  is_empty_            = true;
  num_expansion_terms_ = 0;
  std::vector<double>().swap(omega_);
  std::vector<double>().swap(alpha_);
  a_square_ = 0.0;
}
void

Tucker2EI::ConvolutionComputer::Compute1DConv(
  const int                      rank,
  const Tucker2EI::FE           &fe,
  const Tucker2EI::FE           &fe_conv,
  const Tucker::Matrix          *mat_u_nodal,
  std::vector<Tucker::Matrix *> &conv_mat_1d) const
{
  const double *mat_u_nodal_data = mat_u_nodal->data();

  int n_fe_quad_points      = fe.GetNumberQuadPoints();
  int n_fe_nodes            = fe.GetNumberNodes();
  int n_fe_conv_quad_points = fe_conv.GetNumberQuadPoints();

  const std::vector<double> &quad_point_position      = fe.GetQuadCoord();
  const std::vector<double> &conv_quad_point_position = fe_conv.GetQuadCoord();

  for (int i_term = 0; i_term != num_expansion_terms_; ++i_term)
  {
    double *conv_mat_1d_data = conv_mat_1d[i_term]->data();
    double  alpha            = alpha_[i_term];

    for (auto i_rank = 0, cnt = 0; i_rank < rank; ++i_rank)
    {
      std::vector<double> mat_u_irank_conv_quad(n_fe_conv_quad_points, 0.0);
      const double *mat_u_irank_node = mat_u_nodal_data + i_rank * n_fe_nodes;
      fe_conv.ComputeValueNodal2Quad(mat_u_irank_node,
                                     mat_u_irank_conv_quad.data());

      for (auto i_quad = 0; i_quad < n_fe_quad_points; ++i_quad)
      {
        std::vector<double> temp(n_fe_conv_quad_points, 0.0);

        double quad_coord_i = quad_point_position[i_quad];

        for (auto i_conv_quad = 0; i_conv_quad < n_fe_conv_quad_points;
             ++i_conv_quad)
        {
          double r = quad_coord_i - conv_quad_point_position[i_conv_quad];
          temp[i_conv_quad] =
            std::exp(-alpha * r * r) * mat_u_irank_conv_quad[i_conv_quad];
        }
        conv_mat_1d_data[cnt++] = fe_conv.IntegrateWithQuadValues(temp);
      }
    }
  }
}

void

Tucker2EI::ConvolutionComputer::Compute1DConvOnNode(
  const int                      rank,
  const Tucker2EI::FE           &fe,
  const Tucker2EI::FE           &fe_conv,
  const Tucker::Matrix          *mat_u_nodal,
  std::vector<Tucker::Matrix *> &conv_mat_1d) const
{
  const double *mat_u_nodal_data = mat_u_nodal->data();

  //  int n_fe_quad_points      = fe.GetNumberQuadPoints();
  int n_fe_nodes            = fe.GetNumberNodes();
  int n_fe_conv_quad_points = fe_conv.GetNumberQuadPoints();

  const std::vector<double> &nodes_position           = fe.GetNodalCoord();
  const std::vector<double> &conv_quad_point_position = fe_conv.GetQuadCoord();

  for (int i_term = 0; i_term != num_expansion_terms_; ++i_term)
  {
    double *conv_mat_1d_data = conv_mat_1d[i_term]->data();
    double  alpha            = alpha_[i_term];

    for (auto i_rank = 0, cnt = 0; i_rank < rank; ++i_rank)
    {
      std::vector<double> mat_u_irank_conv_quad(n_fe_conv_quad_points, 0.0);
      const double *mat_u_irank_node = mat_u_nodal_data + i_rank * n_fe_nodes;
      fe_conv.ComputeValueNodal2Quad(mat_u_irank_node,
                                     mat_u_irank_conv_quad.data());

      for (auto i_node = 0; i_node < n_fe_nodes; ++i_node)
      {
        std::vector<double> temp(n_fe_conv_quad_points, 0.0);

        double node_coord_i = nodes_position[i_node];

        for (auto i_conv_quad = 0; i_conv_quad < n_fe_conv_quad_points;
             ++i_conv_quad)
        {
          double r = node_coord_i - conv_quad_point_position[i_conv_quad];
          temp[i_conv_quad] =
            std::exp(-alpha * r * r) * mat_u_irank_conv_quad[i_conv_quad];
        }
        conv_mat_1d_data[cnt++] = fe_conv.IntegrateWithQuadValues(temp);
      }
    }
  }
}


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