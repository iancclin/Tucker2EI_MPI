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

#ifndef TUCKER2EI__CONVOLUTIONCOMPUTER_H_
#define TUCKER2EI__CONVOLUTIONCOMPUTER_H_

#include <vector>
#include <Tucker.hpp>
#include "FE.h"

namespace Tucker2EI
{
  class ConvolutionComputer
  {
  public:
    /**
     * @brief Default constructor, dose nothing
     */
    ConvolutionComputer();

    /**
     * @brief Constructor
     * @param omega_filename the file name of alpha constants
     * @param alpha_filename the file name of omega constants
     * @param a_square
     */
    ConvolutionComputer(const std::string &omega_filename,
                        const std::string &alpha_filename,
                        double             a_square);

    /**
     * @brief Reset the object to other omega/alpha files, almost the
     * same as the constructor
     * @param omega_filename the file name of alpha constants
     * @param alpha_filename the file name of omega constants
     * @param a_square
     */
    void
    SetKernelCoeffFiles(const std::string &omega_filename,
                        const std::string &alpha_filename,
                        double             a_square);

    /**
     * @brief Clear the object, so the object can be reset using
     * @b SetKernelCoeffFiles
     * @sa SetKernelCoeffFiles
     */
    void
    Clear();

    /**
     * @brief Compute the convolution integral of the given input field
     * \f$ f(\mathbf{r})*\frac{1}{\mathbf{r}} =
     * \int\frac{f(\mathbf{r'})}{\left|\mathbf{r'} - \mathbf{r}\right|}
     * d\mathbf{r'} \f$ using kernel expansion method
     * @param[in] fe a container storing finite element information in
     * 3 directions (\f$r_1, r_2, r_3\f$)
     * @param[in] fe_conv a container storing finite element information for
     * doing convolution integral in 3 directions (\f$r'_1, r'_2, r'_3\f$)
     * @param[in] input_field \f$ f(\mathbf{r}) \f$ on nodal points
     * @param[out] convolution_field \f$ f(\mathbf{r})*\frac{1}{\mathbf{r}}
     * \f$ on quadrature points
     * @sa FE
     */
    void
    ComputeQuadConvolutionFromNode(
      Tucker2EI::FEPtrContainer fe,
      Tucker2EI::FEPtrContainer fe_conv,
      double                    tol,
      Tucker::Tensor           *input_nodal_field,
      Tucker::Tensor           *convolution_quad_field) const;

    void
    ComputeQuadConvolutionFromNodeOnNodeInplace(
      Tucker2EI::FEPtrContainer fe,
      Tucker2EI::FEPtrContainer fe_conv,
      double                    tol,
      Tucker::Tensor           *input_nodal_field) const;

    /**
     * @brief Check if the object is set.
     * @return If the object is set: false, otherwise: true
     */
    bool
    IsEmpty() const;

  private:
    bool                is_empty_;
    unsigned int        num_expansion_terms_;
    double              a_square_;
    std::vector<double> omega_;
    std::vector<double> alpha_;

    void
    Compute1DConv(int                            rank,
                  const FE                      &fe,
                  const FE                      &fe_conv,
                  const Tucker::Matrix          *mat_u_nodal,
                  std::vector<Tucker::Matrix *> &conv_mat_1d) const;

    void
    Compute1DConvOnNode(int                            rank,
                        const FE                      &fe,
                        const FE                      &fe_conv,
                        const Tucker::Matrix          *mat_u_nodal,
                        std::vector<Tucker::Matrix *> &conv_mat_1d) const;
  };
} // namespace Tucker2EI

#endif // TUCKER2EI__CONVOLUTIONCOMPUTER_H_
