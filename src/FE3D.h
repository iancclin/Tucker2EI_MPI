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

#ifndef TUCKER2EI_SRC_FE3D_H_
#define TUCKER2EI_SRC_FE3D_H_

#include "FE.h"
#include <string>
#include <Tucker_Tensor.hpp>

namespace Tucker2EI
{
  class FE3D
  {
  public:
    FE3D(const std::string &file_name_x,
         const std::string &file_name_y,
         const std::string &file_name_z,
         unsigned int       num_ele_x,
         unsigned int       num_ele_y,
         unsigned int       num_ele_z,
         QuadPoints         quad_points_x,
         QuadPoints         quad_points_y,
         QuadPoints         quad_points_z);

    void
    InterpolateNode2Quad(const Tucker::Tensor *nodal_val,
                         Tucker::Tensor       *quad_val) const;

    double
    Compute3DIntegralFromNode(const Tucker::Tensor *nodal_val) const;

    double
    Compute3DIntegralFromQuad(const Tucker::Tensor *quad_val) const;

    /**
     * @brief Extract nodal values of an element as a vector, this function
     * does not check bound, use with care.
     * @param ele_id_x element id in x-direction
     * @param ele_id_y element id in y-direction
     * @param ele_id_z element id in z-direction
     * @param nodal_tensor a pointer to the full tensor at node
     * @return nodal value of the element
     */
    std::vector<double>
    ExtractElementalVector(int           ele_id_x,
                           int           ele_id_y,
                           int           ele_id_z,
                           const double *nodal_tensor) const;

    /**
     * @brief Write nodal values from a vector to their correspond place in
     * the tensor, this function does not check bound, use with care.
     * @param ele_id_x element id in x-direction
     * @param ele_id_y element id in y-direction
     * @param ele_id_z element id in z-direction
     * @param val vector storing nodal values
     * @param nodal_tensor the pointer to the tensor
     */
    void
    WriteElementalVector(int                        ele_id_x,
                         int                        ele_id_y,
                         int                        ele_id_z,
                         const std::vector<double> &val,
                         double                    *nodal_tensor) const;

    std::vector<double>
    GetElementalDNiDNj(int ele_id_x, int ele_id_y, int ele_id_z) const;

    FE             fe_x, fe_y, fe_z;
    FEPtrContainer fe;

    unsigned int
    GetNumNodesPer3Dele() const;

    unsigned int
    GetNumQuadPointsPer3Dele() const;

    unsigned int
    GetNum3DEle() const;

    const std::vector<double> &
    GetElementalShapeFunctionAtQuadPoints() const;

    void
    Convert3DMeshToCellwiseVector(const Tucker::Tensor *tensor,
                                  double               *cellwise_vector) const;

    std::vector<double>
    Convert3DMeshToCellwiseVector(const Tucker::Tensor *tensor) const;

  private:
    unsigned int        num_nodes_per_3dele_;
    unsigned int        num_quad_points_per_3dele_;
    unsigned int        num_3d_ele_;
    std::vector<double> elemental_shape_function_at_quad_points_;
    // std::vector<std::vector<double>> elemental_dNI_dNJ_;
    std::array<std::vector<double>, 3> elemental_dNIdNJ_comp_;
    std::array<std::vector<double>, 3> jacob_comp_;

    void
    ComputeShapeFunctionAtQuadPoints(
      std::vector<double> &elemental_shape_function_at_quad_points) const;

    //    void
    //    ComputedNIdNJ(std::vector<std::vector<double>> &elemental_dNI_dNJ)
    //    const;
    void
    ComputedNIdNJComponents(
      std::array<std::vector<double>, 3> &elemental_dNIdNJ_comp,
      std::array<std::vector<double>, 3> &jacob_comp) const;

    void
    ExtractElementalVector(int                  ele_id_x,
                           int                  ele_id_y,
                           int                  ele_id_z,
                           int                  ele_num_nodes_x,
                           int                  ele_num_nodes_y,
                           int                  ele_num_nodes_z,
                           const double        *nodal_tensor,
                           int                  tensor_size_x,
                           int                  tensor_size_y,
                           int                  tensor_size_z,
                           std::vector<double> &elemental_vector) const;

    void
    WriteElementalVector(int                        ele_id_x,
                         int                        ele_id_y,
                         int                        ele_id_z,
                         int                        ele_num_nodes_x,
                         int                        ele_num_nodes_y,
                         int                        ele_num_nodes_z,
                         const std::vector<double> &elemental_vector,
                         int                        tensor_size_x,
                         int                        tensor_size_y,
                         int                        tensor_size_z,
                         double                    *nodal_tensor) const;

    void
    WriteElementalQuadVector(int                        ele_id_x,
                             int                        ele_id_y,
                             int                        ele_id_z,
                             int                        ele_num_quads_x,
                             int                        ele_num_quads_y,
                             int                        ele_num_quads_z,
                             const std::vector<double> &quad_vector,
                             int                        tensor_size_x,
                             int                        tensor_size_y,
                             int                        tensor_size_z,
                             double                    *quad_tensor) const;
  };
} // namespace Tucker2EI

#endif // TUCKER2EI_SRC_FE3D_H_
