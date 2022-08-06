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

#ifndef TUCKER2EI__FE_H_
#define TUCKER2EI__FE_H_

#include <string>
#include <map>
#include "QuadRule.h"

namespace Tucker2EI
{
  class FE
  {
  public:
    //    typedef std::vector<std::vector<std::vector<double>>> TripleDVec;
    typedef std::vector<std::vector<double>> DoubleDVec;

    /**
     * @brief Default constructor does nothing
     */
    FE() = default;

    /**
     * @brief Default copy constructor
     */
    FE(const FE &fe) = default;

    /**
     * @brief Generate coordination from file
     * @param file_name 1-D coordinates file generated from DFT-FE
     * @param number_elements assign number of elements, hard coded for now
     * @param quad_points number of quadrature points from Tucker2EI::QuadPoints
     */
    FE(const std::string &file_name,
       unsigned int       number_elements,
       QuadPoints         quad_points);

    /**
     * @brief Generate coordination from file
     * @param coord 1-D coordinates of all nodes
     * @param number_elements assign number of elements, hard coded for now
     * @param quad_points number of quadrature points from Tucker2EI::QuadPoints
     */
    FE(std::vector<double> coord,
       unsigned int        number_elements,
       QuadPoints          quad_points);

    /**
     * @brief Compute 1-D projection from nodal points to quadrature points
     * using shape functions
     * @param[in] nodal_val nodal values
     * @param[out] quad_val quad values (the vector will NOT be re-allocated
     * inside)
     */
    void
    ComputeValueNodal2Quad(const std::vector<double> &nodal_val,
                           std::vector<double>       &quad_val) const;

    /**
     * @brief Compute 1-D projection from nodal points to quadrature points
     * using shape functions. This function does not check the boundary of
     * the array. Users have to ensure the size of the array matches
     * @param[in] nodal_val the pointer to the array of nodal values
     * @param[out] quad_val the pointer to the array of quad values (the vector
     * will NOT be re-allocated inside)
     */
    void
    ComputeValueNodal2Quad(const double *nodal_val, double *quad_val) const;

    /**
     * @brief Compute integral of an 1-D function from its nodal values, the
     * nodal values are first projected onto quadrature points then reduce to
     * the resultant integral
     * @param[in] nodal_val the nodal values of the 1-D functions
     * @return integral
     */
    double
    IntegrateWithNodalValues(const std::vector<double> &nodal_val) const;

    /**
     * @brief Compute integral of an 1-D function from its nodal values. the
     * nodal values are first projected onto quadrature points then reduce to
     * the resultant integral. This function does not check the boundary of
     * the array. Users have to ensure the size of the array matches
     *  @b number_quad_points_
     * @param[in] nodal_val the pointer to the array of nodal values of the 1-D
     * functions
     * @return integral
     */
    double
    IntegrateWithNodalValues(const double *nodal_val) const;

    /**
     * @brief Compute integral of an 1-D function from its quadrature values
     * @param[in] quad_val the values of the 1-D functions on quadrature points
     * @return integral
     */
    double
    IntegrateWithQuadValues(const std::vector<double> &quad_val) const;

    /**
     * @brief Compute integral of an 1-D function from its quadrature values.
     * This function does not check the boundary of the array. Users have to
     * ensure the size of the array matches @b number_quad_points_
     * @param[in] quad_val the pointer to the array of values of the 1-D
     * functions on quadrature points
     * @return integral
     */
    double
    IntegrateWithQuadValues(const double *quad_val) const;

    // Getters
    unsigned int
    GetNumberElements() const;

    unsigned int
    GetNumberNodes() const;

    unsigned int
    GetNumberNodesPerElement() const;

    unsigned int
    GetNumberQuadPoints() const;

    unsigned int
    GetNumberQuadPointsPerElement() const;

    /**
     * @brief Getter for @b domain_start_
     * @return the left boundary of the domain
     */
    double
    GetDomainStart() const;

    /**
     * @brief Getter for @b domain_end_
     * @return the right boundary of the domain
     */
    double
    GetDomainEnd() const;

    /**
     * @brief Getter for the @nodal_coord_
     * @return the coordinated of all the nodes in the domain
     */
    const std::vector<double> &
    GetNodalCoord() const;

    /**
     * @brief Getter for the @quad_coord_
     * @return the coordinated of all the quadrature points in the domain
     */
    const std::vector<double> &
    GetQuadCoord() const;

    /**
     * @brief Getter for the weights of the quad points in ONE element
     * @return the weight of the quad points in ONE element
     */
    const std::vector<double> &
    GetQuadWeightOnOneEle() const;

    /**
     * @brief Getter for @b elemental_shape_function_at_quad_points_
     * @return a reference to a matrix (stored in a Fortran style 1-D array) of
     * size (\f$N^{ele}_{quad} \times N^{ele}_{node}\f$) storing \f$
     * \mathbf{N}^{ele}_{i}(x_{quad}) =  \prod_{j \neq i}^{N_{ele}}
     * \frac{x_{quad} - x_{j}}{x_{i} - x_{j}} \f$, the shape function values
     * at quadrature points. (Row: quad points, Col: nodal points)
     */
    const std::vector<double> &
    GetElementalShapeFunctionAtQuadPoints() const;

    /**
     * @brief Getter for @b jacob_times_weight_quad_values_
     * @return a container of Jacobian (the length of one FE element length)
     * times weight on each point in the domain
     */
    const std::vector<double> &
    GetJacobTimesWeightQuadValues() const;

    /**
     * @brief Getter for @b invjacob_times_weight_quad_values_
     * @return a container of inverse of Jacobian (the reciprocal of the
     * length of an FE element length) times weight on each point in the domain
     */
    const std::vector<double> &
    GetInvjacobTimesWeightQuadValues() const;

    /**
     * @brief Getter for @b elemental_Ni_Nj_
     * @return The precomputed integration for \f$ \int_{-1}^{1}N_{i}(\xi)
     * N_{j}(xi)d\xi \f$ in the parametric space \f$ \xi = [-1, 1] \f$.
     */
    const DoubleDVec &
    GetElementalNiNj() const;

    /**
     * @brief Getter for @b elemental_dNi_dNj_
     * @return The precomputed integration for \f$ \int_{-1}^{1}\frac{dN_{i}
     * (\xi)}{d\xi}\frac{dN_{j}(xi)}{d\xi}d\xi \f$ in the parametric space
     * \f$ \xi = [-1, 1] \f$.
     */
    const DoubleDVec &
    GetElementalDNiDNj() const;

    /**
     * @brief Getter for @b jacobian_on_ele_
     * @return The precomputed jacobian value \f$ J^{ele} = \frac{dx}{d\xi} =
     * \frac{l^{ele}}{2} \f$ of each element \f$ ele \f$.
     */
    const std::vector<double> &
    GetJacobianOnEle() const;

    /**
     * @brief Getter for @b invjacobian_on_ele_
     * @return The precomputed inverse jacobian value \f$ (J^{ele})^{-1} =
     * \frac{d\xi}{dx} = \frac{2}{l^{ele}} \f$ of each element \f$ ele \f$.
     */
    const std::vector<double> &
    GetInvjacobianOnEle() const;

  private:
    QuadRule quad_rule_;

    unsigned int number_elements_;
    unsigned int number_nodes_;
    unsigned int number_nodes_per_element_;
    unsigned int number_quad_points_;
    unsigned int number_quad_points_per_element_;

    double domain_start_;
    double domain_end_;

    // global container (size = # of all nodes)
    std::vector<double> nodal_coord_;
    std::vector<double> quad_coord_;
    std::vector<double> jacob_times_weight_quad_values_;
    std::vector<double> invjacob_times_weight_quad_values_;
    std::vector<double> jacobian_on_ele_;
    std::vector<double> invjacobian_on_ele_;

    // elemental container (size = # of points in one element)
    std::vector<double> elemental_nodal_coord_;
    //! @brief for computing numerical integration of a function
    std::vector<double> elemental_shape_function_at_quad_points_;
    DoubleDVec          elemental_Ni_Nj_;
    DoubleDVec          elemental_dNi_dNj_;

    void
    InitializeHelper(const std::vector<double> &coord,
                     unsigned int               number_elements,
                     QuadPoints                 quad_points);

    void
    ComputeElementalShapeFunctionAtQuadPoints(
      std::vector<double> &elemental_shape_function_at_quad_points) const;

    /**
     * @brief compute elemental shape function derivative at quadrature points
     * @param[out] elemental_shape_function_derivative_at_quad_points
     * a matrix (stored in a Fortran style 1-D array) of
     * size (\f$N^{ele}_{quad} \times N^{ele}_{node}\f$) storing \f$
     * \left.\frac{d\mathbf{N}^{ele}_{i}}{d\xi}\right|_{\xi_{quad}}  =
     * \sum_{k \neq i} \frac{1}{\xi_{i}-\xi_{k}}\prod_{j \neq i, j \neq
     * k}^{N_{ele}}
     * \frac{\xi_{quad} - \xi_{j}}{\xi_{i} - \xi_{j}} \f$, the derivatives of
     * the shape
     * function at quadrature points. (Row: quad points, Col: nodal points)
     */
    void
    ComputeElementalShapeFunctionDerivativeAtQuadPoints(
      std::vector<double> &elemental_shape_function_derivative_at_quad_points)
      const;

    void
    ComputeJacobTimesWeightQuadValues(
      std::vector<double> &jacob_times_weight_quad_values,
      std::vector<double> &jacobian_on_ele) const;

    void
    ComputeInvjacobTimesWeightQuadValues(
      std::vector<double> &invjacob_times_weight_quad_values,
      std::vector<double> &invjacobian_on_ele) const;

    void
    ComputeElementalNINJ(DoubleDVec &elemental_Ni_Nj) const;

    void
    ComputeElementaldNIdNJ(DoubleDVec &elemental_dNi_dNj) const;
  };

  typedef std::array<const Tucker2EI::FE *, 3> FEPtrContainer;
} // namespace Tucker2EI



#endif // TUCKER2EI__FE_H_
