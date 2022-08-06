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

#ifndef TUCKER2EI_SRC_PARAMETERS_H_
#define TUCKER2EI_SRC_PARAMETERS_H_

#include <string>
#include <vector>
#include <array>
namespace Tucker2EI
{

  class Parameters
  {
  public:
    Parameters();

    explicit Parameters(std::string param_filename);

    void
    Print();

    const std::string &
    GetOutputPrefix() const;
    const std::vector<int> &
    GetInputOrder() const;
    const std::vector<int> &
    GetPrintoutWfns() const;
    const std::string &
    GetXCoordFilename() const;
    const std::string &
    GetYCoordFilename() const;
    const std::string &
    GetZCoordFilename() const;
    unsigned int
    GetNumEleX() const;
    unsigned int
    GetNumEleY() const;
    unsigned int
    GetNumEleZ() const;
    double
    GetTensorTol() const;
    const std::string &
    GetOmegaFilename() const;
    const std::string &
    GetAlphaFilename() const;
    double
    GetASquare() const;
    const std::vector<std::array<double, 4>> &
    GetAtom() const;
    int
    GetNumFeQuadX() const;
    int
    GetNumFeQuadY() const;
    int
    GetNumFeQuadZ() const;
    int
    GetNumFeConvQuadX() const;
    int
    GetNumFeConvQuadY() const;
    int
    GetNumFeConvQuadZ() const;
    const std::vector<double> &
    GetXCoord() const;
    const std::vector<double> &
    GetYCoord() const;
    const std::vector<double> &
    GetZCoord() const;

  private:
    std::string atom_filename_;
    std::string x_coord_filename_, y_coord_filename_, z_coord_filename_;
    std::string param_filename_;

    std::string output_prefix_;
    // input_order_: the order of inputs wavefunctions
    // printout_wfns: how many states of 2EI are printed out
    std::vector<int>                   input_order_, printout_wfns_;
    unsigned int                       num_ele_x_, num_ele_y_, num_ele_z_;
    double                             tensor_tol_;
    std::string                        omega_filename_, alpha_filename_;
    double                             a_square_;
    std::vector<std::array<double, 4>> atom_;
    int num_fe_quad_x_, num_fe_quad_y_, num_fe_quad_z_;
    int num_fe_conv_quad_x_, num_fe_conv_quad_y_, num_fe_conv_quad_z_;
    std::vector<double> x_coord_, y_coord_, z_coord_;
  };

} // namespace Tucker2EI



#endif // TUCKER2EI_SRC_PARAMETERS_H_
