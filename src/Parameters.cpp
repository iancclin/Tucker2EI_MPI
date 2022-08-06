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

#include <fstream>
#include <iostream>
#include <sstream>
#include <utility>
#include "Parameters.h"
#include "DataReader.h"

Tucker2EI::Parameters::Parameters()
  : atom_filename_("atom_coord.inp")
  , x_coord_filename_("x_coord.dat")
  , y_coord_filename_("y_coord.dat")
  , z_coord_filename_("z_coord.dat")
{}

Tucker2EI::Parameters::Parameters(std::string param_filename)
  : param_filename_(std::move(param_filename))
{
  std::ifstream fin(param_filename_.c_str());

  for (std::string line; std::getline(fin, line, '=');)
  {
    std::string content;
    std::getline(fin, content);
    if (line == "input wavefunction order" ||
        line == "output wavefunction order")
    {
      content = std::string(content.begin() + 1, content.end() - 1);
    }
    std::istringstream ssin(content);

    if (line == "input wavefunction order")
    {
      int input_order;
      while (ssin >> input_order)
        input_order_.push_back(input_order);
    }
    else if (line == "output wavefunction order")
    {
      int output_order;
      while (ssin >> output_order)
        printout_wfns_.push_back(output_order);
    }
    else if (line == "num ele x")
    {
      ssin >> num_ele_x_;
    }
    else if (line == "num ele y")
    {
      ssin >> num_ele_y_;
    }
    else if (line == "num ele z")
    {
      ssin >> num_ele_z_;
    }
    else if (line == "num fe quad x")
    {
      ssin >> num_fe_quad_x_;
    }
    else if (line == "num fe quad y")
    {
      ssin >> num_fe_quad_y_;
    }
    else if (line == "num fe quad z")
    {
      ssin >> num_fe_quad_z_;
    }
    else if (line == "num conv fe quad x")
    {
      ssin >> num_fe_conv_quad_x_;
    }
    else if (line == "num conv fe quad y")
    {
      ssin >> num_fe_conv_quad_y_;
    }
    else if (line == "num conv fe quad z")
    {
      ssin >> num_fe_conv_quad_z_;
    }
    else if (line == "tensor tol")
    {
      ssin >> tensor_tol_;
    }
    else if (line == "a square")
    {
      ssin >> a_square_;
    }
    else if (line == "alpha filename")
    {
      ssin >> alpha_filename_;
    }
    else if (line == "omega filename")
    {
      ssin >> omega_filename_;
    }
    else if (line == "atom position file")
    {
      ssin >> atom_filename_;
    }
    else if (line == "x coord file")
    {
      ssin >> x_coord_filename_;
    }
    else if (line == "y coord file")
    {
      ssin >> y_coord_filename_;
    }
    else if (line == "z coord file")
    {
      ssin >> z_coord_filename_;
    }
    else if (line == "output prefix")
    {
      ssin >> output_prefix_;
    }
  }
  fin.close();

  fin.open(atom_filename_.c_str());
  std::string atom_line;
  std::getline(fin, atom_line);
  std::istringstream atom_ssin(atom_line);

  int num_atoms;
  atom_ssin >> num_atoms;
  atom_ssin.clear();
  atom_ = std::vector<std::array<double, 4>>(num_atoms);

  for (int i = 0; i < num_atoms; ++i)
  {
    std::getline(fin, atom_line);
    atom_ssin.str(atom_line);
    for (int j = 0; j < 4; ++j)
    {
      atom_ssin >> atom_[i][j];
    }
    atom_ssin.clear();
  }

  Tucker2EI::utils::VecReader(x_coord_filename_, x_coord_);
  Tucker2EI::utils::VecReader(y_coord_filename_, y_coord_);
  Tucker2EI::utils::VecReader(z_coord_filename_, z_coord_);
}
void
Tucker2EI::Parameters::Print()
{
  printf("Parameters: \n");

  printf("Atomic position file: %s\n", atom_filename_.c_str());
  printf("x coord file: %s\n", x_coord_filename_.c_str());
  printf("y coord file: %s\n", y_coord_filename_.c_str());
  printf("z coord file: %s\n", z_coord_filename_.c_str());
  printf("output prefix %s\n", output_prefix_.c_str());
  printf("input wavefunction order: ");
  for (const auto i : input_order_)
    printf("%d ", i);
  printf("\ninput wavefunction order: ");
  for (const auto i : printout_wfns_)
    printf("%d ", i);
  printf("\nnum ele in x: %d\n", num_ele_x_);
  printf("num ele in y: %d\n", num_ele_y_);
  printf("num ele in z: %d\n", num_ele_z_);
  printf("num quad points per ele in x: %d\n", num_fe_quad_x_);
  printf("num quad points per ele in y: %d\n", num_fe_quad_y_);
  printf("num quad points per ele in z: %d\n", num_fe_quad_z_);
  printf("num quad points per ele in x for convolution: %d\n",
         num_fe_conv_quad_x_);
  printf("num quad points per ele in y for convolution: %d\n",
         num_fe_conv_quad_y_);
  printf("num quad points per ele in z for convolution: %d\n",
         num_fe_conv_quad_z_);
  printf("Tolerance for tensor decomposition: %.6e\n", tensor_tol_);
  printf("A square value: %.18e\n", a_square_);
  printf("alpha filename: %s\n", alpha_filename_.c_str());
  printf("omega filename: %s\n", omega_filename_.c_str());

  printf("\n\nAtomic positions (x, y, z, charge): \n");
  printf("Number of atoms: %d\n", atom_.size());
  for (const auto &i : atom_)
  {
    for (const auto j : i)
      printf("%.8f ", j);
    printf("\n");
  }

  printf("\n\nx, y, z coordinates: \n");
  printf("Number of nodes in each direction: (%d, %d, %d)\n",
         x_coord_.size(),
         y_coord_.size(),
         z_coord_.size());
  printf("\nx coordinates: \n");
  for (const auto i : x_coord_)
    printf("\t%.16e\n", i);
  printf("\ny coordinates: \n");
  for (const auto i : y_coord_)
    printf("\t%.16e\n", i);
  printf("\nz coordinates: \n");
  for (const auto i : z_coord_)
    printf("\t%.16e\n", i);
}
const std::string &
Tucker2EI::Parameters::GetOutputPrefix() const
{
  return output_prefix_;
}
const std::vector<int> &
Tucker2EI::Parameters::GetInputOrder() const
{
  return input_order_;
}
const std::vector<int> &
Tucker2EI::Parameters::GetPrintoutWfns() const
{
  return printout_wfns_;
}
unsigned int
Tucker2EI::Parameters::GetNumEleX() const
{
  return num_ele_x_;
}
unsigned int
Tucker2EI::Parameters::GetNumEleY() const
{
  return num_ele_y_;
}
unsigned int
Tucker2EI::Parameters::GetNumEleZ() const
{
  return num_ele_z_;
}
double
Tucker2EI::Parameters::GetTensorTol() const
{
  return tensor_tol_;
}
const std::string &
Tucker2EI::Parameters::GetOmegaFilename() const
{
  return omega_filename_;
}
const std::string &
Tucker2EI::Parameters::GetAlphaFilename() const
{
  return alpha_filename_;
}
double
Tucker2EI::Parameters::GetASquare() const
{
  return a_square_;
}
const std::vector<std::array<double, 4>> &
Tucker2EI::Parameters::GetAtom() const
{
  return atom_;
}
int
Tucker2EI::Parameters::GetNumFeQuadX() const
{
  return num_fe_quad_x_;
}
int
Tucker2EI::Parameters::GetNumFeQuadY() const
{
  return num_fe_quad_y_;
}
int
Tucker2EI::Parameters::GetNumFeQuadZ() const
{
  return num_fe_quad_z_;
}
int
Tucker2EI::Parameters::GetNumFeConvQuadX() const
{
  return num_fe_conv_quad_x_;
}
int
Tucker2EI::Parameters::GetNumFeConvQuadY() const
{
  return num_fe_conv_quad_y_;
}
int
Tucker2EI::Parameters::GetNumFeConvQuadZ() const
{
  return num_fe_conv_quad_z_;
}
const std::vector<double> &
Tucker2EI::Parameters::GetXCoord() const
{
  return x_coord_;
}
const std::vector<double> &
Tucker2EI::Parameters::GetYCoord() const
{
  return y_coord_;
}
const std::vector<double> &
Tucker2EI::Parameters::GetZCoord() const
{
  return z_coord_;
}
const std::string &
Tucker2EI::Parameters::GetXCoordFilename() const
{
  return x_coord_filename_;
}
const std::string &
Tucker2EI::Parameters::GetYCoordFilename() const
{
  return y_coord_filename_;
}
const std::string &
Tucker2EI::Parameters::GetZCoordFilename() const
{
  return z_coord_filename_;
}
