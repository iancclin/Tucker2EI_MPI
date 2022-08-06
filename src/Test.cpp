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


#include <vector>
#include <cstdlib>
#include <chrono>
#include <iostream>
#include "FE3D.h"

int
main()
{
  std::string file_name_x = "x_coord.dat";
  std::string file_name_y = "y_coord.dat";
  std::string file_name_z = "z_coord.dat";

  unsigned int    num_ele_x = 26, num_ele_y = 26, num_ele_z = 33;
  Tucker2EI::FE3D fe(file_name_x,
                     file_name_y,
                     file_name_z,
                     num_ele_x,
                     num_ele_y,
                     num_ele_z,
                     Tucker2EI::PT_8,
                     Tucker2EI::PT_8,
                     Tucker2EI::PT_8);

  const Tucker2EI::FE &fe_x = fe.fe_x;

  //  const auto &saq = fe_x.GetElementalShapeFunctionAtQuadPoints();
  //  const auto &saq = fe_x.GetElementalShapeFunctionDerivativeAtQuadPoints();
  //
  //  for (int j = 0; j < fe_x.GetNumberQuadPointsPerElement(); ++j)
  //    {
  //      for (int i = 0; i < fe_x.GetNumberNodesPerElement(); ++i)
  //        {
  //          printf("%.16f, ", saq[i*fe_x.GetNumberQuadPointsPerElement()+j]);
  //        }
  //      std::cout << std::endl;
  //    }

  std::cout << std::endl << std::endl;

  auto x_n = fe_x.GetNodalCoord();
  for (int i = 0; i < x_n.size(); ++i)
  {
    x_n[i] *= x_n[i];
  }

  printf("%.10f\n", fe_x.IntegrateWithNodalValues(x_n));
}