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

#include "Integrator.h"
#include "DataReader.h"

Tucker2EI::Integrator::Integrator(const Tucker2EI::FE3D &fe)
  : fe_(fe)
  , num_node_(3)
  , num_quad_(3)
{
  for (int dim = 0; dim < 3; ++dim)
  {
    num_node_[dim] = fe.fe[dim]->GetNumberNodes();
    num_quad_[dim] = fe.fe[dim]->GetNumberQuadPoints();
  }
}

void
Tucker2EI::Integrator::ReadWfn(const int idx, Tucker::Tensor *&tensor)
{
  Tucker2EI::utils::TensorReader("waveFunc_" + std::to_string(idx) + ".dat",
                                 num_node_[0],
                                 num_node_[1],
                                 num_node_[2],
                                 tensor);
}