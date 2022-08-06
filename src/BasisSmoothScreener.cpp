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

#include <cmath>
#include "BasisSmoothScreener.h"

Tucker2EI::BasisSmoothScreener::BasisSmoothScreener(
  const Tucker2EI::FE3D                    &fe,
  const std::vector<std::array<double, 4>> &atom,
  double                                    spread_t,
  double                                    cutoff_r)
{
  Tucker::SizeArray size_array(3);
  size_array[0] = fe.fe_x.GetNumberNodes();
  size_array[1] = fe.fe_y.GetNumberNodes();
  size_array[2] = fe.fe_z.GetNumberNodes();

  screen_function_ =
    Tucker::MemoryManager::safe_new<Tucker::Tensor>(size_array);
  screen_function_->initialize();

  // em stands for the center of the electricity
  double x_em = 0.0, y_em = 0.0, z_em = 0.0;
  double total_charge = 0.0;
  for (const auto &atom_i : atom)
  {
    double charge = atom_i[3];
    x_em += charge * atom_i[0];
    y_em += charge * atom_i[1];
    z_em += charge * atom_i[2];
    total_charge += charge;
  }

  x_em /= total_charge;
  y_em /= total_charge;
  z_em /= total_charge;

  const auto &x_coord       = fe.fe_x.GetNodalCoord();
  const auto &y_coord       = fe.fe_y.GetNodalCoord();
  const auto &z_coord       = fe.fe_z.GetNodalCoord();
  double     *screen_fn_ptr = screen_function_->data();
  printf("size: %d\n", screen_function_->getNumElements());
  printf("size: (%d, %d, %d)\n", size_array[0], size_array[1], size_array[2]);
  int cnt = 0;
  for (int k = 0; k < size_array[2]; ++k)
  {
    for (int j = 0; j < size_array[1]; ++j)
    {
      for (int i = 0; i < size_array[0]; ++i)
      {
        double xx = x_coord[i] - x_em;
        double yy = y_coord[j] - y_em;
        double zz = z_coord[k] - z_em;
        double r  = std::sqrt(xx * xx + yy * yy + zz * zz);
        double rt = 1.0 - spread_t * (r - cutoff_r) / cutoff_r;
        double ur = 0.0, urp = 0.0;
        if (rt > 0)
          ur = std::exp(-1.0 / rt);
        if ((1 - rt) > 0)
          urp = std::exp(-1.0 / (1 - rt));
        *screen_fn_ptr = ur / (ur + urp);
        screen_fn_ptr++;
      }
    }
  }
}
Tucker2EI::BasisSmoothScreener::~BasisSmoothScreener()
{
  Tucker::MemoryManager::safe_delete(screen_function_);
}
Tucker::Tensor *
Tucker2EI::BasisSmoothScreener::GetScreener()
{
  return screen_function_;
}
