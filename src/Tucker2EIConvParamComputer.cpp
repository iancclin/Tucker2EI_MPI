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

#include <string>
#include <vector>
#include "FE3D.h"
#include <cmath>

int
main()
{
  unsigned int num_ele_x = 26, num_ele_y = 26, num_ele_z = 40;

  std::string file_name_x = "x_coord.dat";
  std::string file_name_y = "y_coord.dat";
  std::string file_name_z = "z_coord.dat";

  Tucker2EI::FE3D fe(file_name_x,
                     file_name_y,
                     file_name_z,
                     num_ele_x,
                     num_ele_y,
                     num_ele_z,
                     Tucker2EI::PT_10,
                     Tucker2EI::PT_10,
                     Tucker2EI::PT_10);

  Tucker2EI::FE3D fe_conv(file_name_x,
                          file_name_y,
                          file_name_z,
                          num_ele_x,
                          num_ele_y,
                          num_ele_z,
                          Tucker2EI::PT_14,
                          Tucker2EI::PT_14,
                          Tucker2EI::PT_14);

  const auto &quad_x = fe.fe_x.GetNodalCoord();
  const auto &quad_y = fe.fe_y.GetNodalCoord();
  const auto &quad_z = fe.fe_z.GetNodalCoord();

  const auto &conv_quad_x = fe_conv.fe_x.GetQuadCoord();
  const auto &conv_quad_y = fe_conv.fe_y.GetQuadCoord();
  const auto &conv_quad_z = fe_conv.fe_z.GetQuadCoord();

  double max_rsqx = 0.0, max_rsqy = 0.0, max_rsqz = 0.0;
  double min_rsqx = 0.0, min_rsqy = 0.0, min_rsqz = 0.0;

  auto compute_minmax = [](const std::vector<double> &q,
                           const std::vector<double> &cq,
                           double                    &min,
                           double                    &max) {
    min      = 5e10;
    int n_q  = q.size();
    int n_cq = cq.size();
    for (int i = 0; i < n_q; ++i)
    {
      for (int j = 0; j < n_cq; ++j)
      {
        double r = q[i] - cq[j];
        r        = r * r;
        if (r > max)
          max = r;
        if (r < min)
          min = r;
      }
    }
  };

  compute_minmax(quad_x, conv_quad_x, min_rsqx, max_rsqx);
  compute_minmax(quad_y, conv_quad_y, min_rsqy, max_rsqy);
  compute_minmax(quad_z, conv_quad_z, min_rsqz, max_rsqz);

  printf("max: (%.10e, %.10e, %.10e)\n", max_rsqx, max_rsqy, max_rsqz);
  printf("min: (%.10e, %.10e, %.10e)\n", min_rsqx, min_rsqy, min_rsqz);

  double b_sq = max_rsqx + max_rsqy + max_rsqz;
  double a_sq = min_rsqx + min_rsqy + min_rsqz;
  double R    = b_sq / a_sq;
  printf("R: %.16e, A square: %.16e\n", R, a_sq);
  printf("a: %.16e, b: %.16e\n", std::sqrt(a_sq), std::sqrt(b_sq));
}