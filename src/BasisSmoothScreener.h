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

#ifndef TUCKER2EI_SRC_BASISSMOOTHSCREENER_H_
#define TUCKER2EI_SRC_BASISSMOOTHSCREENER_H_

#include "Tucker.hpp"
#include "FE3D.h"

namespace Tucker2EI
{

  class BasisSmoothScreener
  {
  public:
    BasisSmoothScreener(const FE3D                               &fe,
                        const std::vector<std::array<double, 4>> &atom,
                        double                                    spread_t,
                        double                                    cutoff_r);

    virtual ~BasisSmoothScreener();

    Tucker::Tensor *
    GetScreener();

  protected:
    Tucker::Tensor *screen_function_;
  };

} // namespace Tucker2EI

#endif // TUCKER2EI_SRC_BASISSMOOTHSCREENER_H_
