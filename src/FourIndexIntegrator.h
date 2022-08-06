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

#ifndef TUCKER2EI_SRC_FOURINDEXINTEGRATOR_H_
#define TUCKER2EI_SRC_FOURINDEXINTEGRATOR_H_



#include <array>
#include "FE3D.h"
#include "ConvolutionComputer.h"
#include "Integrator.h"

namespace Tucker2EI
{
  class FourIndexIntegrator : public Integrator
  {
  public:
    FourIndexIntegrator(const FE3D                &fe,
                        const FE3D                &fe_conv,
                        const ConvolutionComputer &conv_comp,
                        double                     decomp_tol);

    double
    Integrate(std::array<int, 4> f_idx);

  private:
    const FE3D                &fe_conv_;
    const ConvolutionComputer &conv_comp_;
    double                     decomp_tol_;
  };
} // namespace Tucker2EI


#endif // TUCKER2EI_SRC_FOURINDEXINTEGRATOR_H_
