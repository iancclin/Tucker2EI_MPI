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

#ifndef TUCKER2EI_MPI_SRC_FOURINDEXGENERATOR_H
#define TUCKER2EI_MPI_SRC_FOURINDEXGENERATOR_H

/**
 * @brief This class generate all four-index integrals all at once and improved the computational efficiency
 */

#include "FeMap.h"
#include "Integrator.h"
#include "ConvolutionComputer.h"

namespace Tucker2EI
{
  class FourIndexGenerator : public Integrator
  {
  public:
    FourIndexGenerator(const FEMap               &fe_map,
                       const FEMap               &conv_fe_map,
                       const ConvolutionComputer &conv_comp,
                       double                     decomp_tol);
    void
    Calculate(const std::vector<int> &input_idx,
              std::vector<double> &unique_four_idx_int);
    const std::vector<double> &
    GetResults() const;

  protected:
    const MPI_Comm                   &comm;
    const FEMap               &fe_map_;
    const FEMap               &conv_fe_map_;
    const ConvolutionComputer &conv_comp_;
    double                     decomp_tol_;

    const std::vector<double> &local_jxw_;

    std::vector<double>
    ComputeConvolutionNodalCell(int k, int l);
  };
} // namespace Tucker2EI

#endif // TUCKER2EI_MPI_SRC_FOURINDEXGENERATOR_H
