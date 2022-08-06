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

#ifndef TUCKER2EI_MPI_SRC_FECELL_H
#define TUCKER2EI_MPI_SRC_FECELL_H

#include "FeMap.h"
#include <vector>
#include <mpi.h>
#include <Tucker_Tensor.hpp>

namespace Tucker2EI
{
  class FECell
  {
  public:
    /**
     * @brief construct the FE cell object from a sequential Tucker Tensor
     * @param [in] tensor the tensor to be distributed
     * @param [in] fe_map the map for broadcasting data
     * @param [in] owned_rank which rank owns the sequential tensor
     */
    FECell(Tucker::Tensor *tensor, const FEMap &fe_map, int owned_rank);

  private:
    const FEMap        &fe_map_;
    std::vector<double> fe_cells_;
  };
} // namespace Tucker2EI

#endif // TUCKER2EI_MPI_SRC_FECELL_H
