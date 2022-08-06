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

#ifndef TUCKER2EI_SRC_INTEGRATOR_H_
#define TUCKER2EI_SRC_INTEGRATOR_H_


#include "FE3D.h"
#include "BasisSmoothScreener.h"
namespace Tucker2EI
{
  class Integrator
  {
  public:
    explicit Integrator(const FE3D &fe);

  protected:
    const FE3D       &fe_;
    Tucker::SizeArray num_node_, num_quad_;
    /**
     * @brief Read in wavefunctions on tensor corresponding to the idx
     * @param[in] idx the index of the wavefunction
     * @param[out] tensor wavefunction
     */
    virtual void
    ReadWfn(int idx, Tucker::Tensor *&tensor);
  };
} // namespace Tucker2EI



#endif // TUCKER2EI_SRC_INTEGRATOR_H_
