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

#ifndef TUCKER2EI__DATAREADER_H_
#define TUCKER2EI__DATAREADER_H_

#include <string>
#include <fstream>
#include <vector>
#include <memory>
#include <Tucker.hpp>



namespace Tucker2EI
{
  namespace utils
  {
    void
    VecReader(const std::string &file_name, std::vector<double> &vec);

    /**
     * @brief Read in the tensor data from DFT-FE wavefunctions
     * @param file_name filename of the wavefunction values
     * @param size_x x-dimension
     * @param size_y y-dimension
     * @param size_z z-dimension
     * @param tensor pointer to the tensor object
     */
    void
    TensorReader(const std::string &file_name,
                 int                size_x,
                 int                size_y,
                 int                size_z,
                 Tucker::Tensor   *&tensor);

    void
    TensorPrint(const std::string &file_name, Tucker::Tensor *&tensor);
  } // namespace utils
} // namespace Tucker2EI

#endif // TUCKER2EI__DATAREADER_H_
