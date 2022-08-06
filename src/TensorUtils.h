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

#ifndef TUCKER2EI__TENSORUTILS_H_
#define TUCKER2EI__TENSORUTILS_H_

#include "Tucker.hpp"
#include <string>

namespace Tucker2EI
{
  namespace tensor
  {
    namespace utils
    {
      /**
       * @brief Initialize a I*J*K tensor and read in data from the file.
       * @param file_name the file
       * @param size_i size in I direction
       * @param size_j size in J direction
       * @param size_k size in K direction
       * @param tensor the tensor
       */
      void
      ReadInTensor(const std::string &file_name, Tucker::Tensor *tensor);
    } // namespace utils
  };  // namespace tensor
};    // namespace Tucker2EI

#endif // TUCKER2EI__TENSORUTILS_H_
