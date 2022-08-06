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

#include <memory>
#include "DataReader.h"

void
Tucker2EI::utils::VecReader(const std::string   &file_name,
                            std::vector<double> &vec)
{
  /**
   * @todo add file not exists exception
   */
  std::ifstream input(file_name);
  double        t;
  while (input >> t)
  {
    vec.push_back(t);
  }
  input.close();
}
void
Tucker2EI::utils::TensorReader(const std::string &file_name,
                               int                size_x,
                               int                size_y,
                               int                size_z,
                               Tucker::Tensor   *&tensor)
{
  Tucker::SizeArray size_array(3);
  size_array[0] = size_x;
  size_array[1] = size_y;
  size_array[2] = size_z;

  tensor = Tucker::MemoryManager::safe_new<Tucker::Tensor>(size_array);
  tensor->initialize();

  //  /**
  //   * @todo add file not exists exception
  //   */
  double *tensor_ptr = tensor->data();
  //  std::ifstream input(file_name);
  //  {
  //    double t;
  //    while (input >> t)
  //    {
  //      *(tensor_ptr++) = t;
  //    }
  //    input.close();
  //  }

  int         size = size_x * size_y * size_z;
  std::string newfilename(file_name.begin(), file_name.end() - 3);
  newfilename = newfilename + "bin";
  std::fstream fin(newfilename, std::ios::in | std::ios::binary);
  fin.read((char *)tensor_ptr, size * sizeof(double));
  fin.close();
  fin.clear();
}

void
Tucker2EI::utils::TensorPrint(const std::string &file_name,
                              Tucker::Tensor   *&tensor)
{
  double *tensor_ptr = tensor->data();
  FILE   *file       = fopen(file_name.c_str(), "w");
  int     n_data     = tensor->getNumElements();
  for (int i = 0; i < n_data; ++i)
  {
    fprintf(file, "%.16e ", *tensor_ptr);
    tensor_ptr++;
  }
  fclose(file);
}