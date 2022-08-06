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


#include <vector>
#include <cstdlib>
#include <chrono>
#include <iostream>
#include <algorithm>

#include <Tucker.hpp>


extern "C"
{
  void
  daxpy_(const unsigned int *n,
         const double       *alpha,
         const double       *x,
         const unsigned int *incx,
         double             *y,
         const unsigned int *incy);
}


int
getN(int N)
{
  return N * N * N;
}

int
main()
{
  //  int n = 300;
  //  int N = n*n*n;
  //  std::vector<double> a(N), b(N);
  //  for (int i = 0; i < N; ++i) {
  //      a[i] = rand();
  //      b[i] = rand();
  //    }
  //
  //  auto start = std::chrono::system_clock::now();
  //  for (int k = 0; k < 10; ++k)
  //    {
  //      for (int i = 0; i < N; ++i)
  //        {
  //          a[i] += b[i];
  //        }
  //    }
  //  auto end = std::chrono::system_clock::now();
  //  auto duration = end - start;
  //  std::cout << std::chrono::duration_cast<std::chrono::milliseconds>
  //    (duration).count() << "s\n";
  //
  //  start = std::chrono::system_clock::now();
  //  for (int k = 0; k < 10; ++k)
  //    {
  //      for (int i = 0; i < getN(n); ++i)
  //        {
  //          a[i] += b[i];
  //        }
  //    }
  //  end = std::chrono::system_clock::now();
  //  duration = end - start;
  //  std::cout << std::chrono::duration_cast<std::chrono::milliseconds>
  //               (duration).count() << "s\n";

  int n = 300;

  unsigned int N = n * n * n;


  Tucker::SizeArray size(3);
  size[0] = n, size[1] = n, size[2] = n;
  Tucker::Tensor *a = Tucker::MemoryManager::safe_new<Tucker::Tensor>(size);
  a->initialize();
  Tucker::Tensor *b = Tucker::MemoryManager::safe_new<Tucker::Tensor>(size);
  b->initialize();

  double *a_ptr = a->data(), *b_ptr = b->data();

  //  std::vector<double> a(N, 0.0), b(N, 0.0);
  for (int i = 0; i < N; ++i)
    a_ptr[i] = rand();
  for (int i = 0; i < N; ++i)
    b_ptr[i] = rand();

  double       alpha = 1.0;
  unsigned int inc   = 1;

  auto start    = std::chrono::system_clock::now();
  auto end      = std::chrono::system_clock::now();
  auto duration = end - start;

  start = std::chrono::system_clock::now();
  for (int k = 0; k < 210; ++k)
  {
    daxpy_(&N, &alpha, a->data(), &inc, b->data(), &inc);
  }
  end      = std::chrono::system_clock::now();
  duration = end - start;
  std::cout
    << "duration: "
    << std::chrono::duration_cast<std::chrono::milliseconds>(duration).count()
    << std::endl;

  start = std::chrono::system_clock::now();
  int q = a->getNumElements();
  for (int k = 0; k < 210; ++k)
  {
    for (unsigned int i = 0; i < n * n * n; ++i)
    {
      a_ptr[i] += b_ptr[i];
    }
  }
  end      = std::chrono::system_clock::now();
  duration = end - start;
  std::cout
    << "duration: "
    << std::chrono::duration_cast<std::chrono::milliseconds>(duration).count()
    << std::endl;

  start = std::chrono::system_clock::now();
  for (int k = 0; k < 210; ++k)
  {
    std::transform(
      a_ptr, a_ptr + a->getNumElements(), b_ptr, a_ptr, std::plus<double>());
  }
  end      = std::chrono::system_clock::now();
  duration = end - start;
  std::cout
    << "duration: "
    << std::chrono::duration_cast<std::chrono::milliseconds>(duration).count()
    << std::endl;
}