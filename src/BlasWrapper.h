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

#ifndef TUCKER2EI__BLASWRAPPER_H_
#define TUCKER2EI__BLASWRAPPER_H_

namespace Tucker2EI
{
  namespace blas_wrapper
  {
    const static double       kDOne     = 1.0;
    const static double       kDZero    = 0.0;
    const static unsigned int kUIOne    = 1;
    const static char         kNonTrans = 'N';

    void
    Dgemvn(unsigned int  m,
           unsigned int  n,
           const double *a,
           const double *x,
           double       *y);

    void
    Dgemm(unsigned int  m,
          unsigned int  k,
          unsigned int  n,
          const double *A,
          const double *B,
          double       *C);

    void
    Daxpy(unsigned int n, double alpha, const double *x, double *y);

  } // namespace blas_wrapper

} // namespace Tucker2EI

#endif // TUCKER2EI__BLASWRAPPER_H_
