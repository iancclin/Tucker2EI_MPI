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

#include "BlasWrapper.h"

extern "C"
{
  void
  dgemv_(const char         *trans,
         const unsigned int *m,
         const unsigned int *n,
         const double       *alpha,
         const double       *a,
         const unsigned int *lda,
         const double       *x,
         const unsigned int *incx,
         const double       *beta,
         double             *y,
         const unsigned int *incy);

  void
  daxpy_(const unsigned int *n,
         const double       *alpha,
         const double       *x,
         const unsigned int *incx,
         double             *y,
         const unsigned int *incy);

  void
  dgemm_(const char         *transa,
         const char         *transb,
         const unsigned int *m,
         const unsigned int *n,
         const unsigned int *k,
         const double       *alpha,
         const double       *a,
         const unsigned int *lda,
         const double       *b,
         const unsigned int *ldb,
         const double       *beta,
         double             *c,
         const unsigned int *ldc);
}

/** @todo change this to inline function (not do-able for now since inline
 * function has to be defined in the header.)
 */
void
Tucker2EI::blas_wrapper::Dgemvn(const unsigned int m,
                                const unsigned int n,
                                const double      *a,
                                const double      *x,
                                double            *y)
{
  dgemv_(&kNonTrans, &m, &n, &kDOne, a, &m, x, &kUIOne, &kDZero, y, &kUIOne);
}
void
Tucker2EI::blas_wrapper::Daxpy(unsigned int  n,
                               double        alpha,
                               const double *x,
                               double       *y)
{
  daxpy_(&n, &alpha, x, &kUIOne, y, &kUIOne);
}
void
Tucker2EI::blas_wrapper::Dgemm(unsigned int  m,
                               unsigned int  k,
                               unsigned int  n,
                               const double *A,
                               const double *B,
                               double       *C)
{
  dgemm_(
    &kNonTrans, &kNonTrans, &m, &n, &k, &kDOne, A, &m, B, &k, &kDZero, C, &m);
}
