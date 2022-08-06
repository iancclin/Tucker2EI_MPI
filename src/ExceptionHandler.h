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
 * @credit Bikash Kanungo from DFT-EFE project
 * https://github.com/dftfeDevelopers/dft-efe.git
 */

#ifndef TUCKER2EI__EXCEOPTIONHANDLER_H_
#define TUCKER2EI__EXCEOPTIONHANDLER_H_

#undef TUCKER2EI_ASSERT
#undef TUCKER2EI_ASSERTWITHSTRING

#if defined(TUCKER2EI_DISABLE_ASSERT) || \
  (!defined(TUCKER2EI_ENABLE_ASSERT) && defined(NDEBUG))
#  include <assert.h> // .h to support old libraries w/o <cassert> - effect is the same
#  define TUCKER2EI_ASSERT(expr) ((void)0)
#  define TUCKER2EI_ASSERTWITHSTRING(expr, msg) ((void)0)

#elif defined(TUCKER2EI_ENABLE_ASSERT) && defined(NDEBUG)
#  undef NDEBUG // disabling NDEBUG to forcibly enable assert for sources that
                // set TUCKER2EI_ENABLE_ASSERT even when in release mode (with
                // NDEBUG)
#  include <assert.h> // .h to support old libraries w/o <cassert> - effect is the same
#  define TUCKER2EI_ASSERT(expr) assert(expr)
#  define TUCKER2EI_ASSERTWITHSTRING(expr, msg) assert((expr) && (msg))

#else
#  include <assert.h> // .h to support old libraries w/o <cassert> - effect is the same
#  define TUCKER2EI_ASSERT(expr) assert(expr)
#  define TUCKER2EI_ASSERTWITHSTRING(expr, msg) assert((expr) && (msg))

#endif

#endif // TUCKER2EI__EXCEOPTIONHANDLER_H_
