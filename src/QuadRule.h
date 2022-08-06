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

#ifndef TUCKER2EI__QUADRULE_H_
#define TUCKER2EI__QUADRULE_H_

#include <vector>

namespace Tucker2EI
{
  enum QuadPoints
  {
    PT_2 = 2,
    PT_3,
    PT_4,
    PT_5,
    PT_6,
    PT_7,
    PT_8,
    PT_9,
    PT_10,
    PT_11,
    PT_12,
    PT_13,
    PT_14,
    PT_15,
    PT_16,
    PT_17,
    PT_18,
    PT_19,
    PT_20,
    PT_21,
    PT_22,
    PT_23,
    PT_24,
    PT_25,
    PT_26,
    PT_27,
    PT_28,
    PT_29,
    PT_30
  };
  class QuadRule
  {
  public:
    QuadRule() = default;

    QuadRule(const QuadRule &quad_rule) = default;

    explicit QuadRule(QuadPoints quad_points);

    void
    ChangeQuadPoints(Tucker2EI::QuadPoints quad_points);

    unsigned int
    GetNumberQuadPoints() const;

    const std::vector<double> &
    GetQuadPoints() const;

    const std::vector<double> &
    GetQuadWeights() const;

  private:
    unsigned int        number_quad_points_;
    std::vector<double> quad_points_;
    std::vector<double> quad_weights_;

    void
    GetQuadAndWeight(Tucker2EI::QuadPoints quad_points,
                     std::vector<double>  &points,
                     std::vector<double>  &weights);
  };

} // namespace Tucker2EI

#endif // TUCKER2EI__QUADRULE_H_
