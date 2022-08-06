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

#ifndef TUCKER2EI_MPI_SRC_FEMAP_H
#define TUCKER2EI_MPI_SRC_FEMAP_H

#include "FE3D.h"
#include <mpi.h>
#include <vector>

namespace Tucker2EI
{
  class FEMap
  {
  public:
    FEMap(const FE3D &fe, MPI_Comm comm);

    void
    DistributeSeqMultiNodalCell(int           num_fields,
                                const double *seq_nodal_cell,
                                double       *mpi_local_multinodalcell);

    void
    MPILocalMultinodalcell2Multiquadcell(const int     num_fields,
                                         const double *mpi_local_multinodalcell,
                                         double       *mpi_local_multiquadcell);

    const FE3D &
    GetFe() const;

    const MPI_Comm &
    GetComm() const;

    int
    GetMpiRank() const;
    int
    GetMpiSize() const;
    int
    GetNumLocalCells() const;
    int
    GetNumTotalCells() const;
    int
    GetNumLocalNodes() const;
    int
    GetNumTotalNodes() const;
    int
    GetNumLocalQuadPoints() const;
    int
    GetNumTotalQuadPoints() const;
    const std::vector<int> &
    GetCellMpiCnt() const;
    const std::vector<int> &
    GetCellMpiDispl() const;
    const std::vector<int> &
    GetNodalMpiCnt() const;
    const std::vector<int> &
    GetNodalMpiDispl() const;
    const std::vector<int> &
    GetQuadMpiCnt() const;
    const std::vector<int> &
    GetQuadMpiDispl() const;
    const std::vector<double> &
    GetLocalCellJxw() const;

    const FE3D &fe_;

  protected:
    MPI_Comm comm_;
    int      mpi_rank_, mpi_size_;
    int      num_local_cells_, num_total_cells_;
    int      num_local_nodes_;
    int      num_total_nodes_; // local nodes include ghost nodes, hence do not
                               // sum up to total nodes
    int num_total_cell_nodes_;

  public:
    int
    GetNumTotalCellNodes() const;

  protected:
    int                 num_local_quad_points_, num_total_quad_points_;
    std::vector<int>    cell_mpi_cnt_, cell_mpi_displ_;
    std::vector<int>    nodal_mpi_cnt_, nodal_mpi_displ_;
    std::vector<int>    quad_mpi_cnt_, quad_mpi_displ_;
    std::vector<double> local_cell_jxw_;

    void
    CellId2Cart(int id, int &x, int &y, int &z) const;
    void
    Cart2CellId(const int x, const int y, const int z, int &id) const;

    // const std::vector<double> &elemental_shape_function_at_quad_points_;
    //    int      num_total_cells_, num_local_cells_;
    //  sum up
    // to total nodes
    //    std::vector<double> mpi_cell_id_map_;
    //    std::vector<int>    local_cell_id_, global_cell_id_;
    //    std::vector<int> global_cell_cartesian_id_x_,
    //    global_cell_cartesian_id_y_,
    //      global_cell_cartesian_id_z_;
    //    std::vector<double>        local_cell_jacobian_;

    //    std::vector<double>        quad_points_weights_on_one_cell_;
  };
} // namespace Tucker2EI

#endif // TUCKER2EI_MPI_SRC_FEMAP_H
