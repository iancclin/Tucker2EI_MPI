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

#include "FeMap.h"
#include "FE3D.h"
#include "BlasWrapper.h"

Tucker2EI::FEMap::FEMap(const Tucker2EI::FE3D &fe, MPI_Comm comm)
  : fe_(fe)
  , comm_(comm)
{
  MPI_Comm_rank(comm_, &mpi_rank_);
  MPI_Comm_size(comm_, &mpi_size_);
  num_total_cells_ = fe.GetNum3DEle();
  num_local_cells_ = num_total_cells_ / mpi_size_ +
                     (mpi_rank_ < (num_total_cells_ % mpi_size_) ? 1 : 0);
  cell_mpi_cnt_.assign(mpi_size_, 0);
  cell_mpi_displ_.assign(mpi_size_ + 1, 0);
  nodal_mpi_cnt_.assign(mpi_size_, 0);
  nodal_mpi_displ_.assign(mpi_size_ + 1, 0);
  quad_mpi_cnt_.assign(mpi_size_, 0);
  quad_mpi_displ_.assign(mpi_size_ + 1, 0);

  for (int i = 0; i < mpi_size_; ++i)
  {
    int num_i_cells = num_total_cells_ / mpi_size_ +
                      (i < (num_total_cells_ % mpi_size_) ? 1 : 0);
    cell_mpi_cnt_[i]        = num_i_cells;
    cell_mpi_displ_[i + 1]  = cell_mpi_displ_[i] + cell_mpi_cnt_[i];
    nodal_mpi_cnt_[i]       = num_i_cells * fe_.GetNumNodesPer3Dele();
    nodal_mpi_displ_[i + 1] = nodal_mpi_displ_[i] + nodal_mpi_cnt_[i];
    quad_mpi_cnt_[i]        = num_i_cells * fe_.GetNumQuadPointsPer3Dele();
    quad_mpi_displ_[i + 1]  = quad_mpi_displ_[i] + quad_mpi_cnt_[i];
  }

  num_local_nodes_ = num_local_cells_ * fe_.GetNumNodesPer3Dele();
  if (num_local_nodes_ != nodal_mpi_cnt_[mpi_rank_])
  {
    std::cout
      << "mpi rank: " << mpi_rank_
      << "num_local_nodes_ and nodal_mpi_cnt_[mpi_rank_] do not match\n";
    std::terminate();
  }
  num_total_nodes_ = fe_.fe_x.GetNumberNodes() * fe_.fe_y.GetNumberNodes() *
                     fe_.fe_z.GetNumberNodes();
  num_total_cell_nodes_ = num_total_cells_ * fe_.GetNumNodesPer3Dele();;
  num_local_quad_points_ = num_local_cells_ * fe_.GetNumQuadPointsPer3Dele();
  local_cell_jxw_.assign(num_local_quad_points_, 0.0);

  const auto &jxw_x                = fe_.fe_x.GetJacobTimesWeightQuadValues();
  const auto &jxw_y                = fe_.fe_y.GetJacobTimesWeightQuadValues();
  const auto &jxw_z                = fe_.fe_z.GetJacobTimesWeightQuadValues();
  int         num_quads_per_cell_x = fe_.fe_x.GetNumberQuadPointsPerElement();
  int         num_quads_per_cell_y = fe_.fe_y.GetNumberQuadPointsPerElement();
  int         num_quads_per_cell_z = fe_.fe_z.GetNumberQuadPointsPerElement();
  for (int cell_i = 0, cnt = 0; cell_i < num_local_cells_; ++cell_i)
  {
    int cell_id = cell_mpi_displ_[mpi_rank_] + cell_i;
    int cell_id_x, cell_id_y, cell_id_z;
    CellId2Cart(cell_id, cell_id_x, cell_id_y, cell_id_z);

    int quad_start_x = cell_id_x * num_quads_per_cell_x;
    int quad_start_y = cell_id_y * num_quads_per_cell_y;
    int quad_start_z = cell_id_z * num_quads_per_cell_z;

    for (int k = 0; k < num_quads_per_cell_z; ++k)
    {
      for (int j = 0; j < num_quads_per_cell_y; ++j)
      {
        for (int i = 0; i < num_quads_per_cell_x; ++i)
        {
          int ii               = quad_start_x + i;
          int jj               = quad_start_y + j;
          int kk               = quad_start_z + k;
          local_cell_jxw_[cnt] = jxw_x[ii] * jxw_y[jj] * jxw_z[kk];
          cnt++;
        }
      }
    }
  }
}
const Tucker2EI::FE3D &
Tucker2EI::FEMap::GetFe() const
{
  return fe_;
}
const MPI_Comm &
Tucker2EI::FEMap::GetComm() const
{
  return comm_;
}
void
Tucker2EI::FEMap::DistributeSeqMultiNodalCell(int           num_fields,
                                              const double *seq_nodal_cell,
                                              double *mpi_local_multinodalcell)
{
  std::vector<double> send_buffer(num_fields * num_local_nodes_, 0.0);

  auto iter = send_buffer.begin();
  for (int i_rank = 0; i_rank = mpi_size_; ++i_rank)
  {
    int owned_node_begin_id = nodal_mpi_displ_[i_rank];
    int owned_node_end_id   = nodal_mpi_displ_[i_rank + 1];
    for (int i_field = 0; i_field < num_fields; ++i_field)
    {
      const double *begin =
        seq_nodal_cell + i_field * num_total_nodes_ + owned_node_begin_id;
      const double *end =
        seq_nodal_cell + i_field * num_total_nodes_ + owned_node_end_id;
      iter = std::copy(begin, end, iter);
    }
  }

  std::vector<int> mpi_cnt = nodal_mpi_cnt_;
  std::vector<int> mpi_displ(mpi_size_ + 1, 0);
  for (int i = 0; i < mpi_size_; ++i)
  {
    mpi_cnt[i] *= num_fields;
    mpi_displ[i + 1] = mpi_cnt[i] + mpi_displ[i];
  }

  MPI_Scatterv(send_buffer.data(),
               mpi_cnt.data(),
               mpi_displ.data(),
               MPI_DOUBLE,
               mpi_local_multinodalcell,
               num_local_nodes_ * num_fields,
               MPI_DOUBLE,
               mpi_rank_,
               comm_);
}
void
Tucker2EI::FEMap::MPILocalMultinodalcell2Multiquadcell(
  const int     num_fields,
  const double *mpi_local_multinodalcell,
  double       *mpi_local_multiquadcell)
{
  const auto &cell_shape_functions_at_quad_points =
    fe_.GetElementalShapeFunctionAtQuadPoints();
  blas_wrapper::Dgemm(fe_.GetNumQuadPointsPer3Dele(),
                      fe_.GetNumNodesPer3Dele(),
                      num_local_cells_ * num_fields,
                      cell_shape_functions_at_quad_points.data(),
                      mpi_local_multinodalcell,
                      mpi_local_multiquadcell);
}
void
Tucker2EI::FEMap::CellId2Cart(int id, int &x, int &y, int &z) const
{
  int num_ele_x = fe_.fe_x.GetNumberElements();
  int num_ele_y = fe_.fe_y.GetNumberElements();

  x = id % num_ele_x;
  y = id / num_ele_x % num_ele_y;
  z = id / (num_ele_x * num_ele_y);
}
void
Tucker2EI::FEMap::Cart2CellId(const int x,
                              const int y,
                              const int z,
                              int      &id) const
{
  int num_ele_x = fe_.fe_x.GetNumberElements();
  int num_ele_y = fe_.fe_y.GetNumberElements();

  id = x + y * num_ele_x + z * num_ele_x * num_ele_y;
}
int
Tucker2EI::FEMap::GetMpiRank() const
{
  return mpi_rank_;
}
int
Tucker2EI::FEMap::GetMpiSize() const
{
  return mpi_size_;
}
int
Tucker2EI::FEMap::GetNumLocalCells() const
{
  return num_local_cells_;
}
int
Tucker2EI::FEMap::GetNumTotalCells() const
{
  return num_total_cells_;
}
int
Tucker2EI::FEMap::GetNumLocalNodes() const
{
  return num_local_nodes_;
}
int
Tucker2EI::FEMap::GetNumTotalNodes() const
{
  return num_total_nodes_;
}
int
Tucker2EI::FEMap::GetNumLocalQuadPoints() const
{
  return num_local_quad_points_;
}
int
Tucker2EI::FEMap::GetNumTotalQuadPoints() const
{
  return num_total_quad_points_;
}
const std::vector<int> &
Tucker2EI::FEMap::GetCellMpiCnt() const
{
  return cell_mpi_cnt_;
}
const std::vector<int> &
Tucker2EI::FEMap::GetCellMpiDispl() const
{
  return cell_mpi_displ_;
}
const std::vector<int> &
Tucker2EI::FEMap::GetNodalMpiCnt() const
{
  return nodal_mpi_cnt_;
}
const std::vector<int> &
Tucker2EI::FEMap::GetNodalMpiDispl() const
{
  return nodal_mpi_displ_;
}
const std::vector<int> &
Tucker2EI::FEMap::GetQuadMpiCnt() const
{
  return quad_mpi_cnt_;
}
const std::vector<int> &
Tucker2EI::FEMap::GetQuadMpiDispl() const
{
  return quad_mpi_displ_;
}
const std::vector<double> &
Tucker2EI::FEMap::GetLocalCellJxw() const
{
  return local_cell_jxw_;
}
int
Tucker2EI::FEMap::GetNumTotalCellNodes() const
{
  return num_total_cell_nodes_;
}
