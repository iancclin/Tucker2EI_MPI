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

#include "FourIndexGenerator.h"
#include "DataReader.h"
#include "BlasWrapper.h"

Tucker2EI::FourIndexGenerator::FourIndexGenerator(
  const FEMap               &fe_map,
  const FEMap               &conv_fe_map,
  const ConvolutionComputer &conv_comp,
  double                     decomp_tol)
  : Integrator(fe_map.GetFe())
  , fe_map_(fe_map)
  , conv_fe_map_(conv_fe_map)
  , comm(fe_map_.GetComm())
  , conv_comp_(conv_comp)
  , decomp_tol_(decomp_tol)
  , local_jxw_(fe_map_.GetLocalCellJxw())
{}

void
Tucker2EI::FourIndexGenerator::Calculate(const std::vector<int> &input_idx,
                                         std::vector<double> &unique_four_idx_int)
{
  const int num_total_wfns  = input_idx.size();
  const int num_local_cells = fe_map_.GetNumLocalCells();
  const int num_local_nodes = fe_map_.GetNumLocalNodes();
  const int num_total_nodes = fe_map_.GetNumTotalNodes(); // total 3d points
  const int num_total_cell_nodes     = fe_map_.GetNumTotalCellNodes();
  const int num_local_quad_points    = fe_map_.GetNumLocalQuadPoints();
  const int num_total_quad_points    = fe_map_.GetNumTotalQuadPoints();
  const int num_nodes_per_cell       = fe_map_.fe_.GetNumNodesPer3Dele();
  const int num_quad_points_per_cell = fe_map_.fe_.GetNumQuadPointsPer3Dele();
  const int mpi_size                 = fe_map_.GetMpiSize();
  const int mpi_rank                 = fe_map_.GetMpiRank();
  const std::vector<int>    &mpi_node_cnt   = fe_map_.GetNodalMpiCnt();
  const std::vector<int>    &mpi_node_displ = fe_map_.GetNodalMpiDispl();
  const std::vector<int>    &mpi_quad_cnt   = fe_map_.GetQuadMpiCnt();
  const std::vector<int>    &mpi_quad_displ = fe_map_.GetQuadMpiDispl();
  const std::vector<double> &cell_shape_functions_at_quad_points =
    fe_map_.fe_.GetElementalShapeFunctionAtQuadPoints();

  std::vector<std::vector<double>> local_quad_wfns(
    num_total_wfns, std::vector<double>(num_local_quad_points, 0.0));
  {
    std::vector<double> local_nodal_wfns(num_total_wfns * num_local_nodes, 0);
    std::vector<int>    read_in_wfn_cnt(mpi_size, 0);
    std::vector<int>    read_in_wfn_displ(mpi_size + 1, 0);
    for (int i = 0; i < mpi_size; ++i)
    {
      read_in_wfn_cnt[i] =
        num_total_wfns / mpi_size + (i < num_total_wfns % mpi_size ? 1 : 0);
      read_in_wfn_displ[i + 1] = read_in_wfn_displ[i] + read_in_wfn_cnt[i];
    }

    int num_local_wfns = read_in_wfn_cnt[mpi_rank];
    // read in wavefunctions
    std::vector<std::vector<double>> read_in_buffer(mpi_size);
    for (int i_rank = 0; i_rank < mpi_size; ++i_rank)
    {
      read_in_buffer[i_rank].assign(num_local_wfns * mpi_node_cnt[i_rank], 0.0);
    }

    for (int i = read_in_wfn_displ[mpi_rank], ii = 0;
         i < read_in_wfn_displ[mpi_rank + 1];
         ++i, ++ii)
    {
      int             wfn_i = input_idx[i];
      Tucker::Tensor *wfn;
      ReadWfn(wfn_i, wfn);
      std::vector<double> cell_wfn;
      std::vector<double> wfn_cell =
        std::move(fe_map_.fe_.Convert3DMeshToCellwiseVector(wfn));
      Tucker::MemoryManager::safe_delete(wfn);
      for (int i_rank = 0; i_rank < mpi_size; ++i_rank)
      {
        const auto begin = wfn_cell.begin() + mpi_node_displ[i_rank];
        const auto end   = wfn_cell.begin() + mpi_node_displ[i_rank + 1];
        std::copy(begin,
                  end,
                  read_in_buffer[i_rank].begin() + ii * mpi_node_cnt[i_rank]);
      }
    }

    std::vector<double> send_buffer(num_local_wfns * num_total_cell_nodes, 0.0);
    std::vector<int>    send_cnt(mpi_size, 0), send_displ(mpi_size + 1, 0);
    auto                iter = send_buffer.begin();
    for (int i_rank = 0; i_rank < mpi_size; ++i_rank)
    {
      iter = std::copy(read_in_buffer[i_rank].begin(),
                       read_in_buffer[i_rank].end(),
                       iter);

      send_cnt[i_rank]       = read_in_buffer[i_rank].size();
      send_displ[i_rank + 1] = send_displ[i_rank] + send_cnt[i_rank];
    }
    std::vector<std::vector<double>>().swap(read_in_buffer);

    std::vector<int> recv_cnt(mpi_size, 0.0), recv_displ(mpi_size + 1, 0.0);
    for (int i_rank = 0; i_rank < mpi_size; ++i_rank)
    {
      recv_cnt[i_rank]       = read_in_wfn_cnt[i_rank] * num_local_nodes;
      recv_displ[i_rank + 1] = recv_displ[i_rank] + recv_cnt[i_rank];
    }

    MPI_Alltoallv(send_buffer.data(),
                  send_cnt.data(),
                  send_displ.data(),
                  MPI_DOUBLE,
                  local_nodal_wfns.data(),
                  recv_cnt.data(),
                  recv_displ.data(),
                  MPI_DOUBLE,
                  MPI_COMM_WORLD);
    std::vector<double>().swap(send_buffer);
    std::vector<double> local_quad_wfns_temp(num_total_wfns *
                                               num_local_quad_points,
                                             0);
    blas_wrapper::Dgemm(num_quad_points_per_cell,
                        num_nodes_per_cell,
                        num_total_wfns * num_local_cells,
                        cell_shape_functions_at_quad_points.data(),
                        local_nodal_wfns.data(),
                        local_quad_wfns_temp.data());
    for (int i = 0; i < num_total_wfns; ++i)
    {
      const auto begin =
        local_quad_wfns_temp.begin() + i * num_local_quad_points;
      std::copy(begin,
                begin + num_local_quad_points,
                local_quad_wfns[i].begin());
    }
  }

  // start to compute the convolution integral sequentially with batch
  const int num_convs_per_group_per_task = 20; // after each mpi task
                                              // computed 10 convolution
                                              // integrals, communicate
                                              // once
  const int num_wfns        = input_idx.size();
  const int num_total_convs = num_wfns * (num_wfns + 1) / 2;

  const int num_convs_per_group =
    num_convs_per_group_per_task * mpi_size >= num_total_convs ?
      num_total_convs :
      num_convs_per_group_per_task * mpi_size;

  const int num_unique_four_idx_int =
    num_total_convs * (num_total_convs + 1) / 2;

  std::vector<std::pair<int, int>> unique_conv_idx(num_total_convs);
  for (int i = 0, cnt = 0; i < num_wfns; ++i)
  {
    for (int j = i; j < num_wfns; ++j)
    {
      unique_conv_idx[cnt].first  = i;
      unique_conv_idx[cnt].second = j;
      cnt++;
    }
  }

  unique_four_idx_int.assign(num_unique_four_idx_int, 0.0);

  for (int i_conv = 0, fidx_cnt = 0; i_conv < num_total_convs;
       i_conv += num_convs_per_group)
  {
    int conv_start = i_conv, conv_end = i_conv + num_convs_per_group;
    if (conv_end >= num_total_convs)
    {
      conv_end = num_total_convs;
    }
    int num_convs_this_group = conv_end - conv_start;


    // distribute computation load to processors
    std::vector<int> num_convs_cnt(mpi_size, 0),
      num_convs_displ(mpi_size + 1, conv_start);

    for (int i_rank = 0; i_rank < mpi_size; ++i_rank)
    {
      num_convs_cnt[i_rank] =
        num_convs_this_group / mpi_size +
        (i_rank < num_convs_this_group % mpi_size ? 1 : 0);
      num_convs_displ[i_rank + 1] =
        num_convs_displ[i_rank] + num_convs_cnt[i_rank];
    }

    const int num_conv_this_task   = num_convs_cnt[mpi_rank];
    const int conv_start_this_task = num_convs_displ[mpi_rank];
    const int conv_end_this_task   = num_convs_displ[mpi_rank + 1];


    std::vector<std::vector<double>> conv_nodal(mpi_size);
    for (int i_rank = 0; i_rank < mpi_size; ++i_rank)
    {
      conv_nodal[i_rank] =
        std::vector<double>(num_conv_this_task * mpi_node_cnt[i_rank], 0.0);
    }


    for (int i_conv_this_task = conv_start_this_task, iter = 0;
         i_conv_this_task < conv_end_this_task;
         ++i_conv_this_task, ++iter)
    {
      int wfn_id_i = input_idx[unique_conv_idx[i_conv_this_task].first];
      int wfn_id_j = input_idx[unique_conv_idx[i_conv_this_task].second];

      std::vector<double> conv =
        std::move(ComputeConvolutionNodalCell(wfn_id_i, wfn_id_j));

      for (int i_rank = 0; i_rank < mpi_size; ++i_rank)
      {
        std::copy(conv.begin() + mpi_node_displ[i_rank],
                  conv.begin() + mpi_node_displ[i_rank + 1],
                  conv_nodal[i_rank].begin() + iter * mpi_node_cnt[i_rank]);
      }
    }

    std::vector<double> send_buffer(num_conv_this_task * num_total_cell_nodes);
    auto                send_buffer_iter = send_buffer.begin();
    std::vector<int>    send_cnt(mpi_size, 0), send_displ(mpi_size + 1, 0);
    std::vector<int>    recv_cnt(mpi_size, 0), recv_displ(mpi_size + 1, 0);
    for (int i_rank = 0; i_rank < mpi_size; ++i_rank)
    {
      send_cnt[i_rank]       = conv_nodal[i_rank].size();
      send_displ[i_rank + 1] = send_displ[i_rank] + send_cnt[i_rank];
      send_buffer_iter       = std::copy(conv_nodal[i_rank].begin(),
                                   conv_nodal[i_rank].end(),
                                   send_buffer_iter);

      recv_cnt[i_rank]       = num_convs_cnt[i_rank] * num_local_nodes;
      recv_displ[i_rank + 1] = recv_displ[i_rank] + recv_cnt[i_rank];
    }
    std::vector<std::vector<double>>().swap(conv_nodal);

    std::vector<double> recv_buffer(num_convs_this_group * num_local_nodes,
                                    0.0);
    MPI_Alltoallv(send_buffer.data(),
                  send_cnt.data(),
                  send_displ.data(),
                  MPI_DOUBLE,
                  recv_buffer.data(),
                  recv_cnt.data(),
                  recv_displ.data(),
                  MPI_DOUBLE,
                  MPI_COMM_WORLD);
    std::vector<double>().swap(send_buffer);

    std::vector<double> conv_quad_local(num_convs_this_group *
                                          num_local_quad_points,
                                        0.0);
    blas_wrapper::Dgemm(num_quad_points_per_cell,
                        num_nodes_per_cell,
                        num_convs_this_group * num_local_cells,
                        cell_shape_functions_at_quad_points.data(),
                        recv_buffer.data(),
                        conv_quad_local.data());
    std::vector<double>().swap(recv_buffer);


    for (int i_conv_this_group = conv_start, iter = 0;
         i_conv_this_group < conv_end;
         ++i_conv_this_group, ++iter)
    {
      double *conv_quad_ii =
        conv_quad_local.data() + iter * num_local_quad_points;
      for (int jj = i_conv_this_group; jj < num_total_convs; ++jj)
      {
        const std::vector<double> &wfn_i =
          local_quad_wfns[unique_conv_idx[jj].first];
        const std::vector<double> &wfn_j =
          local_quad_wfns[unique_conv_idx[jj].second];
        double fidx_int = 0.0;
        for (int i_quad = 0; i_quad < num_local_quad_points; ++i_quad)
        {
          fidx_int += local_jxw_[i_quad] * wfn_i[i_quad] * wfn_j[i_quad] *
                      conv_quad_ii[i_quad];
        }

        unique_four_idx_int[fidx_cnt] = fidx_int;
        fidx_cnt++;
      }
    }
  }

  MPI_Allreduce(MPI_IN_PLACE,
                unique_four_idx_int.data(),
                num_unique_four_idx_int,
                MPI_DOUBLE,
                MPI_SUM,
                MPI_COMM_WORLD);

}
std::vector<double>
Tucker2EI::FourIndexGenerator::ComputeConvolutionNodalCell(int k, int l)
{
  // read in and compute wfn_k * wfn_l on nodal points
  Tucker::Tensor *wfn;
  ReadWfn(k, wfn);
  Tucker::Tensor *wfn_prod;
  ReadWfn(l, wfn_prod);
  double *wfn_prod_ptr = wfn_prod->data();
  double *wfn_ptr      = wfn->data();
  for (int i = 0; i < wfn_prod->getNumElements(); ++i)
  {
    wfn_prod_ptr[i] *= wfn_ptr[i];
  }
  Tucker::MemoryManager::safe_delete(wfn);

  conv_comp_.ComputeQuadConvolutionFromNodeOnNodeInplace(fe_map_.fe_.fe,
                                                         conv_fe_map_.fe_.fe,
                                                         decomp_tol_,
                                                         wfn_prod);

  std::vector<double> conv_kl =
    std::move(fe_map_.fe_.Convert3DMeshToCellwiseVector(wfn_prod));

  Tucker::MemoryManager::safe_delete(wfn_prod);

  return conv_kl;
}
// const std::vector<double> &
// Tucker2EI::FourIndexGenerator::GetResults() const
//{
//   return four_idx_;
// }