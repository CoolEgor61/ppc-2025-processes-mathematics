#include "petrov_e_find_max_in_columns_matrix/mpi/include/ops_mpi.hpp"

#include <algorithm>
#include <mpi.h>
#include <type_traits>
#include <vector>

#include "petrov_e_find_max_in_columns_matrix/common/include/common.hpp"

namespace petrov_e_find_max_in_columns_matrix {

PetrovEFindMaxInColumnsMatrixMPI::PetrovEFindMaxInColumnsMatrixMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = {};
}

bool PetrovEFindMaxInColumnsMatrixMPI::ValidationImpl() {
  return (std::get<0>(GetInput()) * std::get<1>(GetInput()) == static_cast<int>(std::get<2>(GetInput()).size())) && (GetOutput().empty());
}

bool PetrovEFindMaxInColumnsMatrixMPI::PreProcessingImpl() {
  return (std::get<0>(GetInput()) * std::get<1>(GetInput()) == static_cast<int>(std::get<2>(GetInput()).size()));
}

bool PetrovEFindMaxInColumnsMatrixMPI::RunImpl() {
  if ((std::get<0>(GetInput()) * std::get<1>(GetInput()) != static_cast<int>(std::get<2>(GetInput()).size()))) {
    return false;
  }

  auto &n = std::get<0>(GetInput());
  auto &m = std::get<1>(GetInput());
  auto &matrix = std::get<2>(GetInput());
  OutType &res = GetOutput();
  using MatrixElemType = std::remove_reference_t<decltype(matrix[0])>;
  MPI_Datatype mpi_matrix_elem_type = petrov_e_find_max_in_columns_matrix::GetMPIDatatype<MatrixElemType>();
  if (mpi_matrix_elem_type == MPI_DATATYPE_NULL) {
    return false;
  }
  OutType proc_res;
  int ProcNum = 0;
  int ProcRank = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &ProcNum);
  MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);
  int i = 0;
  int j = 0;
  MatrixElemType max = NAN;
  res.resize(m);
  int col_num_per_proc = m / ProcNum;
  int col_num_wo_proc = m % ProcNum;
  int current_displs_send = 0;
  int current_displs_recv = 0;
  std::vector<int> sendcounts;
  std::vector<int> recvcounts;
  std::vector<int> displssend;
  std::vector<int> displsrecv;
  sendcounts.resize(ProcNum);
  displssend.resize(ProcNum);
  recvcounts.resize(ProcNum);
  displsrecv.resize(ProcNum);
  int flag = 0;

  for (i = 0; i < ProcNum; i++) {
    if (i < col_num_wo_proc) {
      flag = 1;
    } else {
      flag = 0;
    }
    sendcounts[i] = n * (col_num_per_proc + flag);
    recvcounts[i] = col_num_per_proc + flag;
    displssend[i] = current_displs_send;
    current_displs_send += sendcounts[i];
    displsrecv[i] = current_displs_recv;
    current_displs_recv += recvcounts[i];
  }
  int proc_size = 0;
  if (ProcRank < col_num_wo_proc) {
    proc_size = col_num_per_proc+1;
  } else {
    proc_size = col_num_per_proc;
  }
  std::vector<MatrixElemType> proc_data;
  int n_proc_size = n * proc_size;
  proc_data.resize(n_proc_size);

  MPI_Scatterv(matrix.data(), sendcounts.data(), displssend.data(), mpi_matrix_elem_type, proc_data.data(), n_proc_size, mpi_matrix_elem_type, 0, MPI_COMM_WORLD);
  proc_res.resize(proc_size);

  for (i = 0; std::cmp_less(i, proc_size); i++) {
    max = proc_data[i * n];
    for (j = 1; j < n; j++) {
      max = std::max(proc_data[(i * n) + j], max);
    }
    proc_res[i]=max;
  }

  MPI_Gatherv(proc_res.data(), proc_size, mpi_matrix_elem_type, res.data(), recvcounts.data(), displsrecv.data(), mpi_matrix_elem_type, 0, MPI_COMM_WORLD);

  return true;
}

bool PetrovEFindMaxInColumnsMatrixMPI::PostProcessingImpl() {
  return true;
}

}  // namespace petrov_e_find_max_in_columns_matrix
