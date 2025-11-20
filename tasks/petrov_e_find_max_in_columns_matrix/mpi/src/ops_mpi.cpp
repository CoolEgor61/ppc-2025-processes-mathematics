#include "petrov_e_find_max_in_columns_matrix/mpi/include/ops_mpi.hpp"

#include <numeric>
#include <vector>

#include "petrov_e_find_max_in_columns_matrix/common/include/common.hpp"
#include "util/include/util.hpp"

namespace petrov_e_find_max_in_columns_matrix {

PetrovEFindMaxInColumnsMatrixMPI::PetrovEFindMaxInColumnsMatrixMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = {};
}

bool PetrovEFindMaxInColumnsMatrixMPI::ValidationImpl() {
  return (std::get<0>(GetInput()) * std::get<1>(GetInput()) == (int)std::get<2>(GetInput()).size()) && (GetOutput().empty());
}

bool PetrovEFindMaxInColumnsMatrixMPI::PreProcessingImpl() {
  return (std::get<0>(GetInput()) * std::get<1>(GetInput()) == (int)std::get<2>(GetInput()).size());
}

bool PetrovEFindMaxInColumnsMatrixMPI::RunImpl() {
  if ((std::get<0>(GetInput()) * std::get<1>(GetInput()) != (int)std::get<2>(GetInput()).size())) {
    return false;
  }

  auto &n = std::get<0>(GetInput());
  auto &m = std::get<1>(GetInput());
  auto &matrix = std::get<2>(GetInput());
  OutType &res = GetOutput();
  using MatrixElemType = std::remove_reference_t<decltype(matrix[0])>;
  MPI_Datatype MPIMatrixElemType = petrov_e_find_max_in_columns_matrix::getMPIDatatype<MatrixElemType>();
  if (MPIMatrixElemType == MPI_DATATYPE_NULL) {
    return false;
  }
  OutType ProcRes;
  int ProcNum, ProcRank;
  MPI_Comm_size(MPI_COMM_WORLD, &ProcNum);
  MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);
  int i, j;
  MatrixElemType max;
  res.resize(m);
  int ColNumPerProc = m / ProcNum;
  int ColNumWOProc = m % ProcNum;
  int CurrentDisplsSend = 0;
  int CurrentDisplsRecv = 0;
  std::vector<int> sendcounts, recvcounts;
  std::vector<int> displssend, displsrecv;
  sendcounts.resize(ProcNum);
  displssend.resize(ProcNum);
  recvcounts.resize(ProcNum);
  displsrecv.resize(ProcNum);
  int flag;

  for (i = 0; i < ProcNum; i++) {
    if (i < ColNumWOProc) {
      flag = 1;
    } else {
      flag = 0;
    }
    sendcounts[i] = n * (ColNumPerProc + flag);
    recvcounts[i] = ColNumPerProc + flag;
    displssend[i] = CurrentDisplsSend;
    CurrentDisplsSend += sendcounts[i];
    displsrecv[i] = CurrentDisplsRecv;
    CurrentDisplsRecv += recvcounts[i];
  }
  int ProcSize;
  if (ProcRank < ColNumWOProc) {
    ProcSize = ColNumPerProc+1;
  } else {
    ProcSize = ColNumPerProc;
  }
  std::vector<MatrixElemType> ProcData;
  int nProcSize = n * ProcSize;
  ProcData.resize(nProcSize);

  MPI_Scatterv(matrix.data(), sendcounts.data(), displssend.data(), MPIMatrixElemType, ProcData.data(), nProcSize, MPIMatrixElemType, 0, MPI_COMM_WORLD);
  ProcRes.resize(ProcSize);

  for (i = 0; i < ProcSize; i++) {
    max = ProcData[i * n];
    for (j = 1; j < n; j++) {
      max = std::max(ProcData[i * n + j], max);
    }
    ProcRes[i]=max;
  }

  MPI_Gatherv(ProcRes.data(), ProcSize, MPIMatrixElemType, res.data(), recvcounts.data(), displsrecv.data(), MPIMatrixElemType, 0, MPI_COMM_WORLD);

  return true;
}

bool PetrovEFindMaxInColumnsMatrixMPI::PostProcessingImpl() {
  return !GetOutput().empty() + GetOutput().empty();
}

}  // namespace petrov_e_find_max_in_columns_matrix
