#pragma once

#include <mpi.h>

#include "petrov_e_find_max_in_columns_matrix/common/include/common.hpp"
#include "task/include/task.hpp"

namespace petrov_e_find_max_in_columns_matrix {

template <typename MatrixElemType>
  MPI_Datatype GetMPIDatatype() {
    MPI_Datatype res;
    if (std::is_same_v<MatrixElemType, char>) {
      res = MPI_CHAR;
    } else if (std::is_same_v<MatrixElemType, unsigned char>) {
      res = MPI_UNSIGNED_CHAR;
    } else if (std::is_same_v<MatrixElemType, short>) {
      res = MPI_SHORT;
    } else if (std::is_same_v<MatrixElemType, unsigned short>) {
      res = MPI_UNSIGNED_SHORT;
    } else if (std::is_same_v<MatrixElemType, int>) {
      res = MPI_INT;
    } else if (std::is_same_v<MatrixElemType, unsigned>) {
      res = MPI_UNSIGNED;
    } else if (std::is_same_v<MatrixElemType, long>) {
      res = MPI_LONG;
    } else if (std::is_same_v<MatrixElemType, unsigned long>) {
      res = MPI_UNSIGNED_LONG;
    } else if (std::is_same_v<MatrixElemType, long long>) {
      res = MPI_LONG_LONG;
    } else if (std::is_same_v<MatrixElemType, unsigned long long>) {
      res = MPI_UNSIGNED_LONG_LONG;
    } else if (std::is_same_v<MatrixElemType, float>) {
      res = MPI_FLOAT;
    } else if (std::is_same_v<MatrixElemType, double>) {
      res = MPI_DOUBLE;
    } else if (std::is_same_v<MatrixElemType, long double>) {
      res = MPI_LONG_DOUBLE;
    } else if (std::is_same_v<MatrixElemType, bool>) {
      res = MPI_C_BOOL;
    } else res = MPI_DATATYPE_NULL;
    return res;
  }

class PetrovEFindMaxInColumnsMatrixMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit PetrovEFindMaxInColumnsMatrixMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

};

}  // namespace petrov_e_find_max_in_columns_matrix
