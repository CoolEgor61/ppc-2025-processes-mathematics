#pragma once

#include <mpi.h>

#include "petrov_e_find_max_in_columns_matrix/common/include/common.hpp"
#include "task/include/task.hpp"

namespace petrov_e_find_max_in_columns_matrix {

template <typename MatrixElemType>
  MPI_Datatype getMPIDatatype() {
    if (std::is_same<MatrixElemType, char>::value) {
      return MPI_CHAR;
    } else if (std::is_same<MatrixElemType, unsigned char>::value) {
      return MPI_UNSIGNED_CHAR;
    } else if (std::is_same<MatrixElemType, short>::value) {
      return MPI_SHORT;
    } else if (std::is_same<MatrixElemType, unsigned short>::value) {
      return MPI_UNSIGNED_SHORT;
    } else if (std::is_same<MatrixElemType, int>::value) {
      return MPI_INT;
    } else if (std::is_same<MatrixElemType, unsigned>::value) {
      return MPI_UNSIGNED;
    } else if (std::is_same<MatrixElemType, long>::value) {
      return MPI_LONG;
    } else if (std::is_same<MatrixElemType, unsigned long>::value) {
      return MPI_UNSIGNED_LONG;
    } else if (std::is_same<MatrixElemType, long long>::value) {
      return MPI_LONG_LONG;
    } else if (std::is_same<MatrixElemType, unsigned long long>::value) {
      return MPI_UNSIGNED_LONG_LONG;
    } else if (std::is_same<MatrixElemType, float>::value) {
      return MPI_FLOAT;
    } else if (std::is_same<MatrixElemType, double>::value) {
      return MPI_DOUBLE;
    } else if (std::is_same<MatrixElemType, long double>::value) {
      return MPI_LONG_DOUBLE;
    } else if (std::is_same<MatrixElemType, bool>::value) {
      return MPI_C_BOOL;
    } else return MPI_DATATYPE_NULL;
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
