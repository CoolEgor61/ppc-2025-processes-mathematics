#pragma once

#include <mpi.h>

#include <algorithm>
#include <cstdint>

#include "petrov_e_allreduce/common/include/common.hpp"
#include "task/include/task.hpp"

namespace petrov_e_allreduce {

inline void Operation(void *dest, void *src, int count, MPI_Datatype datatype, MPI_Op op) {
  // Ops: MPI_SUM, MPI_PROD, MPI_MAX, MPI_MIN
  if (op == MPI_SUM) {
    if (datatype == MPI_UNSIGNED_CHAR) {
      auto *d = static_cast<unsigned char *>(dest);
      auto *s = static_cast<unsigned char *>(src);
      for (int i = 0; i < count; i++) {
        d[i] += s[i];
      }
    } else if (datatype == MPI_SHORT) {
      auto *d = static_cast<int16_t *>(dest);
      auto *s = static_cast<int16_t *>(src);
      for (int i = 0; i < count; i++) {
        d[i] += s[i];
      }
    } else if (datatype == MPI_UNSIGNED_SHORT) {
      auto *d = static_cast<uint16_t *>(dest);
      auto *s = static_cast<uint16_t *>(src);
      for (int i = 0; i < count; i++) {
        d[i] += s[i];
      }
    } else if (datatype == MPI_INT) {
      auto *d = static_cast<int *>(dest);
      auto *s = static_cast<int *>(src);
      for (int i = 0; i < count; i++) {
        d[i] += s[i];
      }
    } else if (datatype == MPI_UNSIGNED) {
      auto *d = static_cast<unsigned *>(dest);
      auto *s = static_cast<unsigned *>(src);
      for (int i = 0; i < count; i++) {
        d[i] += s[i];
      }
    } else if (datatype == MPI_LONG) {
      auto *d = static_cast<int64_t *>(dest);
      auto *s = static_cast<int64_t *>(src);
      for (int i = 0; i < count; i++) {
        d[i] += s[i];
      }
    } else if (datatype == MPI_UNSIGNED_LONG) {
      auto *d = static_cast<uint64_t *>(dest);
      auto *s = static_cast<uint64_t *>(src);
      for (int i = 0; i < count; i++) {
        d[i] += s[i];
      }
    } else if (datatype == MPI_FLOAT) {
      auto *d = static_cast<float *>(dest);
      auto *s = static_cast<float *>(src);
      for (int i = 0; i < count; i++) {
        d[i] += s[i];
      }
    } else if (datatype == MPI_DOUBLE) {
      auto *d = static_cast<double *>(dest);
      auto *s = static_cast<double *>(src);
      for (int i = 0; i < count; i++) {
        d[i] += s[i];
      }
    } else if (datatype == MPI_LONG_DOUBLE) {
      auto *d = static_cast<long double *>(dest);
      auto *s = static_cast<long double *>(src);
      for (int i = 0; i < count; i++) {
        d[i] += s[i];
      }
    }
  } else if (op == MPI_PROD) {
    if (datatype == MPI_UNSIGNED_CHAR) {
      auto *d = static_cast<unsigned char *>(dest);
      auto *s = static_cast<unsigned char *>(src);
      for (int i = 0; i < count; i++) {
        d[i] *= s[i];
      }
    } else if (datatype == MPI_SHORT) {
      auto *d = static_cast<int16_t *>(dest);
      auto *s = static_cast<int16_t *>(src);
      for (int i = 0; i < count; i++) {
        d[i] *= s[i];
      }
    } else if (datatype == MPI_UNSIGNED_SHORT) {
      auto *d = static_cast<uint16_t *>(dest);
      auto *s = static_cast<uint16_t *>(src);
      for (int i = 0; i < count; i++) {
        d[i] *= s[i];
      }
    } else if (datatype == MPI_INT) {
      auto *d = static_cast<int *>(dest);
      auto *s = static_cast<int *>(src);
      for (int i = 0; i < count; i++) {
        d[i] *= s[i];
      }
    } else if (datatype == MPI_UNSIGNED) {
      auto *d = static_cast<unsigned *>(dest);
      auto *s = static_cast<unsigned *>(src);
      for (int i = 0; i < count; i++) {
        d[i] *= s[i];
      }
    } else if (datatype == MPI_LONG) {
      auto *d = static_cast<int64_t *>(dest);
      auto *s = static_cast<int64_t *>(src);
      for (int i = 0; i < count; i++) {
        d[i] *= s[i];
      }
    } else if (datatype == MPI_UNSIGNED_LONG) {
      auto *d = static_cast<uint64_t *>(dest);
      auto *s = static_cast<uint64_t *>(src);
      for (int i = 0; i < count; i++) {
        d[i] *= s[i];
      }
    } else if (datatype == MPI_FLOAT) {
      auto *d = static_cast<float *>(dest);
      auto *s = static_cast<float *>(src);
      for (int i = 0; i < count; i++) {
        d[i] *= s[i];
      }
    } else if (datatype == MPI_DOUBLE) {
      auto *d = static_cast<double *>(dest);
      auto *s = static_cast<double *>(src);
      for (int i = 0; i < count; i++) {
        d[i] *= s[i];
      }
    } else if (datatype == MPI_LONG_DOUBLE) {
      auto *d = static_cast<long double *>(dest);
      auto *s = static_cast<long double *>(src);
      for (int i = 0; i < count; i++) {
        d[i] *= s[i];
      }
    }
  } else if (op == MPI_MAX) {
    if (datatype == MPI_UNSIGNED_CHAR) {
      auto *d = static_cast<unsigned char *>(dest);
      auto *s = static_cast<unsigned char *>(src);
      for (int i = 0; i < count; i++) {
        d[i] = std::max(d[i], s[i]);
      }
    } else if (datatype == MPI_SHORT) {
      auto *d = static_cast<int16_t *>(dest);
      auto *s = static_cast<int16_t *>(src);
      for (int i = 0; i < count; i++) {
        d[i] = std::max(d[i], s[i]);
      }
    } else if (datatype == MPI_UNSIGNED_SHORT) {
      auto *d = static_cast<uint16_t *>(dest);
      auto *s = static_cast<uint16_t *>(src);
      for (int i = 0; i < count; i++) {
        d[i] = std::max(d[i], s[i]);
      }
    } else if (datatype == MPI_INT) {
      auto *d = static_cast<int *>(dest);
      auto *s = static_cast<int *>(src);
      for (int i = 0; i < count; i++) {
        d[i] = std::max(d[i], s[i]);
      }
    } else if (datatype == MPI_UNSIGNED) {
      auto *d = static_cast<unsigned *>(dest);
      auto *s = static_cast<unsigned *>(src);
      for (int i = 0; i < count; i++) {
        d[i] = std::max(d[i], s[i]);
      }
    } else if (datatype == MPI_LONG) {
      auto *d = static_cast<int64_t *>(dest);
      auto *s = static_cast<int64_t *>(src);
      for (int i = 0; i < count; i++) {
        d[i] = std::max(d[i], s[i]);
      }
    } else if (datatype == MPI_UNSIGNED_LONG) {
      auto *d = static_cast<uint64_t *>(dest);
      auto *s = static_cast<uint64_t *>(src);
      for (int i = 0; i < count; i++) {
        d[i] = std::max(d[i], s[i]);
      }
    } else if (datatype == MPI_FLOAT) {
      auto *d = static_cast<float *>(dest);
      auto *s = static_cast<float *>(src);
      for (int i = 0; i < count; i++) {
        d[i] = std::max(d[i], s[i]);
      }
    } else if (datatype == MPI_DOUBLE) {
      auto *d = static_cast<double *>(dest);
      auto *s = static_cast<double *>(src);
      for (int i = 0; i < count; i++) {
        d[i] = std::max(d[i], s[i]);
      }
    } else if (datatype == MPI_LONG_DOUBLE) {
      auto *d = static_cast<long double *>(dest);
      auto *s = static_cast<long double *>(src);
      for (int i = 0; i < count; i++) {
        d[i] = std::max(d[i], s[i]);
      }
    } else if (datatype == MPI_C_BOOL) {
      auto *d = static_cast<bool *>(dest);
      auto *s = static_cast<bool *>(src);
      for (int i = 0; i < count; i++) {
        d[i] = std::max(d[i], s[i]);
      }
    }
  } else if (op == MPI_MIN) {
    if (datatype == MPI_UNSIGNED_CHAR) {
      auto *d = static_cast<unsigned char *>(dest);
      auto *s = static_cast<unsigned char *>(src);
      for (int i = 0; i < count; i++) {
        d[i] = std::min(d[i], s[i]);
      }
    } else if (datatype == MPI_SHORT) {
      auto *d = static_cast<int16_t *>(dest);
      auto *s = static_cast<int16_t *>(src);
      for (int i = 0; i < count; i++) {
        d[i] = std::min(d[i], s[i]);
      }
    } else if (datatype == MPI_UNSIGNED_SHORT) {
      auto *d = static_cast<uint16_t *>(dest);
      auto *s = static_cast<uint16_t *>(src);
      for (int i = 0; i < count; i++) {
        d[i] = std::min(d[i], s[i]);
      }
    } else if (datatype == MPI_INT) {
      auto *d = static_cast<int *>(dest);
      auto *s = static_cast<int *>(src);
      for (int i = 0; i < count; i++) {
        d[i] = std::min(d[i], s[i]);
      }
    } else if (datatype == MPI_UNSIGNED) {
      auto *d = static_cast<unsigned *>(dest);
      auto *s = static_cast<unsigned *>(src);
      for (int i = 0; i < count; i++) {
        d[i] = std::min(d[i], s[i]);
      }
    } else if (datatype == MPI_LONG) {
      auto *d = static_cast<int64_t *>(dest);
      auto *s = static_cast<int64_t *>(src);
      for (int i = 0; i < count; i++) {
        d[i] = std::min(d[i], s[i]);
      }
    } else if (datatype == MPI_UNSIGNED_LONG) {
      auto *d = static_cast<uint64_t *>(dest);
      auto *s = static_cast<uint64_t *>(src);
      for (int i = 0; i < count; i++) {
        d[i] = std::min(d[i], s[i]);
      }
    } else if (datatype == MPI_FLOAT) {
      auto *d = static_cast<float *>(dest);
      auto *s = static_cast<float *>(src);
      for (int i = 0; i < count; i++) {
        d[i] = std::min(d[i], s[i]);
      }
    } else if (datatype == MPI_DOUBLE) {
      auto *d = static_cast<double *>(dest);
      auto *s = static_cast<double *>(src);
      for (int i = 0; i < count; i++) {
        d[i] = std::min(d[i], s[i]);
      }
    } else if (datatype == MPI_LONG_DOUBLE) {
      auto *d = static_cast<long double *>(dest);
      auto *s = static_cast<long double *>(src);
      for (int i = 0; i < count; i++) {
        d[i] = std::min(d[i], s[i]);
      }
    } else if (datatype == MPI_C_BOOL) {
      auto *d = static_cast<bool *>(dest);
      auto *s = static_cast<bool *>(src);
      for (int i = 0; i < count; i++) {
        d[i] = std::min(d[i], s[i]);
      }
    }
  }
}

inline void GetSizeOf2(MPI_Datatype type, int &size) {
  int res = 4;
  if (type == MPI_CHAR) {
    res = sizeof(char);
  } else if (type == MPI_UNSIGNED_CHAR) {
    res = sizeof(unsigned char);
  } else if (type == MPI_SHORT) {
    res = sizeof(int16_t);
  } else if (type == MPI_UNSIGNED_SHORT) {
    res = sizeof(uint16_t);
  } else if (type == MPI_INT) {
    res = sizeof(int);
  } else if (type == MPI_UNSIGNED) {
    res = sizeof(unsigned);
  } else if (type == MPI_LONG) {
    res = sizeof(int64_t);
  } else if (type == MPI_UNSIGNED_LONG) {
    res = sizeof(uint64_t);
  } else if (type == MPI_FLOAT) {
    res = sizeof(float);
  } else if (type == MPI_DOUBLE) {
    res = sizeof(double);
  } else if (type == MPI_LONG_DOUBLE) {
    res = sizeof(long double);
  } else if (type == MPI_C_BOOL) {
    res = sizeof(bool);
  }
  size = res;
}

template <typename MatrixElemType>
MPI_Datatype GetMPIDatatype() {
  MPI_Datatype res = MPI_DATATYPE_NULL;
  if (std::is_same_v<MatrixElemType, char>) {
    res = MPI_CHAR;
  } else if (std::is_same_v<MatrixElemType, unsigned char>) {
    res = MPI_UNSIGNED_CHAR;
  } else if (std::is_same_v<MatrixElemType, int16_t>) {
    res = MPI_SHORT;
  } else if (std::is_same_v<MatrixElemType, uint16_t>) {
    res = MPI_UNSIGNED_SHORT;
  } else if (std::is_same_v<MatrixElemType, int>) {
    res = MPI_INT;
  } else if (std::is_same_v<MatrixElemType, unsigned>) {
    res = MPI_UNSIGNED;
  } else if (std::is_same_v<MatrixElemType, int64_t>) {
    res = MPI_LONG;
  } else if (std::is_same_v<MatrixElemType, uint64_t>) {
    res = MPI_UNSIGNED_LONG;
  } else if (std::is_same_v<MatrixElemType, float>) {
    res = MPI_FLOAT;
  } else if (std::is_same_v<MatrixElemType, double>) {
    res = MPI_DOUBLE;
  } else if (std::is_same_v<MatrixElemType, long double>) {
    res = MPI_LONG_DOUBLE;
  } else if (std::is_same_v<MatrixElemType, bool>) {
    res = MPI_C_BOOL;
  }
  return res;
}

inline int MpiMyAllreduce(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op,
                           MPI_Comm comm) {
  int proc_num = 0;
  int proc_rank = 0;
  MPI_Comm_size(comm, &proc_num);
  MPI_Comm_rank(comm, &proc_rank);
  int type_size = 0;
  GetSizeOf2(datatype, type_size);
  int data_size = count * type_size;
  MPI_Status status;
  memcpy(recvbuf, sendbuf, data_size);
  void *tempbuf = malloc(data_size);

  int parent = (proc_rank - 1) / 2;
  int left = (2 * proc_rank) + 1;
  int right = (2 * proc_rank) + 2;

  if (left < proc_num) {
    MPI_Recv(tempbuf, count, datatype, left, 0, comm, &status);
    Operation(recvbuf, tempbuf, count, datatype, op);
  }

  if (right < proc_num) {
    MPI_Recv(tempbuf, count, datatype, right, 0, comm, &status);
    Operation(recvbuf, tempbuf, count, datatype, op);
  }

  if (proc_rank != 0) {
    MPI_Send(recvbuf, count, datatype, parent, 0, comm);
    MPI_Recv(recvbuf, count, datatype, parent, 1, comm, &status);
  }

  if (left < proc_num) {
    MPI_Send(recvbuf, count, datatype, left, 1, comm);
  }

  if (right < proc_num) {
    MPI_Send(recvbuf, count, datatype, right, 1, comm);
  }

  free(tempbuf);
  return MPI_SUCCESS;
}

class PetrovEMyAllreduceMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit PetrovEMyAllreduceMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace petrov_e_allreduce
