#include "petrov_e_jarvis_algorithm/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <set>
#include <utility>
#include <vector>

#include "petrov_e_jarvis_algorithm/common/include/common.hpp"

namespace petrov_e_jarvis_algorithm {

PetrovEJarvisMPI::PetrovEJarvisMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = {};
}

bool PetrovEJarvisMPI::ValidationImpl() {
  return (GetInput().size() >= 3) && (GetOutput().empty());
}

bool PetrovEJarvisMPI::PreProcessingImpl() {
  return (GetInput().size() >= 3) && (GetOutput().empty());
}

bool PetrovEJarvisMPI::RunImpl() {
  if (GetInput().size() < 3) {
    return false;
  }

  int proc_num = 0;
  int proc_rank = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &proc_num);
  MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);

  std::set<std::pair<double, double>> s(GetInput().begin(), GetInput().end());
  GetInput().assign(s.begin(), s.end());
  std::sort(GetInput().begin(), GetInput().end());

  GetOutput().clear();

  auto &input = GetInput();
  int n = static_cast<int>(GetInput().size());

  int col_num_per_proc = n / proc_num;
  int col_num_wo_proc = n % proc_num;
  int flag = 0;

  std::vector<int> start(proc_num);
  std::vector<int> end(proc_num);

  int proc_start = 0;
  int proc_end = 0;

  if (proc_rank == 0) {
    for (auto i = 0; i < proc_num; i++) {
      if (i < col_num_wo_proc) {
        flag = 1;
      } else {
        flag = 0;
      }
      start[i] = (i * col_num_per_proc) + std::min(i, col_num_wo_proc);
      end[i] = start[i] + col_num_per_proc + flag;
    }
  }

  MPI_Scatter(start.data(), 1, MPI_INT, &proc_start, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Scatter(end.data(), 1, MPI_INT, &proc_end, 1, MPI_INT, 0, MPI_COMM_WORLD);

  int local_points = proc_end - proc_start;

  int mindotindex = 0;
  if (proc_rank == 0) {
    for (auto i = 0; i < n; i++) {
      if (GetInput()[i].second < GetInput()[mindotindex].second ||
          (GetInput()[i].second == GetInput()[mindotindex].second &&
           GetInput()[i].first < GetInput()[mindotindex].first)) {
        mindotindex = i;
      }
    }
  }
  MPI_Bcast(&mindotindex, 1, MPI_INT, 0, MPI_COMM_WORLD);

  int currentdotindex = mindotindex;
  int nextdotindex = 0;
  std::vector<int> proc_points(proc_num);

  do {
    if (proc_rank == 0) {
      GetOutput().push_back(GetInput()[currentdotindex]);
    }

    MPI_Bcast(&currentdotindex, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int localnextdotindex = -1;

    if (local_points > 0) {
      for (int k = proc_start; k < proc_end; k++) {
        if (k == currentdotindex) {
          continue;
        }

        if (localnextdotindex == -1) {
          localnextdotindex = k;
          continue;
        }

        double x1 = input[k].first - input[currentdotindex].first;
        double y1 = input[k].second - input[currentdotindex].second;
        double x2 = input[localnextdotindex].first - input[currentdotindex].first;
        double y2 = input[localnextdotindex].second - input[currentdotindex].second;

        double orientation = (x1 * y2) - (y1 * x2);

        if (orientation > 0) {
          localnextdotindex = k;
        } else if (std::fabs(orientation) < 1e-10 || std::fabs(orientation) == 0) {
          double dist1 = (x1 * x1) + (y1 * y1);
          double dist2 = (x2 * x2) + (y2 * y2);
          if (dist1 > dist2) {
            localnextdotindex = k;
          }
        }
      }
    }

    MPI_Gather(&localnextdotindex, 1, MPI_INT, proc_points.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (proc_rank == 0) {
      nextdotindex = -1;

      for (auto j = 0; j < proc_num; j++) {
        int candidatedot = proc_points[j];
        if (candidatedot == -1) {
          continue;
        }

        if (nextdotindex == -1) {
          nextdotindex = candidatedot;
          continue;
        }

        if (candidatedot == currentdotindex) {
          continue;
        }

        double x1 = input[candidatedot].first - input[currentdotindex].first;
        double y1 = input[candidatedot].second - input[currentdotindex].second;
        double x2 = input[nextdotindex].first - input[currentdotindex].first;
        double y2 = input[nextdotindex].second - input[currentdotindex].second;

        double orientation = (x1 * y2) - (y1 * x2);

        if (orientation > 0) {
          nextdotindex = candidatedot;
        } else if (std::fabs(orientation) < 1e-10 || std::fabs(orientation) == 0) {
          double dist1 = (x1 * x1) + (y1 * y1);
          double dist2 = (x2 * x2) + (y2 * y2);
          if (dist1 > dist2) {
            nextdotindex = candidatedot;
          }
        }
      }
    }

    MPI_Bcast(&nextdotindex, 1, MPI_INT, 0, MPI_COMM_WORLD);

    currentdotindex = nextdotindex;

  } while (currentdotindex != mindotindex);

  int buffsize = 0;
  int buffsize2 = 0;
  std::vector<double> buffer;

  if (proc_rank == 0) {
    std::set<std::pair<double, double>> s1(GetOutput().begin(), GetOutput().end());
    GetOutput().assign(s1.begin(), s1.end());
    std::sort(GetOutput().begin(), GetOutput().end());

    buffsize = static_cast<int>(GetOutput().size());
    buffsize2 = 2 * static_cast<int>(GetOutput().size());

    buffer.resize(buffsize2);
    for (int i = 0; i < buffsize; i++) {
      buffer[2 * i] = GetOutput()[i].first;
      buffer[(2 * i) + 1] = GetOutput()[i].second;
    }
  }

  MPI_Bcast(&buffsize, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (proc_rank != 0) {
    buffer.resize(buffsize2);
    GetOutput().clear();
    GetOutput().resize(buffsize);
  }

  MPI_Bcast(buffer.data(), buffsize2, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  if (proc_rank != 0) {
    for (int i = 0; i < buffsize; i++) {
      GetOutput()[i] = {buffer[2 * i], buffer[(2 * i) + 1]};
    }
  }

  return !GetOutput().empty();
}

bool PetrovEJarvisMPI::PostProcessingImpl() {
  return !GetOutput().empty();
}

}  // namespace petrov_e_jarvis_algorithm
