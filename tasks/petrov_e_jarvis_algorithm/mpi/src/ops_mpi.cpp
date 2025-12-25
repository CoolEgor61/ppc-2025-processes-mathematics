#include "petrov_e_jarvis_algorithm/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <set>
#include <utility>
#include <vector>

#include "petrov_e_jarvis_algorithm/common/include/common.hpp"

namespace {

double CountOrientation(std::pair<double, double> p1, std::pair<double, double> p2, std::pair<double, double> p3) {
  return ((p2.first - p1.first) * (p3.second - p1.second)) - ((p2.second - p1.second) * (p3.first - p1.first));
}

double CountDistance(std::pair<double, double> p1, std::pair<double, double> p2) {
  return ((p2.first - p1.first) * (p2.first - p1.first)) + ((p2.second - p1.second) * (p2.second - p1.second));
}

int FindFirstPoint(std::vector<std::pair<double, double>> &points) {
  int mindotindex = 0;
  int n = static_cast<int>(points.size());
  for (int i = 1; i < n; i++) {
    if (points[i].second < points[mindotindex].second ||
        (points[i].second == points[mindotindex].second && points[i].first < points[mindotindex].first)) {
      mindotindex = i;
    }
  }
  return mindotindex;
}

int FindLocal(std::vector<std::pair<double, double>> &points, int start, int end, int currentdotindex) {
  int localnextdotindex = -1;

  for (int k = start; k < end; k++) {
    if (k == currentdotindex) {
      continue;
    }

    if (localnextdotindex == -1) {
      localnextdotindex = k;
      continue;
    }

    double orientation = CountOrientation(points[currentdotindex], points[localnextdotindex], points[k]);

    if (orientation > 0) {
      localnextdotindex = k;
    } else if (std::fabs(orientation) < 1e-10) {
      if (CountDistance(points[currentdotindex], points[k]) >
          CountDistance(points[currentdotindex], points[localnextdotindex])) {
        localnextdotindex = k;
      }
    }
  }

  return localnextdotindex;
}

int FindNext(std::vector<std::pair<double, double>> &points, std::vector<int> &proc_points, int currentdotindex) {
  int nextdotindex = -1;

  for (int j = 0; j < proc_points.size(); j++) {
    int candidatedot = proc_points[j];
    if (candidatedot == -1 || candidatedot == currentdotindex) {
      continue;
    }

    if (nextdotindex == -1) {
      nextdotindex = candidatedot;
      continue;
    }

    double orientation = CountOrientation(points[currentdotindex], points[nextdotindex], points[candidatedot]);

    if (orientation > 0) {
      nextdotindex = candidatedot;
    } else if (std::fabs(orientation) < 1e-10) {
      if (CountDistance(points[currentdotindex], points[candidatedot]) >
          CountDistance(points[currentdotindex], points[nextdotindex])) {
        nextdotindex = candidatedot;
      }
    }
  }

  return nextdotindex;
}

}  // namespace

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

  int mindotindex = 0;
  if (proc_rank == 0) {
    mindotindex = FindFirstPoint(input);
  }
  MPI_Bcast(&mindotindex, 1, MPI_INT, 0, MPI_COMM_WORLD);

  int currentdotindex = mindotindex;
  int nextdotindex = 0;
  std::vector<int> proc_points(proc_num);
  int flag1 = 1;

  while (flag1 != 0) {
    if (proc_rank == 0) {
      GetOutput().push_back(GetInput()[currentdotindex]);
    }

    MPI_Bcast(&currentdotindex, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int localnextdotindex = FindLocal(input, proc_start, proc_end, currentdotindex);

    MPI_Gather(&localnextdotindex, 1, MPI_INT, proc_points.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (proc_rank == 0) {
      nextdotindex = FindNext(input, proc_points, currentdotindex);
    }

    MPI_Bcast(&nextdotindex, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (nextdotindex == mindotindex) {
      flag1 = 0;
    }

    currentdotindex = nextdotindex;
  }
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
      buffer[static_cast<int64_t>(2) * i] = GetOutput()[i].first;
      buffer[(2 * i) + 1] = GetOutput()[i].second;
    }
  }

  MPI_Bcast(&buffsize, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&buffsize2, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (proc_rank != 0) {
    buffer.resize(buffsize2);
    GetOutput().clear();
    GetOutput().resize(buffsize);
  }

  MPI_Bcast(buffer.data(), buffsize2, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  if (proc_rank != 0) {
    for (int i = 0; i < buffsize; i++) {
      GetOutput()[i] = {buffer[static_cast<int64_t>(2) * i], buffer[(2 * i) + 1]};
    }
  }

  return !GetOutput().empty();
}

bool PetrovEJarvisMPI::PostProcessingImpl() {
  return !GetOutput().empty();
}

}  // namespace petrov_e_jarvis_algorithm
