#include "petrov_e_jarvis_algorithm/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <set>
#include <utility>
#include <vector>

#include "petrov_e_jarvis_algorithm/common/include/common.hpp"
#include "util/include/util.hpp"

namespace petrov_e_jarvis_algorithm {

PetrovEJarvisSEQ::PetrovEJarvisSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = {};
}

bool PetrovEJarvisSEQ::ValidationImpl() {
  return (GetInput().size() >= 3) && (GetOutput().size() == 0);
}

bool PetrovEJarvisSEQ::PreProcessingImpl() {
  return (GetInput().size() >= 3) && (GetOutput().size() == 0);
}

bool PetrovEJarvisSEQ::RunImpl() {
  if (GetInput().size() < 3) {
    return false;
  }

  std::set<std::pair<double, double>> s(GetInput().begin(), GetInput().end());
  GetInput().assign(s.begin(), s.end());
  std::sort(GetInput().begin(), GetInput().end());

  GetOutput().clear();

  auto &input = GetInput();
  int n = static_cast<int>(input.size());

  int mindotindex = 0;
  for (auto i = 0; i < n; i++) {
    if (input[i].second < input[mindotindex].second ||
        (input[i].second == input[mindotindex].second && input[i].first < input[mindotindex].first)) {
      mindotindex = i;
    }
  }

  int currentdotindex = mindotindex;
  int nextdotindex = 0;

  do {
    GetOutput().push_back(input[currentdotindex]);

    nextdotindex = -1;

    for (int k = 0; k < n; k++) {
      if (k == currentdotindex) {
        continue;
      }

      if (nextdotindex == -1) {
        nextdotindex = k;
        continue;
      }

      double x1 = input[k].first - input[currentdotindex].first;
      double y1 = input[k].second - input[currentdotindex].second;
      double x2 = input[nextdotindex].first - input[currentdotindex].first;
      double y2 = input[nextdotindex].second - input[currentdotindex].second;

      double orientation = x1 * y2 - y1 * x2;

      if (orientation > 0) {
        nextdotindex = k;
      } else if (std::fabs(orientation) < 1e-10 || std::fabs(orientation) == 0) {
        double dist1 = x1 * x1 + y1 * y1;
        double dist2 = x2 * x2 + y2 * y2;
        if (dist1 > dist2) {
          nextdotindex = k;
        }
      }
    }

    currentdotindex = nextdotindex;

  } while (currentdotindex != mindotindex);

  std::set<std::pair<double, double>> s1(GetOutput().begin(), GetOutput().end());
  GetOutput().assign(s1.begin(), s1.end());
  std::sort(GetOutput().begin(), GetOutput().end());

  return GetOutput().size() > 0;
}

bool PetrovEJarvisSEQ::PostProcessingImpl() {
  return GetOutput().size() > 0;
}

}  // namespace petrov_e_jarvis_algorithm
