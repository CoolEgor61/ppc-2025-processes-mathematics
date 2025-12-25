#include "petrov_e_jarvis_algorithm/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <set>
#include <utility>
#include <vector>

#include "petrov_e_jarvis_algorithm/common/include/common.hpp"

namespace petrov_e_jarvis_algorithm {

static double CountOrientation(std::pair<double, double> p1, std::pair<double, double> p2, std::pair<double, double> p3) {
  return ((p2.first - p1.first) * (p3.second - p1.second)) - ((p2.second - p1.second) * (p3.first - p1.first));
}

static double CountDistance(std::pair<double, double> p1, std::pair<double, double> p2) {
  return ((p2.first - p1.first) * (p2.first - p1.first)) + ((p2.second - p1.second) * (p2.second - p1.second));
}

static int FindFirstPoint(std::vector<std::pair<double, double>> &points) {
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

static int FindNextPoint(std::vector<std::pair<double, double>>& points, int currentdotindex) {
  int nextdotindex = -1;
  int n = static_cast<int>(points.size());

  for (int k = 0; k < n; k++) {
    if (k == currentdotindex) {
      continue;
    }

    if (nextdotindex == -1) {
      nextdotindex = k;
      continue;
    }

    double orientation = CountOrientation(points[currentdotindex], points[nextdotindex], points[k]);

    if (orientation > 0) {
      nextdotindex = k;
    } else if (std::fabs(orientation) < 1e-10) {
      if (CountDistance(points[currentdotindex], points[k]) > CountDistance(points[currentdotindex], points[nextdotindex])){
        nextdotindex = k;
      }
    }
  }

  return nextdotindex;
}

PetrovEJarvisSEQ::PetrovEJarvisSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = {};
}

bool PetrovEJarvisSEQ::ValidationImpl() {
  return (GetInput().size() >= 3) && (GetOutput().empty());
}

bool PetrovEJarvisSEQ::PreProcessingImpl() {
  return (GetInput().size() >= 3) && (GetOutput().empty());
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

  int mindotindex = FindFirstPoint(input);

  int currentdotindex = mindotindex;
  int nextdotindex = 0;
  int flag1 = 1;

  while (flag1 != 0) {
    GetOutput().push_back(input[currentdotindex]);

    nextdotindex = FindNextPoint(input, currentdotindex);

    if (nextdotindex == mindotindex) {
      flag1 = 0;
    }
    currentdotindex = nextdotindex;
  }

  std::set<std::pair<double, double>> s1(GetOutput().begin(), GetOutput().end());
  GetOutput().assign(s1.begin(), s1.end());
  std::sort(GetOutput().begin(), GetOutput().end());

  return !GetOutput().empty();
}

bool PetrovEJarvisSEQ::PostProcessingImpl() {
  return !GetOutput().empty();
}

}  // namespace petrov_e_jarvis_algorithm
