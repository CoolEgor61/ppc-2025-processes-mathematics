#pragma once

#include <tuple>
#include <utility>
#include <vector>

#include "task/include/task.hpp"

namespace petrov_e_jarvis_algorithm {

using InType = std::vector<std::pair<double, double>>;
using OutType = std::vector<std::pair<double, double>>;
using TestType = std::tuple<int, std::vector<std::pair<double, double>>, std::vector<std::pair<double, double>>>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace petrov_e_jarvis_algorithm
