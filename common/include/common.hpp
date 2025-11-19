#pragma once

#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace petrov_e_find_max_in_columns_matrix {

using InType = std::tuple<size_t, size_t, std::vector<double>>;
using OutType = std::vector<double>;
using TestType = std::tuple<size_t, size_t, std::vector<double>, std::vector<double>>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace petrov_e_find_max_in_columns_matrix
