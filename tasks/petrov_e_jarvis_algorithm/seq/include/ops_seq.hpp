#pragma once

#include "petrov_e_jarvis_algorithm/common/include/common.hpp"
#include "task/include/task.hpp"

namespace petrov_e_jarvis_algorithm {

class PetrovEJarvisSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit PetrovEJarvisSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace petrov_e_jarvis_algorithm
