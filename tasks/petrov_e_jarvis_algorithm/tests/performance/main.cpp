#include <gtest/gtest.h>

#include "petrov_e_jarvis_algorithm/common/include/common.hpp"
#include "petrov_e_jarvis_algorithm/mpi/include/ops_mpi.hpp"
#include "petrov_e_jarvis_algorithm/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace petrov_e_jarvis_algorithm {

class PetrovERunPerfTestJarvis : public ppc::util::BaseRunPerfTests<InType, OutType> {
  InType input_data_{};
  OutType output_data_{};

  void SetUp() override {
    std::string abs_path1 = ppc::util::GetAbsoluteTaskPath(PPC_ID_petrov_e_jarvis_algorithm, "perf_test.txt");
    std::string abs_path2 = ppc::util::GetAbsoluteTaskPath(PPC_ID_petrov_e_jarvis_algorithm, "ans.txt");
    std::ifstream in1(abs_path1);
    std::ifstream in2(abs_path2);
    std::vector<std::pair<double, double>> indata;
    std::vector<std::pair<double, double>> ans;
    double x = 0., y = 0.;
    if (in1.is_open()) {
      while (in1 >> x >> y) {
        indata.push_back({x, y});
      }
      input_data_ = indata;
      in1.close();
    }
    if (in2.is_open()) {
      while (in2 >> x >> y) {
        ans.push_back({x, y});
      }
      output_data_ = ans;
      in2.close();
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (std::cmp_not_equal(static_cast<int>(output_data_.size()), static_cast<int>(output_data.size()))) {
      return false;
    }
    std::set<std::pair<double, double>> expected(output_data_.begin(), output_data_.end());
    std::set<std::pair<double, double>> actual(output_data.begin(), output_data.end());

    return expected == actual;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(PetrovERunPerfTestJarvis, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, PetrovEJarvisMPI, PetrovEJarvisSEQ>(PPC_SETTINGS_petrov_e_jarvis_algorithm);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = PetrovERunPerfTestJarvis::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, PetrovERunPerfTestJarvis, kGtestValues, kPerfTestName);

}  // namespace petrov_e_jarvis_algorithm
