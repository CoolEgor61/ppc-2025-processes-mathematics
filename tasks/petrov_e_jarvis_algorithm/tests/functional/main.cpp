#include <gtest/gtest.h>

#include <algorithm>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "petrov_e_jarvis_algorithm/common/include/common.hpp"
#include "petrov_e_jarvis_algorithm/mpi/include/ops_mpi.hpp"
#include "petrov_e_jarvis_algorithm/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace petrov_e_jarvis_algorithm {

class PetrovERunFuncTestsJarvis : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return "set_of_points_" + std::to_string(std::get<0>(test_param));
  }

 protected:
  void SetUp() override {
    params_ = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    input_data_ = std::get<1>(params_);
    output_vector_ = std::get<2>(params_);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return output_data == output_vector_;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  TestType params_;
  InType input_data_;
  OutType output_vector_;
};

namespace {

TEST_P(PetrovERunFuncTestsJarvis, JarvisAlgorithm) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 7> kTestParam = {
    std::make_tuple(
        1, std::vector<std::pair<double, double>>{{0., 3.}, {2., 2.}, {1., 1.}, {2., 1.}, {3., 0.}, {0., 0.}, {3., 3.}},
        std::vector<std::pair<double, double>>{{0., 0.}, {0., 3.}, {3., 0.}, {3., 3.}}),
    std::make_tuple(2, std::vector<std::pair<double, double>>{{0., 3.}, {2., 2.}, {1., 1.}},
                    std::vector<std::pair<double, double>>{{0., 3.}, {1., 1.}, {2., 2.}}),
    std::make_tuple(3, std::vector<std::pair<double, double>>{{0., 0.}, {1., 1.}, {2., 2.}, {3., 3.}, {4., 4.}},
                    std::vector<std::pair<double, double>>{{0., 0.}, {4., 4.}}),
    std::make_tuple(4, std::vector<std::pair<double, double>>{{0., 0.}, {1., 0.}, {2., 1.}, {1., 2.}, {0., 1.}},
                    std::vector<std::pair<double, double>>{{0., 0.}, {0., 1.}, {1., 0.}, {1., 2.}, {2., 1.}}),
    std::make_tuple(5, std::vector<std::pair<double, double>>{{0., 0.}, {2., 0.}, {2., 2.}, {0., 2.}, {1., 1.}},
                    std::vector<std::pair<double, double>>{{0., 0.}, {0., 2.}, {2., 0.}, {2., 2.}}),
    std::make_tuple(6,
                    std::vector<std::pair<double, double>>{
                        {-2., -2.}, {-2., 2.}, {2., 2.}, {2., -2.}, {0., 0.}, {-1., 1.}, {1., -1.}},
                    std::vector<std::pair<double, double>>{{-2., -2.}, {-2., 2.}, {2., -2.}, {2., 2.}}),
    std::make_tuple(
        7,
        std::vector<std::pair<double, double>>{
            {0., 0.}, {1., 0.}, {2., 0.}, {3., 0.}, {0., 1.}, {1., 1.}, {2., 1.}, {3., 1.}, {0., 2.}, {3., 2.}},
        std::vector<std::pair<double, double>>{{0., 0.}, {0., 2.}, {3., 0.}, {3., 2.}})};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<PetrovEJarvisMPI, InType>(kTestParam, PPC_SETTINGS_petrov_e_jarvis_algorithm),
    ppc::util::AddFuncTask<PetrovEJarvisSEQ, InType>(kTestParam, PPC_SETTINGS_petrov_e_jarvis_algorithm));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = PetrovERunFuncTestsJarvis::PrintFuncTestName<PetrovERunFuncTestsJarvis>;

INSTANTIATE_TEST_SUITE_P(JarvisAlgorithm, PetrovERunFuncTestsJarvis, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace petrov_e_jarvis_algorithm
