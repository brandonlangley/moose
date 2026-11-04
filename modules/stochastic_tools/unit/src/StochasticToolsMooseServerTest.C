//* This file is part of the MOOSE framework
//* https://mooseframework.inl.gov
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#include "MooseServer.h"
#include "Moose.h"
#include "FEProblemBase.h"
#include "Sampler.h"
#include "QuadratureSampler.h"
#include "wasplsp/LSP.h"
#include "waspplot/CustomPlotSerialization.h"
#include "gtest/gtest.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>

class StochasticToolsMooseServerTest : public ::testing::Test
{
protected:
  void SetUp() override;
  void TearDown() override;
  bool check_plotting_input(const std::string & doc_text_open);
  bool check_plotting_request(const std::string & input_marker);
  std::shared_ptr<wasp::CustomPlot> get_plotting_object(const std::string & plot_title) const;
  std::string write_input_file(const std::string & file_name, const std::string & file_text);
  static std::string build_plotting_input(const std::string & input_objects);

  std::shared_ptr<MooseApp> moose_unit_app;
  std::unique_ptr<MooseServer> moose_server;
  wasp::DataArray plotting_responses;
  std::filesystem::path test_input_dir;
  std::string test_input_text;
  std::string test_input_uri;
};

void
StochasticToolsMooseServerTest::SetUp()
{
  // create module app and server before preparing input files for plotting
  moose_unit_app = Moose::createMooseApp("StochasticToolsApp", 0, nullptr);
  moose_server = std::make_unique<MooseServer>(*moose_unit_app);
  test_input_dir = std::filesystem::temp_directory_path() /
                   ("stochastic_input_plot_" +
                    std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
  ASSERT_TRUE(std::filesystem::create_directory(test_input_dir));
  test_input_uri = wasp::lsp::m_uri_prefix + (test_input_dir / "input.i").string();

  // build initialize request to enable the existing plot extension support
  wasp::DataObject client_caps, initialize_request, initialize_response;
  client_caps[wasp::lsp::m_extensions] = wasp::DataObject();
  client_caps[wasp::lsp::m_extensions]["plotting"] = true;
  std::stringstream errors;
  ASSERT_TRUE(
      wasp::lsp::buildInitializeRequest(initialize_request, errors, 1, -1, "", client_caps));
  ASSERT_TRUE(moose_server->handleInitializeRequest(initialize_request, initialize_response));

  // build initialized notification and handle it using the new plot server
  wasp::DataObject initialized_notification;
  ASSERT_TRUE(wasp::lsp::buildInitializedNotification(initialized_notification, errors));
  ASSERT_TRUE(moose_server->handleInitializedNotification(initialized_notification));
}

void
StochasticToolsMooseServerTest::TearDown()
{
  // release input objects before removing their associated temporary files
  moose_server.reset();
  moose_unit_app.reset();
  std::error_code error;
  std::filesystem::remove_all(test_input_dir, error);
}

std::string
StochasticToolsMooseServerTest::write_input_file(const std::string & file_name,
                                                 const std::string & file_text)
{
  // write the configured input companion and return its absolute file path
  const auto file_path = test_input_dir / file_name;
  std::ofstream input_file(file_path);
  input_file << file_text;
  input_file.close();
  if (!input_file)
    throw std::runtime_error("Could not write input plotting fixture");
  return file_path.string();
}

std::string
StochasticToolsMooseServerTest::build_plotting_input(const std::string & input_objects)
{
  // supply a minimal problem around the objects used for each plot request
  return R"INPUT(
[Mesh]
  type = GeneratedMesh
  dim = 1
[]
[Problem]
  solve = false
[]
)INPUT" + input_objects +
         R"INPUT(
[Executioner]
  type = Steady
[]
)INPUT";
}

bool
StochasticToolsMooseServerTest::check_plotting_input(const std::string & doc_text_open)
{
  // open the input using its file location so companions resolve correctly
  test_input_text = doc_text_open;
  write_input_file("input.i", doc_text_open);
  wasp::DataObject didopen_notification, diagnostics_notification;
  std::stringstream errors;
  if (!wasp::lsp::buildDidOpenNotification(
          didopen_notification, errors, test_input_uri, "moose", 1, test_input_text) ||
      !moose_server->handleDidOpenNotification(didopen_notification, diagnostics_notification))
    return false;

  // dissect the diagnostics and confirm the input has no validation errors
  std::string response_uri;
  wasp::DataArray diagnostics_array;
  return wasp::lsp::dissectPublishDiagnosticsNotification(
             diagnostics_notification, errors, response_uri, diagnostics_array) &&
         response_uri == test_input_uri && diagnostics_array.empty();
}

bool
StochasticToolsMooseServerTest::check_plotting_request(const std::string & input_marker)
{
  // find the marked input line and clear plots before requesting new plots
  plotting_responses.resize(0);
  const auto marker_offset = test_input_text.find(input_marker);
  if (marker_offset == std::string::npos)
    throw std::runtime_error("Missing input plotting cursor marker");
  const int request_line =
      std::count(test_input_text.begin(), test_input_text.begin() + marker_offset, '\n');
  wasp::DataObject ext_request, ext_response;
  std::stringstream errors;
  if (!wasp::lsp::buildExtensionRequest(
          ext_request, errors, "plotting", 2, test_input_uri, request_line, 4) ||
      !moose_server->handleExtensionRequest("plotting", ext_request, ext_response))
    return false;

  // dissect the extension response and verify its request identifier value
  int response_id;
  return wasp::lsp::dissectExtensionResponse(
             ext_response, errors, response_id, plotting_responses) &&
         response_id == 2;
}

std::shared_ptr<wasp::CustomPlot>
StochasticToolsMooseServerTest::get_plotting_object(const std::string & plot_title) const
{
  // find the requested plot by title within the extension response objects
  for (const auto & response_object : plotting_responses)
  {
    auto plot_object = wasp::deserializeCustomPlot(*response_object.to_object());
    if (plot_object->title().text().find(plot_title) != std::string::npos)
      return plot_object;
  }
  return nullptr;
}

TEST_F(StochasticToolsMooseServerTest, PlottingSamplerInputPairs)
{
  // prepare input objects used by the specified plotting extension request
  ASSERT_TRUE(check_plotting_input(build_plotting_input(R"INPUT(
[Samplers]
  [design]
    type = InputMatrix
    matrix = '3 30; 1 10; 2 20'
  []
[]
)INPUT")));
  ASSERT_TRUE(check_plotting_request("type = InputMatrix"));
  const auto design_plot = get_plotting_object("input design:");
  ASSERT_TRUE(design_plot);
  EXPECT_EQ((std::vector<double>{0, 1, 2}), design_plot->series()[0]->keys());
  EXPECT_EQ((std::vector<double>{3, 1, 2}), design_plot->series()[0]->values());
  const auto pairs_plot = get_plotting_object("paired columns");
  ASSERT_TRUE(pairs_plot);
  EXPECT_EQ((std::vector<double>{3, 1, 2}), pairs_plot->series()[0]->keys());
  EXPECT_EQ((std::vector<double>{30, 10, 20}), pairs_plot->series()[0]->values());
  EXPECT_EQ(
      wasp::CustomPlot::lsNone,
      std::static_pointer_cast<wasp::CustomPlot::Graph>(pairs_plot->series()[0])->lineStyle());
}

TEST_F(StochasticToolsMooseServerTest, PlottingSamplerInputRows)
{
  // prepare input objects used by the specified plotting extension request
  ASSERT_TRUE(check_plotting_input(build_plotting_input(R"INPUT(
[Samplers]
  [design]
    type = CartesianProduct
    linear_space_items = '0 1 4001'
  []
[]
)INPUT")));
  ASSERT_TRUE(check_plotting_request("type = CartesianProduct"));
  const auto design_plot = get_plotting_object("input design:");
  ASSERT_TRUE(design_plot);
  ASSERT_EQ(2000u, design_plot->series()[0]->keys().size());
  EXPECT_EQ(0, design_plot->series()[0]->values().front());
  EXPECT_EQ(4000, design_plot->series()[0]->values().back());
  EXPECT_NE(std::string::npos, design_plot->title().text().find("2000/4001 rows"));
  ASSERT_TRUE(check_plotting_request("type = CartesianProduct"));
  EXPECT_EQ(design_plot->series()[0]->values(),
            get_plotting_object("input design:")->series()[0]->values());
}

TEST_F(StochasticToolsMooseServerTest, PlottingSamplerInputColumns)
{
  // prepare input objects used by the specified plotting extension request
  const auto input_csv_path = write_input_file("design.csv", "a,b,c\n1,10,100\n2,20,200\n");
  ASSERT_TRUE(check_plotting_input(build_plotting_input(R"INPUT(
[Samplers]
  [design]
    type = CSVSampler
    column_names = 'c a'
    samples_file = ')INPUT" + input_csv_path + R"INPUT('
  []
[]
)INPUT")));
  ASSERT_TRUE(check_plotting_request("type = CSVSampler"));
  const auto pairs_plot = get_plotting_object("paired columns");
  ASSERT_TRUE(pairs_plot);
  EXPECT_EQ((std::vector<double>{100, 200}), pairs_plot->series()[0]->keys());
  EXPECT_EQ((std::vector<double>{1, 2}), pairs_plot->series()[0]->values());
}

TEST_F(StochasticToolsMooseServerTest, PlottingSamplerInputVectors)
{
  // prepare input objects used by the specified plotting extension request
  ASSERT_TRUE(check_plotting_input(build_plotting_input(R"INPUT(
[Reporters]
  [data]
    type = ConstantReporter
    real_vector_names = 'x y'
    real_vector_values = '1 2 3; 10 20 30'
  []
[]
[Samplers]
  [design]
    type = VectorPostprocessorSampler
    vectors_names = 'data/x data/y'
  []
[]
)INPUT")));
  ASSERT_TRUE(check_plotting_request("type = VectorPostprocessorSampler"));
  const auto pairs_plot = get_plotting_object("paired columns");
  ASSERT_TRUE(pairs_plot);
  EXPECT_EQ((std::vector<double>{1, 2, 3}), pairs_plot->series()[0]->keys());
  EXPECT_EQ((std::vector<double>{10, 20, 30}), pairs_plot->series()[0]->values());
  auto & sampler =
      moose_server->getCheckApp().actionWarehouse().problemBase()->getSampler("design");
  // check previews leave the sampler row counts unchanged before execution
  EXPECT_EQ(1, sampler.getNumberOfRows());
}

TEST_F(StochasticToolsMooseServerTest, PlottingSamplerRandomDraws)
{
  // prepare input objects used by the specified plotting extension request
  std::string dist_names;
  for (const auto i : make_range(4000))
    dist_names += i ? " u" : "u";
  ASSERT_TRUE(check_plotting_input(build_plotting_input(R"INPUT(
[Distributions]
  [u]
    type = Uniform
    lower_bound = 0
    upper_bound = 1
  []
[]
[Samplers]
  [design]
    type = MonteCarlo
    num_rows = 5000
    distributions = ')INPUT" + dist_names + R"INPUT('
  []
[]
)INPUT")));
  ASSERT_TRUE(check_plotting_request("type = MonteCarlo"));
  const auto design_plot = get_plotting_object("input design:");
  ASSERT_TRUE(design_plot);
  ASSERT_EQ(4u, design_plot->series().size());
  ASSERT_EQ(50u, design_plot->series()[0]->keys().size());
  EXPECT_EQ(49, design_plot->series()[0]->keys().back());
  auto & sampler =
      moose_server->getCheckApp().actionWarehouse().problemBase()->getSampler("design");
  EXPECT_DOUBLE_EQ(sampler.getSample(49, 3), design_plot->series()[3]->values().back());
  EXPECT_NE(std::string::npos, design_plot->title().text().find("first rows"));
}

TEST_F(StochasticToolsMooseServerTest, PlottingSamplerLatinHypercube)
{
  // prepare input objects used by the specified plotting extension request
  ASSERT_TRUE(check_plotting_input(build_plotting_input(R"INPUT(
[Distributions]
  [u]
    type = Uniform
    lower_bound = 0
    upper_bound = 1
  []
[]
[Samplers]
  [design]
    type = LatinHypercube
    num_rows = 16
    distributions = 'u u'
  []
[]
)INPUT")));
  ASSERT_TRUE(check_plotting_request("type = LatinHypercube"));
  const auto design_plot = get_plotting_object("input design:");
  ASSERT_TRUE(design_plot);
  for (const auto & graph_series : design_plot->series())
  {
    std::vector<unsigned int> bins;
    for (const auto value : graph_series->values())
      bins.push_back(std::floor(16 * value));
    std::sort(bins.begin(), bins.end());
    for (const auto i : index_range(bins))
      EXPECT_EQ(i, bins[i]);
  }
  auto & problem = *moose_server->getCheckApp().actionWarehouse().problemBase();
  problem.executeSamplers(EXEC_INITIAL);
  auto & sampler = problem.getSampler("design");
  for (const auto col : index_range(design_plot->series()))
    for (const auto row : index_range(design_plot->series()[col]->values()))
      EXPECT_DOUBLE_EQ(design_plot->series()[col]->values()[row], sampler.getSample(row, col));
}

TEST_F(StochasticToolsMooseServerTest, PlottingSamplerNestedMonteCarlo)
{
  // prepare input objects used by the specified plotting extension request
  ASSERT_TRUE(check_plotting_input(build_plotting_input(R"INPUT(
[Distributions]
  [u]
    type = Uniform
    lower_bound = 0
    upper_bound = 1
  []
[]
[Samplers]
  [design]
    type = NestedMonteCarlo
    num_rows = '2 3'
    distributions = 'u; u'
  []
[]
)INPUT")));
  ASSERT_TRUE(check_plotting_request("type = NestedMonteCarlo"));
  const auto design_plot = get_plotting_object("input design:");
  ASSERT_TRUE(design_plot);
  const auto & outer_values = design_plot->series()[0]->values();
  ASSERT_EQ(6u, outer_values.size());
  EXPECT_EQ(outer_values[0], outer_values[1]);
  EXPECT_EQ(outer_values[1], outer_values[2]);
  EXPECT_EQ(outer_values[3], outer_values[4]);
  EXPECT_EQ(outer_values[4], outer_values[5]);
}

TEST_F(StochasticToolsMooseServerTest, PlottingSamplerMorris)
{
  // prepare input objects used by the specified plotting extension request
  ASSERT_TRUE(check_plotting_input(build_plotting_input(R"INPUT(
[Distributions]
  [u]
    type = Uniform
    lower_bound = 0
    upper_bound = 1
  []
[]
[Samplers]
  [design]
    type = MorrisSampler
    distributions = 'u u'
    trajectories = 10
  []
[]
)INPUT")));
  ASSERT_TRUE(check_plotting_request("type = MorrisSampler"));
  const auto paths_plot = get_plotting_object("Morris input trajectories");
  ASSERT_TRUE(paths_plot);
  ASSERT_EQ(16u, paths_plot->series().size());
  auto & sampler =
      moose_server->getCheckApp().actionWarehouse().problemBase()->getSampler("design");
  for (const auto i : index_range(paths_plot->series()))
  {
    const auto & graph_series = paths_plot->series()[i];
    ASSERT_EQ(3u, graph_series->keys().size());
    for (const auto j : make_range(3))
    {
      EXPECT_DOUBLE_EQ(j, graph_series->keys()[j]);
      EXPECT_DOUBLE_EQ(sampler.getSample(3 * (i / 2) + j, i % 2), graph_series->values()[j]);
    }
  }
}

TEST_F(StochasticToolsMooseServerTest, PlottingSamplerSobol)
{
  // prepare input objects used by the specified plotting extension request
  ASSERT_TRUE(check_plotting_input(build_plotting_input(R"INPUT(
[Samplers]
  [a]
    type = InputMatrix
    matrix = '10 20; 11 21'
  []
  [b]
    type = InputMatrix
    matrix = '100 200; 101 201'
  []
  [design]
    type = Sobol
    sampler_a = a
    sampler_b = b
    resample = false
  []
[]
)INPUT")));
  ASSERT_TRUE(check_plotting_request("type = Sobol"));
  const auto blocks_plot = get_plotting_object("Sobol input matrix blocks");
  ASSERT_TRUE(blocks_plot);
  ASSERT_EQ(2u, blocks_plot->series().size());
  EXPECT_EQ((std::vector<double>{100, 10, 100, 10}), blocks_plot->series()[0]->values());
  EXPECT_EQ((std::vector<double>{200, 200, 20, 20}), blocks_plot->series()[1]->values());
  EXPECT_EQ((std::vector<double>{0, 1, 2, 3}), blocks_plot->series()[0]->keys());
}

TEST_F(StochasticToolsMooseServerTest, PlottingSamplerSobolResampling)
{
  // prepare input objects used by the specified plotting extension request
  ASSERT_TRUE(check_plotting_input(build_plotting_input(R"INPUT(
[Samplers]
  [a]
    type = InputMatrix
    matrix = '10 20'
  []
  [b]
    type = InputMatrix
    matrix = '100 200'
  []
  [design]
    type = Sobol
    sampler_a = a
    sampler_b = b
    resample = true
  []
[]
)INPUT")));
  ASSERT_TRUE(check_plotting_request("type = Sobol"));
  const auto blocks_plot = get_plotting_object("Sobol input matrix blocks");
  ASSERT_TRUE(blocks_plot);
  EXPECT_EQ((std::vector<double>{100, 10, 100, 100, 10, 10}), blocks_plot->series()[0]->values());
  EXPECT_EQ((std::vector<double>{200, 200, 20, 20, 200, 20}), blocks_plot->series()[1]->values());
}

TEST_F(StochasticToolsMooseServerTest, PlottingSamplerQuadratureWeights)
{
  // prepare input objects used by the specified plotting extension request
  ASSERT_TRUE(check_plotting_input(build_plotting_input(R"INPUT(
[Distributions]
  [u]
    type = Uniform
    lower_bound = -1
    upper_bound = 1
  []
[]
[Samplers]
  [design]
    type = Quadrature
    distributions = 'u u'
    order = 3
    sparse_grid = smolyak
  []
[]
)INPUT")));
  ASSERT_TRUE(check_plotting_request("type = Quadrature"));
  const auto weights_plot = get_plotting_object("quadrature weights (unnormalized)");
  ASSERT_TRUE(weights_plot);
  auto & sampler = dynamic_cast<QuadratureSampler &>(
      moose_server->getCheckApp().actionWarehouse().problemBase()->getSampler("design"));
  const auto & observation_values = weights_plot->series()[0]->values();
  ASSERT_EQ(sampler.getNumberOfRows(), observation_values.size());
  EXPECT_TRUE(std::any_of(observation_values.begin(),
                          observation_values.end(),
                          [](Real graph_keys) { return graph_keys < 0; }));
  for (const auto row : index_range(observation_values))
    EXPECT_DOUBLE_EQ(sampler.getQuadratureWeight(row), observation_values[row]);
  EXPECT_LT(weights_plot->y1Axis().rangeMin(), 0);
}

TEST_F(StochasticToolsMooseServerTest, PlottingSamplerPolynomialBases)
{
  // prepare input objects used by the specified plotting extension request
  ASSERT_TRUE(check_plotting_input(build_plotting_input(R"INPUT(
[Distributions]
  [u]
    type = Uniform
    lower_bound = -1
    upper_bound = 1
  []
[]
[Samplers]
  [design]
    type = Quadrature
    distributions = u
    order = 2
  []
[]
)INPUT")));
  ASSERT_TRUE(check_plotting_request("type = Quadrature"));
  const auto basis_plot = get_plotting_object("polynomial basis: u");
  ASSERT_TRUE(basis_plot);
  ASSERT_EQ(3u, basis_plot->series().size());
  const auto & graph_keys = basis_plot->series()[0]->keys();
  ASSERT_EQ(201u, graph_keys.size());
  for (const auto i : index_range(graph_keys))
  {
    EXPECT_DOUBLE_EQ(1, basis_plot->series()[0]->values()[i]);
    EXPECT_NEAR(graph_keys[i], basis_plot->series()[1]->values()[i], 1e-12);
    // check quadratic basis results with tolerance for polynomial rounding
    EXPECT_NEAR(
        (3 * graph_keys[i] * graph_keys[i] - 1) / 2, basis_plot->series()[2]->values()[i], 1e-12);
  }
}

TEST_F(StochasticToolsMooseServerTest, PlottingSamplerMarginals)
{
  // prepare input objects used by the specified plotting extension request
  ASSERT_TRUE(check_plotting_input(build_plotting_input(R"INPUT(
[Samplers]
  [design]
    type = InputMatrix
    matrix = '1; 1; 2; 4'
  []
[]
)INPUT")));
  ASSERT_TRUE(check_plotting_request("type = InputMatrix"));
  const auto histogram_plot = get_plotting_object("unweighted input histograms");
  const auto cumulative_plot = get_plotting_object("unweighted input empirical CDFs");
  ASSERT_TRUE(histogram_plot);
  ASSERT_TRUE(cumulative_plot);
  EXPECT_EQ((std::vector<double>{3, 1, 0}), histogram_plot->series()[0]->values());
  const auto & graph_keys = cumulative_plot->series()[0]->keys();
  const auto & graph_vals = cumulative_plot->series()[0]->values();
  const auto one_iter = std::find(graph_keys.begin(), graph_keys.end(), 1);
  ASSERT_NE(graph_keys.end(), one_iter);
  EXPECT_DOUBLE_EQ(0.5, graph_vals[one_iter - graph_keys.begin()]);
  EXPECT_EQ(0, graph_vals.front());
  EXPECT_EQ(1, graph_vals.back());
  EXPECT_EQ(
      wasp::CustomPlot::lsStepLeft,
      std::static_pointer_cast<wasp::CustomPlot::Graph>(cumulative_plot->series()[0])->lineStyle());
}

TEST_F(StochasticToolsMooseServerTest, PlottingSamplerConstantMarginals)
{
  // prepare input objects used by the specified plotting extension request
  ASSERT_TRUE(check_plotting_input(build_plotting_input(R"INPUT(
[Samplers]
  [design]
    type = InputMatrix
    matrix = '5; 5; 5'
  []
[]
)INPUT")));
  ASSERT_TRUE(check_plotting_request("type = InputMatrix"));
  const auto cumulative_plot = get_plotting_object("unweighted input empirical CDFs");
  ASSERT_TRUE(cumulative_plot);
  EXPECT_LT(cumulative_plot->x1Axis().rangeMin(), 5);
  EXPECT_GT(cumulative_plot->x1Axis().rangeMax(), 5);
  const auto & graph_keys = cumulative_plot->series()[0]->keys();
  const auto five_iter = std::find(graph_keys.begin(), graph_keys.end(), 5);
  ASSERT_NE(graph_keys.end(), five_iter);
  EXPECT_EQ(1, cumulative_plot->series()[0]->values()[five_iter - graph_keys.begin()]);
}

TEST_F(StochasticToolsMooseServerTest, PlottingSamplerCorrelations)
{
  // prepare input objects used by the specified plotting extension request
  ASSERT_TRUE(check_plotting_input(build_plotting_input(R"INPUT(
[Samplers]
  [design]
    type = InputMatrix
    matrix = '1 2 6 7; 2 4 4 7; 3 6 2 7'
  []
[]
)INPUT")));
  ASSERT_TRUE(check_plotting_request("type = InputMatrix"));
  const auto correlation_plot = get_plotting_object("displayed input correlations");
  ASSERT_TRUE(correlation_plot);
  ASSERT_EQ(3u, correlation_plot->series()[0]->values().size());
  // allow rounding in standardized correlations of exactly related columns
  EXPECT_NEAR(1, correlation_plot->series()[0]->values()[0], 1e-12);
  EXPECT_NEAR(-1, correlation_plot->series()[0]->values()[1], 1e-12);
  EXPECT_NEAR(-1, correlation_plot->series()[0]->values()[2], 1e-12);
  EXPECT_EQ("0-1", correlation_plot->x1Axis().ticks()[0].first);
  EXPECT_EQ("1-2", correlation_plot->x1Axis().ticks()[2].first);
  EXPECT_LT(correlation_plot->y1Axis().rangeMin(), -1);
  EXPECT_GT(correlation_plot->y1Axis().rangeMax(), 1);
}

TEST_F(StochasticToolsMooseServerTest, PlottingSamplerConstantCorrelations)
{
  // prepare input objects used by the specified plotting extension request
  ASSERT_TRUE(check_plotting_input(build_plotting_input(R"INPUT(
[Samplers]
  [design]
    type = InputMatrix
    matrix = '1 7; 2 7; 3 7; 4 7; 5 7; 6 7; 7 7; 8 7; 9 7; 10 7'
  []
[]
)INPUT")));
  ASSERT_TRUE(check_plotting_request("type = InputMatrix"));
  EXPECT_FALSE(get_plotting_object("displayed input correlations"));
}

TEST_F(StochasticToolsMooseServerTest, PlottingDistributionQuantiles)
{
  // prepare input objects used by the specified plotting extension request
  ASSERT_TRUE(check_plotting_input(build_plotting_input(R"INPUT(
[Distributions]
  [u]
    type = Uniform
    lower_bound = 1
    upper_bound = 5
  []
[]
)INPUT")));
  ASSERT_TRUE(check_plotting_request("type = Uniform"));
  const auto quantile_plot = get_plotting_object("input quantiles and interval endpoints");
  ASSERT_TRUE(quantile_plot);
  ASSERT_EQ(4u, quantile_plot->series().size());
  EXPECT_DOUBLE_EQ(3, quantile_plot->series()[1]->values()[0]);
  EXPECT_NEAR(1.1, quantile_plot->series()[3]->values()[0], 1e-12);
  EXPECT_NEAR(4.9, quantile_plot->series()[3]->values()[1], 1e-12);
  EXPECT_NE(std::string::npos, quantile_plot->series()[3]->name().find("CDF mass=0.950000"));
  const auto inverse_plot = get_plotting_object("input inverse-CDF check");
  ASSERT_TRUE(inverse_plot);
  const auto & probability_values = inverse_plot->series()[0]->keys();
  for (const auto i : index_range(probability_values))
    EXPECT_NEAR(probability_values[i], inverse_plot->series()[0]->values()[i], 1e-12);
}

TEST_F(StochasticToolsMooseServerTest, PlottingDistributionInverseCDF)
{
  // prepare input objects used by the specified plotting extension request
  ASSERT_TRUE(check_plotting_input(build_plotting_input(R"INPUT(
[Distributions]
  [kde]
    type = KernelDensity1D
    data = '-3 3'
    bandwidth_rule = userdefined
    bandwidth = 0.1
    kernel_function = gaussian
  []
[]
)INPUT")));
  ASSERT_TRUE(check_plotting_request("type = KernelDensity1D"));
  const auto inverse_plot = get_plotting_object("input inverse-CDF check");
  ASSERT_TRUE(inverse_plot);
  const auto & probabilities = inverse_plot->series()[0]->keys();
  const auto & cdf_values = inverse_plot->series()[0]->values();
  // check the approximate quantiles against their actual cumulative values
  EXPECT_GT(std::abs(cdf_values[50] - probabilities[50]), 0.05);
  auto & distribution =
      moose_server->getCheckApp().actionWarehouse().problemBase()->getDistribution("kde");
  EXPECT_DOUBLE_EQ(distribution.cdf(distribution.quantile(probabilities[50])), cdf_values[50]);
}

TEST_F(StochasticToolsMooseServerTest, PlottingDistributionUpperTail)
{
  // prepare input objects used by the specified plotting extension request
  ASSERT_TRUE(check_plotting_input(build_plotting_input(R"INPUT(
[Distributions]
  [u]
    type = Uniform
    lower_bound = 0
    upper_bound = 1
  []
[]
)INPUT")));
  ASSERT_TRUE(check_plotting_request("type = Uniform"));
  const auto tail_plot = get_plotting_object("input upper tail (1-CDF subtraction)");
  ASSERT_TRUE(tail_plot);
  const auto & graph_keys = tail_plot->series()[0]->keys();
  const auto & graph_vals = tail_plot->series()[0]->values();
  ASSERT_EQ(201u, graph_keys.size());
  for (const auto i : index_range(graph_keys))
    EXPECT_NEAR(1 - graph_keys[i], graph_vals[i], 1e-12);
  EXPECT_GT(graph_vals.front(), graph_vals.back());
  EXPECT_EQ(0, tail_plot->y1Axis().rangeMin());
  EXPECT_EQ(1, tail_plot->y1Axis().rangeMax());
}

TEST_F(StochasticToolsMooseServerTest, PlottingDistributionLogDensity)
{
  // prepare input objects used by the specified plotting extension request
  ASSERT_TRUE(check_plotting_input(build_plotting_input(R"INPUT(
[Distributions]
  [u]
    type = Uniform
    lower_bound = 0.5
    upper_bound = 2.5
  []
[]
)INPUT")));
  ASSERT_TRUE(check_plotting_request("type = Uniform"));
  const auto density_plot = get_plotting_object("input positive PDF");
  ASSERT_TRUE(density_plot);
  EXPECT_EQ(wasp::CustomPlot::stLogarithmic, density_plot->y1Axis().scaleType());
  EXPECT_DOUBLE_EQ(0.25, density_plot->y1Axis().rangeMin());
  EXPECT_DOUBLE_EQ(1, density_plot->y1Axis().rangeMax());
  for (const auto value : density_plot->series()[0]->values())
    EXPECT_DOUBLE_EQ(0.5, value);
  const auto original_plot = get_plotting_object(" PDF Distribution");
  ASSERT_TRUE(original_plot);
  EXPECT_DOUBLE_EQ(0.45, original_plot->y1Axis().rangeMin());
  EXPECT_DOUBLE_EQ(0.55, original_plot->y1Axis().rangeMax());
}

TEST_F(StochasticToolsMooseServerTest, PlottingDistributionComparisons)
{
  // prepare input objects used by the specified plotting extension request
  ASSERT_TRUE(check_plotting_input(build_plotting_input(R"INPUT(
[Distributions]
  [a]
    type = Uniform
    lower_bound = 0
    upper_bound = 1
  []
  [b]
    type = Uniform
    lower_bound = 2
    upper_bound = 4
  []
[]
)INPUT")));
  ASSERT_TRUE(check_plotting_request("lower_bound = 2"));
  const auto density_plot = get_plotting_object("configured PDF comparisons");
  const auto cumulative_plot = get_plotting_object("configured CDF comparisons");
  ASSERT_TRUE(density_plot);
  ASSERT_TRUE(cumulative_plot);
  ASSERT_EQ(2u, density_plot->series().size());
  EXPECT_EQ("b", density_plot->series()[0]->name());
  EXPECT_EQ("a", density_plot->series()[1]->name());
  EXPECT_DOUBLE_EQ(0.5, density_plot->series()[0]->values()[100]);
  EXPECT_DOUBLE_EQ(1, density_plot->series()[1]->values()[100]);
  EXPECT_EQ(0, cumulative_plot->y1Axis().rangeMin());
  EXPECT_EQ(1, cumulative_plot->y1Axis().rangeMax());
}

TEST_F(StochasticToolsMooseServerTest, PlottingSamplerPriors)
{
  // prepare input objects used by the specified plotting extension request
  const auto input_config_path = write_input_file("configuration.csv", "configuration\n1\n");
  ASSERT_TRUE(check_plotting_input(build_plotting_input(R"INPUT(
[Distributions]
  [u]
    type = Uniform
    lower_bound = 0
    upper_bound = 1
  []
[]
[Samplers]
  [chain]
    type = PMCMCBase
    prior_distributions = u
    initial_values = 0.25
    num_parallel_proposals = 1
    num_random_seeds = 2
    file_name = ')INPUT" + input_config_path + R"INPUT('
  []
[]
)INPUT")));
  ASSERT_TRUE(check_plotting_request("type = PMCMCBase"));
  const auto prior_plot = get_plotting_object("configured sampling distribution 0/1: u CDF");
  ASSERT_TRUE(prior_plot);
  ASSERT_EQ(2u, prior_plot->series().size());
  EXPECT_EQ((std::vector<double>{0.25}), prior_plot->series()[1]->keys());
  EXPECT_EQ((std::vector<double>{0.25}), prior_plot->series()[1]->values());
  EXPECT_FALSE(get_plotting_object("input design:"));
}
