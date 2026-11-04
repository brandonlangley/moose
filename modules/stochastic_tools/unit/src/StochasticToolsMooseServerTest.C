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
