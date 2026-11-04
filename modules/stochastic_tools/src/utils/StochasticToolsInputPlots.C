//* This file is part of the MOOSE framework
//* https://mooseframework.inl.gov
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#include "StochasticToolsInputPlots.h"
#include "MooseServer.h"
#include "FEProblemBase.h"
#include "Sampler.h"
#include "ConstantReporter.h"
#include "TheWarehouse.h"
#include "waspplot/CustomPlotSerialization.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace
{
template <typename T>
T *
getInputPlotObject(FEProblemBase & problem,
                   const std::string & object_name,
                   const std::string & object_system)
{
  // query the appropriate input object from the main execution thread only
  std::vector<MooseObject *> input_objects;
  problem.theWarehouse()
      .query()
      .condition<AttribSystem>(object_system)
      .condition<AttribName>(object_name)
      .condition<AttribThread>(0)
      .queryInto(input_objects);
  for (auto * object_ptr : input_objects)
    if (auto * input_object = dynamic_cast<T *>(object_ptr))
      return input_object;
  return nullptr;
}

void
addInputPlotCurve(MooseServer & moose_server,
                  wasp::CustomPlot & plot_object,
                  const std::string & series_name,
                  const std::vector<double> & graph_keys,
                  const std::vector<double> & graph_vals,
                  bool connect_points = false)
{
  // build validated plot data using the existing line graph range settings
  wasp::CustomPlot curve_object;
  moose_server.buildLineGraphPlot(curve_object, "", "", "", graph_keys, graph_vals);
  auto line_graph =
      std::static_pointer_cast<wasp::CustomPlot::Graph>(curve_object.series().front());
  line_graph->name(series_name);
  line_graph->lineStyle(connect_points ? wasp::CustomPlot::lsLine : wasp::CustomPlot::lsNone);
  line_graph->scatterShape(connect_points && graph_keys.size() > 1 ? wasp::CustomPlot::ssNone
                                                                   : wasp::CustomPlot::ssDisc);
  // use four different colors so plotted input columns remain identifiable
  const int colors[][3] = {{31, 119, 180}, {214, 39, 40}, {44, 160, 44}, {148, 103, 189}};
  const auto & plot_color = colors[plot_object.series().size() % 4];
  line_graph->pen().brush().color().red(plot_color[0]);
  line_graph->pen().brush().color().green(plot_color[1]);
  line_graph->pen().brush().color().blue(plot_color[2]);
  if (plot_object.series().empty())
  {
    plot_object.title().font() = curve_object.title().font();
    plot_object.x1Axis().labelFont() = curve_object.x1Axis().labelFont();
    plot_object.y1Axis().labelFont() = curve_object.y1Axis().labelFont();
    plot_object.x1Axis().tickLabelFont() = curve_object.x1Axis().tickLabelFont();
    plot_object.y1Axis().tickLabelFont() = curve_object.y1Axis().tickLabelFont();
    plot_object.x1Axis().rangeMin(curve_object.x1Axis().rangeMin());
    plot_object.x1Axis().rangeMax(curve_object.x1Axis().rangeMax());
    plot_object.y1Axis().rangeMin(curve_object.y1Axis().rangeMin());
    plot_object.y1Axis().rangeMax(curve_object.y1Axis().rangeMax());
  }
  else
  {
    plot_object.x1Axis().rangeMin(
        std::min(plot_object.x1Axis().rangeMin(), curve_object.x1Axis().rangeMin()));
    plot_object.x1Axis().rangeMax(
        std::max(plot_object.x1Axis().rangeMax(), curve_object.x1Axis().rangeMax()));
    plot_object.y1Axis().rangeMin(
        std::min(plot_object.y1Axis().rangeMin(), curve_object.y1Axis().rangeMin()));
    plot_object.y1Axis().rangeMax(
        std::max(plot_object.y1Axis().rangeMax(), curve_object.y1Axis().rangeMax()));
  }
  plot_object.series().push_back(line_graph);
}

wasp::CustomPlot
buildInputPlot(const std::string & plot_title,
               const std::string & x_axis_label,
               const std::string & y_axis_label)
{
  // prepare the plot titles and primary axes for all module input previews
  wasp::CustomPlot plot_object;
  plot_object.title().text(plot_title);
  plot_object.title().visible(true);
  plot_object.legend().visible(true);
  plot_object.x1Axis().label(x_axis_label);
  plot_object.y1Axis().label(y_axis_label);
  plot_object.x2Axis().visible(false);
  plot_object.y2Axis().visible(false);
  return plot_object;
}

void
addInputPlotResponse(wasp::DataArray & plotting_responses, const wasp::CustomPlot & plot_object)
{
  // append a serialized response only when graph data series are available
  if (!plot_object.series().empty())
    plotting_responses.push_back(wasp::serializeCustomPlot(plot_object));
}

bool
isSamplerInputPlotSupported(FEProblemBase &, const Sampler & sampler, unsigned int = 0)
{
  const std::set<std::string> sampler_types = {"InputMatrix",
                                               "CSVSampler",
                                               "CartesianProduct",
                                               "CartesianProductSampler",
                                               "Cartesian1D",
                                               "DirectPerturbationSampler",
                                               "Quadrature",
                                               "QuadratureSampler",
                                               "MonteCarlo",
                                               "MonteCarloSampler",
                                               "LatinHypercube",
                                               "NestedMonteCarlo"};
  return sampler_types.count(sampler.type());
}

bool
getSamplerInputVectors(FEProblemBase & problem,
                       const Sampler & sampler,
                       std::vector<const std::vector<Real> *> & column_values)
{
  // read constant Reporter input vectors without executing their reporters
  for (const auto & reporter_name : sampler.getParam<std::vector<ReporterName>>("vectors_names"))
  {
    const auto * reporter_ptr =
        getInputPlotObject<ConstantReporter>(problem, reporter_name.getObjectName(), "UserObject");
    if (!reporter_ptr || !reporter_ptr->isParamValid("real_vector_names"))
      return false;
    const auto & vector_names =
        reporter_ptr->getParam<std::vector<ReporterValueName>>("real_vector_names");
    const auto & vector_values =
        reporter_ptr->getParam<std::vector<std::vector<Real>>>("real_vector_values");
    const auto name_iter =
        std::find(vector_names.begin(), vector_names.end(), reporter_name.getValueName());
    if (name_iter == vector_names.end())
      return false;
    const auto & input_values = vector_values[name_iter - vector_names.begin()];
    if (input_values.empty() ||
        (!column_values.empty() && input_values.size() != column_values.front()->size()))
      return false;
    column_values.push_back(&input_values);
  }
  return !column_values.empty();
}

bool
getSamplerInputData(FEProblemBase & problem,
                    const Sampler & sampler,
                    std::vector<double> & row_indices,
                    std::vector<std::vector<double>> & column_values,
                    std::string & plot_title)
{
  // limit plot data to 2000 rows and four inputs without sampler execution
  constexpr dof_id_type row_limit = 2000;
  constexpr dof_id_type column_limit = 4;
  dof_id_type num_rows = sampler.getNumberOfRows();
  dof_id_type num_columns = sampler.getNumberOfCols();
  std::vector<const std::vector<Real> *> input_vectors;
  if (sampler.type() == "VectorPostprocessorSampler")
  {
    if (!getSamplerInputVectors(problem, sampler, input_vectors))
      return false;
    num_rows = input_vectors.front()->size();
    num_columns = input_vectors.size();
  }
  else if (!isSamplerInputPlotSupported(problem, sampler))
    return false;
  const bool use_first_rows =
      sampler.type() == "MonteCarlo" || sampler.type() == "MonteCarloSampler" ||
      sampler.type() == "LatinHypercube" || sampler.type() == "NestedMonteCarlo";
  // cap indexed work at 200000 draws as earlier random draws are traversed
  constexpr dof_id_type draw_limit = 200000;
  if (sampler.type() == "LatinHypercube" && num_columns > 1024)
    // limit LHS to 1024 columns as each column needs its permutation cache
    return false;
  auto plot_num_rows = std::min(num_rows, row_limit);
  if (use_first_rows)
    plot_num_rows = std::min(plot_num_rows,
                             1 + (draw_limit - std::min(num_columns, column_limit)) / num_columns);
  if (sampler.type() == "NestedMonteCarlo")
    // tighten the draw budget for reseeding with nested input sampler rows
    plot_num_rows =
        std::min(plot_num_rows,
                 static_cast<dof_id_type>(std::sqrt(draw_limit / double(num_columns) /
                                                    std::min(num_columns, column_limit))));
  if (!plot_num_rows || !num_columns)
    return false;
  column_values.resize(std::min(num_columns, column_limit));
  for (const auto i : make_range(plot_num_rows))
  {
    const auto sample_row =
        use_first_rows || plot_num_rows == 1
            ? i
            : ((num_rows - 1) / (plot_num_rows - 1)) * i +
                  ((num_rows - 1) % (plot_num_rows - 1)) * i / (plot_num_rows - 1);
    row_indices.push_back(sample_row);
    for (const auto column : index_range(column_values))
    {
      const auto sample_value = input_vectors.empty() ? sampler.getSample(sample_row, column)
                                                      : (*input_vectors[column])[sample_row];
      if (!std::isfinite(sample_value))
        throw std::invalid_argument("Input sampler preview contains a nonfinite value");
      column_values[column].push_back(sample_value);
    }
  }
  plot_title = sampler.name() + " (input design: " + std::to_string(plot_num_rows) + "/" +
               std::to_string(num_rows) + " rows; first " + std::to_string(column_values.size()) +
               "/" + std::to_string(num_columns) + " columns; " +
               (use_first_rows ? "first rows" : "evenly spaced rows") + ")";
  return true;
}

void
buildSamplerPlotResponses(MooseServer & moose_server,
                          wasp::DataArray & plotting_responses,
                          FEProblemBase & problem,
                          const std::string & object_name)
{
  // read the bounded sampler design once for all associated input previews
  const auto * sampler_ptr = getInputPlotObject<Sampler>(problem, object_name, "Sampler");
  if (!sampler_ptr)
    return;
  std::vector<double> row_indices;
  std::vector<std::vector<double>> column_values;
  std::string plot_title;
  if (!getSamplerInputData(problem, *sampler_ptr, row_indices, column_values, plot_title))
    return;
  auto design_plot = buildInputPlot(plot_title, "sample row index", "input value");
  for (const auto column : index_range(column_values))
    addInputPlotCurve(moose_server,
                      design_plot,
                      "column " + std::to_string(column),
                      row_indices,
                      column_values[column]);
  addInputPlotResponse(plotting_responses, design_plot);
  if (column_values.size() > 1)
  {
    auto scatter_plot = buildInputPlot(plot_title + " paired columns", "column 0", "column 1");
    addInputPlotCurve(
        moose_server, scatter_plot, "input pairs", column_values[0], column_values[1]);
    addInputPlotResponse(plotting_responses, scatter_plot);
  }
}
}

void
StochasticTools::registerInputPlots()
{
  MooseServer::registerInputPlotter("Samplers", "input-design", buildSamplerPlotResponses);
}
