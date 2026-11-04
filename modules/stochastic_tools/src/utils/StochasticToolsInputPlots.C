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
#include "QuadratureSampler.h"
#include "TruncatedNormal.h"
#include "Gaussian.h"
#include "ScaledAbsDifferenceDRLRewardFunction.h"
#include "CovarianceFunctionBase.h"
#include "ConstantReporter.h"
#include "TheWarehouse.h"
#include "ParameterStudyAction.h"
#include "ActionWarehouse.h"
#include "waspplot/CustomPlotSerialization.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <limits>

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
isSamplerInputPlotSupported(FEProblemBase & problem,
                            const Sampler & sampler,
                            unsigned int sampler_depth = 0)
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
                                               "NestedMonteCarlo",
                                               "MorrisSampler"};
  if (sampler.type() == "Sobol" || sampler.type() == "SobolSampler")
  {
    // bound Sobol composition depth to four nested input sampler instances
    if (sampler_depth >= 4)
      return false;
    const auto * sampler_a =
        getInputPlotObject<Sampler>(problem, sampler.getParam<SamplerName>("sampler_a"), "Sampler");
    const auto * sampler_b =
        getInputPlotObject<Sampler>(problem, sampler.getParam<SamplerName>("sampler_b"), "Sampler");
    return sampler_a && sampler_b &&
           isSamplerInputPlotSupported(problem, *sampler_a, sampler_depth + 1) &&
           isSamplerInputPlotSupported(problem, *sampler_b, sampler_depth + 1);
  }
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
  const bool use_first_rows = sampler.type() == "Sobol" || sampler.type() == "SobolSampler" ||
                              sampler.type() == "MorrisSampler" || sampler.type() == "MonteCarlo" ||
                              sampler.type() == "MonteCarloSampler" ||
                              sampler.type() == "LatinHypercube" ||
                              sampler.type() == "NestedMonteCarlo";
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
  // cap dense Morris work at eight dimensions and eight input trajectories
  if (sampler.type() == "MorrisSampler")
  {
    if (num_columns > 8)
      return false;
    plot_num_rows = std::min<dof_id_type>(num_rows, 8 * (num_columns + 1));
  }
  if (sampler.type() == "Sobol" || sampler.type() == "SobolSampler")
  {
    if (num_columns > 8)
      return false;
    const auto blocks = (sampler.getParam<bool>("resample") ? 2 : 1) * num_columns + 2;
    // read all Sobol blocks from eight base rows to bound indexed sampling
    plot_num_rows = std::min<dof_id_type>(num_rows, 8 * blocks);
  }
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
buildMorrisPlotResponse(MooseServer & moose_server,
                        wasp::DataArray & plotting_responses,
                        const Sampler & sampler,
                        const std::vector<std::vector<double>> & column_values)
{
  // index every input trajectory by steps so graph sorting preserves order
  if (sampler.type() != "MorrisSampler")
    return;
  const auto path_length = sampler.getNumberOfCols() + 1;
  auto plot_object = buildInputPlot(
      sampler.name() + " Morris input trajectories", "trajectory step", "input value");
  std::vector<double> graph_keys;
  for (const auto i : make_range(path_length))
    graph_keys.push_back(i);
  for (const auto trajectory : make_range(column_values[0].size() / path_length))
  {
    const auto start_row = trajectory * path_length;
    for (const auto column : index_range(column_values))
    {
      const std::vector<double> graph_vals(column_values[column].begin() + start_row,
                                           column_values[column].begin() + start_row + path_length);
      addInputPlotCurve(moose_server,
                        plot_object,
                        "trajectory " + std::to_string(trajectory) + ", column " +
                            std::to_string(column),
                        graph_keys,
                        graph_vals,
                        true);
    }
  }
  addInputPlotResponse(plotting_responses, plot_object);
}

void
buildSobolPlotResponse(MooseServer & moose_server,
                       wasp::DataArray & plotting_responses,
                       const Sampler & sampler,
                       const std::vector<std::vector<double>> & column_values)
{
  // show the initial B and hybrid input rows followed by their final A row
  if (sampler.type() != "Sobol" && sampler.type() != "SobolSampler")
    return;
  const auto num_blocks =
      (sampler.getParam<bool>("resample") ? 2 : 1) * sampler.getNumberOfCols() + 2;
  if (column_values[0].size() < num_blocks)
    return;
  std::vector<double> graph_keys;
  for (const auto i : make_range(num_blocks))
    graph_keys.push_back(i);
  auto plot_object = buildInputPlot(sampler.name() + " Sobol input matrix blocks (first base row)",
                                    "matrix block index (0=B, last=A)",
                                    "input value");
  for (const auto column : index_range(column_values))
    addInputPlotCurve(moose_server,
                      plot_object,
                      "column " + std::to_string(column),
                      graph_keys,
                      std::vector<double>(column_values[column].begin(),
                                          column_values[column].begin() + num_blocks));
  addInputPlotResponse(plotting_responses, plot_object);
}

void
buildQuadraturePlotResponse(MooseServer & moose_server,
                            wasp::DataArray & plotting_responses,
                            const Sampler & sampler,
                            const std::vector<double> & row_indices,
                            const std::string & plot_title)
{
  // read each signed quadrature weight for the selected input sampler rows
  const auto * quad_sampler = dynamic_cast<const QuadratureSampler *>(&sampler);
  if (!quad_sampler)
    return;
  std::vector<double> graph_vals;
  for (const auto sample_row : row_indices)
    graph_vals.push_back(quad_sampler->getQuadratureWeight(static_cast<dof_id_type>(sample_row)));
  auto plot_object = buildInputPlot(plot_title + " quadrature weights (unnormalized)",
                                    "quadrature row index",
                                    "quadrature weight");
  addInputPlotCurve(moose_server, plot_object, "displayed node weights", row_indices, graph_vals);
  addInputPlotResponse(plotting_responses, plot_object);
}

void
buildPolynomialPlotResponses(MooseServer & moose_server,
                             wasp::DataArray & plotting_responses,
                             FEProblemBase & problem,
                             const Sampler & sampler)
{
  if (!dynamic_cast<const QuadratureSampler *>(&sampler))
    return;
  const auto & dist_names = sampler.getParam<std::vector<DistributionName>>("distributions");
  // show two dimensions and degrees up through four for simple basis plots
  for (const auto column : make_range(std::min<std::size_t>(2, dist_names.size())))
  {
    const auto & dist = problem.getDistribution(dist_names[column]);
    const auto poly_ptr = PolynomialQuadrature::makePolynomial(&dist);
    std::vector<double> graph_keys;
    for (const auto i : make_range(201))
      graph_keys.push_back(dist.quantile(0.001 + 0.998 * i / 200.0));
    auto plot_object = buildInputPlot(sampler.name() + " polynomial basis: " + dist_names[column],
                                      "input value",
                                      "unnormalized basis value");
    for (const auto degree : make_range(std::min(4u, sampler.getParam<unsigned int>("order")) + 1))
    {
      std::vector<double> graph_vals;
      for (const auto sample_value : graph_keys)
        graph_vals.push_back(poly_ptr->compute(degree, sample_value, false));
      addInputPlotCurve(moose_server,
                        plot_object,
                        "degree " + std::to_string(degree),
                        graph_keys,
                        graph_vals,
                        true);
    }
    addInputPlotResponse(plotting_responses, plot_object);
  }
}

void
buildMarginalPlotResponses(MooseServer & moose_server,
                           wasp::DataArray & plotting_responses,
                           const std::vector<std::vector<double>> & column_values,
                           const std::string & plot_title)
{
  // build unweighted histograms and cumulative fractions of displayed rows
  auto histogram_plot = buildInputPlot(
      plot_title + " unweighted input histograms", "input value", "displayed row count");
  auto cumulative_plot = buildInputPlot(
      plot_title + " unweighted input empirical CDFs", "input value", "displayed fraction <= x");
  for (const auto column : index_range(column_values))
  {
    auto sorted_values = column_values[column];
    std::sort(sorted_values.begin(), sorted_values.end());
    const auto num_bins = std::min<std::size_t>(32, std::ceil(std::sqrt(sorted_values.size())));
    double range_min = sorted_values.front(), range_max = sorted_values.back();
    if (range_min == range_max)
    {
      const auto range_padding = 0.05 * std::max(1.0, std::abs(range_min));
      range_min -= range_padding;
      range_max += range_padding;
    }
    const auto bin_width = (range_max - range_min) / num_bins;
    if (!std::isfinite(bin_width) || bin_width <= 0)
      throw std::invalid_argument("Input histogram has no representable finite bin width");
    std::vector<double> bin_edges, bin_counts(num_bins + 1, 0);
    for (const auto i : make_range(num_bins + 1))
      bin_edges.push_back(range_min + (range_max - range_min) * (double(i) / num_bins));
    for (const auto sample_value : sorted_values)
      ++bin_counts[std::min<std::size_t>(num_bins - 1, (sample_value - range_min) / bin_width)];
    addInputPlotCurve(moose_server,
                      histogram_plot,
                      "column " + std::to_string(column),
                      bin_edges,
                      bin_counts,
                      true);
    std::static_pointer_cast<wasp::CustomPlot::Graph>(histogram_plot.series().back())
        ->lineStyle(wasp::CustomPlot::lsStepLeft);

    const auto range_padding =
        0.05 * std::max(range_max - range_min, std::max(1.0, std::abs(range_min)));
    std::vector<double> graph_keys, graph_vals;
    if (std::isfinite(sorted_values.front() - range_padding) &&
        sorted_values.front() - range_padding < sorted_values.front())
    {
      graph_keys.push_back(sorted_values.front() - range_padding);
      graph_vals.push_back(0);
    }
    for (std::size_t i = 0; i < sorted_values.size();)
    {
      const auto sample_value = sorted_values[i];
      while (i < sorted_values.size() && sorted_values[i] == sample_value)
        ++i;
      graph_keys.push_back(sample_value);
      graph_vals.push_back(double(i) / sorted_values.size());
    }
    if (std::isfinite(sorted_values.back() + range_padding) &&
        sorted_values.back() + range_padding > sorted_values.back())
    {
      graph_keys.push_back(sorted_values.back() + range_padding);
      graph_vals.push_back(1);
    }
    addInputPlotCurve(moose_server,
                      cumulative_plot,
                      "column " + std::to_string(column),
                      graph_keys,
                      graph_vals,
                      true);
    std::static_pointer_cast<wasp::CustomPlot::Graph>(cumulative_plot.series().back())
        ->lineStyle(wasp::CustomPlot::lsStepLeft);
  }
  histogram_plot.y1Axis().rangeMin(0);
  cumulative_plot.y1Axis().rangeMin(0);
  cumulative_plot.y1Axis().rangeMax(1);
  addInputPlotResponse(plotting_responses, histogram_plot);
  addInputPlotResponse(plotting_responses, cumulative_plot);
}

void
buildCorrelationPlotResponse(MooseServer & moose_server,
                             wasp::DataArray & plotting_responses,
                             const std::vector<std::vector<double>> & column_values,
                             const std::string & plot_title)
{
  // scale the input columns before centering to avoid unnecessary overflow
  auto scaled_values = column_values;
  std::vector<double> column_variance(column_values.size(), 0);
  for (const auto column : index_range(scaled_values))
  {
    const auto data_bounds =
        std::minmax_element(column_values[column].begin(), column_values[column].end());
    if (*data_bounds.first == *data_bounds.second)
      continue;
    double value_scale = 0, sample_mean = 0;
    for (const auto sample_value : scaled_values[column])
      value_scale = std::max(value_scale, std::abs(sample_value));
    if (!value_scale)
      continue;
    const auto first_value = scaled_values[column].front();
    for (auto & sample_value : scaled_values[column])
    {
      const auto value_difference = sample_value - first_value;
      sample_value = std::isfinite(value_difference)
                         ? value_difference / value_scale
                         : sample_value / value_scale - first_value / value_scale;
      sample_mean += sample_value / scaled_values[column].size();
    }
    for (auto & sample_value : scaled_values[column])
    {
      sample_value -= sample_mean;
      column_variance[column] += sample_value * sample_value;
    }
  }
  auto plot_object = buildInputPlot(
      plot_title + " displayed input correlations", "column pair", "Pearson sample correlation");
  std::vector<double> graph_keys, graph_vals;
  for (const auto i : index_range(column_values))
    for (std::size_t j = i + 1; j < column_values.size(); ++j)
    {
      if (!column_variance[i] || !column_variance[j])
        // omit constant input pairs since their correlations are undefined
        continue;
      double covariance = 0;
      for (const auto sample_row : index_range(column_values[i]))
        covariance += scaled_values[i][sample_row] * scaled_values[j][sample_row];
      const auto correlation =
          covariance / (std::sqrt(column_variance[i]) * std::sqrt(column_variance[j]));
      graph_keys.push_back(graph_keys.size());
      graph_vals.push_back(std::clamp(correlation, -1.0, 1.0));
      plot_object.x1Axis().ticks().push_back(
          {std::to_string(i) + "-" + std::to_string(j), graph_keys.back()});
    }
  if (graph_keys.empty())
    return;
  addInputPlotCurve(moose_server, plot_object, "displayed rows", graph_keys, graph_vals);
  // pad correlation bounds five percent to leave the endpoint dots visible
  plot_object.y1Axis().rangeMin(-1.05);
  plot_object.y1Axis().rangeMax(1.05);
  addInputPlotResponse(plotting_responses, plot_object);
}

void
getDistributionPlotData(const Distribution & dist,
                        std::vector<double> & probabilities,
                        std::vector<double> & quantiles)
{
  // use 201 nodes from 0.1 to 99.9 percent to omit tails and retain median
  for (const auto i : make_range(201))
  {
    probabilities.push_back(0.001 + 0.998 * i / 200.0);
    quantiles.push_back(dist.quantile(probabilities.back()));
    if (!std::isfinite(quantiles.back()) ||
        (quantiles.size() > 1 && quantiles.back() < quantiles[quantiles.size() - 2]))
      throw std::invalid_argument("Input quantile preview contains nonfinite or decreasing values");
  }
}

std::vector<double>
getDistributionPlotCDF(const Distribution & dist, const std::vector<double> & quantiles)
{
  std::vector<double> probabilities;
  // allow 64 epsilon rounding as in the usual CDF probability range checks
  const auto cdf_tolerance = 64 * std::numeric_limits<double>::epsilon();
  for (const auto quantile_value : quantiles)
  {
    const auto cdf_value = dist.cdf(quantile_value);
    if (!std::isfinite(cdf_value) || cdf_value < -cdf_tolerance || cdf_value > 1 + cdf_tolerance ||
        (!probabilities.empty() && cdf_value + cdf_tolerance < probabilities.back()))
      throw std::invalid_argument("Input CDF preview contains invalid probabilities");
    probabilities.push_back(std::clamp(cdf_value, 0.0, 1.0));
  }
  return probabilities;
}

void
buildQuantilePlotResponses(MooseServer & moose_server,
                           wasp::DataArray & plotting_responses,
                           FEProblemBase & problem,
                           const std::string & object_name)
{
  // evaluate input quantiles and mark the median and probability intervals
  const auto * dist_ptr = getInputPlotObject<Distribution>(problem, object_name, "Distribution");
  if (!dist_ptr)
    return;
  std::vector<double> probabilities, quantiles;
  getDistributionPlotData(*dist_ptr, probabilities, quantiles);
  auto plot_object = buildInputPlot(
      object_name + " input quantiles and interval endpoints", "probability", "returned quantile");
  addInputPlotCurve(moose_server, plot_object, "Q(p)", probabilities, quantiles, true);
  addInputPlotCurve(moose_server, plot_object, "median", {0.5}, {dist_ptr->quantile(0.5)});
  for (const auto tail_probability : {0.25, 0.025})
  {
    const std::vector<double> interval_values = {dist_ptr->quantile(tail_probability),
                                                 dist_ptr->quantile(1 - tail_probability)};
    std::string series_name = std::to_string(tail_probability) + " to " +
                              std::to_string(1 - tail_probability) + " quantiles";
    try
    {
      const auto cdf_values = getDistributionPlotCDF(*dist_ptr, interval_values);
      series_name += "; CDF mass=" + std::to_string(cdf_values[1] - cdf_values[0]);
    }
    catch (const std::exception &)
    {
      series_name += "; CDF mass unavailable";
    }
    addInputPlotCurve(moose_server,
                      plot_object,
                      series_name,
                      {tail_probability, 1 - tail_probability},
                      interval_values);
  }
  addInputPlotResponse(plotting_responses, plot_object);
  // keep the quantile plot if cumulative probabilities cannot be evaluated
  const auto cdf_values = getDistributionPlotCDF(*dist_ptr, quantiles);
  auto inverse_plot = buildInputPlot(
      object_name + " input inverse-CDF check", "requested probability p", "CDF(Q(p))");
  addInputPlotCurve(moose_server, inverse_plot, "CDF(Q(p))", probabilities, cdf_values, true);
  addInputPlotCurve(
      moose_server, inverse_plot, "identity reference", probabilities, probabilities, true);
  inverse_plot.y1Axis().rangeMin(0);
  inverse_plot.y1Axis().rangeMax(1);
  addInputPlotResponse(plotting_responses, inverse_plot);
}

void
buildTailPlotResponse(MooseServer & moose_server,
                      wasp::DataArray & plotting_responses,
                      FEProblemBase & problem,
                      const std::string & object_name)
{
  // subtract returned cumulative probabilities to obtain upper tail values
  const auto * dist_ptr = getInputPlotObject<Distribution>(problem, object_name, "Distribution");
  if (!dist_ptr)
    return;
  std::vector<double> probabilities, quantiles;
  getDistributionPlotData(*dist_ptr, probabilities, quantiles);
  auto graph_vals = getDistributionPlotCDF(*dist_ptr, quantiles);
  for (auto & cdf_value : graph_vals)
    cdf_value = 1 - cdf_value;
  auto plot_object = buildInputPlot(
      object_name + " input upper tail (1-CDF subtraction)", "input value", "P(X > x) = 1-CDF(x)");
  addInputPlotCurve(moose_server, plot_object, "1-CDF(x)", quantiles, graph_vals, true);
  plot_object.y1Axis().rangeMin(0);
  plot_object.y1Axis().rangeMax(1);
  addInputPlotResponse(plotting_responses, plot_object);
}

void
buildLogDensityPlotResponse(MooseServer & moose_server,
                            wasp::DataArray & plotting_responses,
                            FEProblemBase & problem,
                            const std::string & object_name)
{
  // split positive density curves at zeros which log axes cannot represent
  const auto * dist_ptr = getInputPlotObject<Distribution>(problem, object_name, "Distribution");
  if (!dist_ptr)
    return;
  std::vector<double> probabilities, quantiles, graph_keys, graph_vals;
  getDistributionPlotData(*dist_ptr, probabilities, quantiles);
  auto plot_object =
      buildInputPlot(object_name + " input positive PDF (log scale; sampled zeros omitted)",
                     "input value",
                     "probability density (log scale)");
  double density_min = std::numeric_limits<double>::max(), density_max = 0;
  const auto append_curve = [&]()
  {
    if (!graph_keys.empty())
      addInputPlotCurve(
          moose_server, plot_object, "positive density", graph_keys, graph_vals, true);
    graph_keys.clear();
    graph_vals.clear();
  };
  for (const auto quantile_value : quantiles)
  {
    const auto density = dist_ptr->pdf(quantile_value);
    if (!std::isfinite(density) || density < 0)
      throw std::invalid_argument("Log-density preview contains an invalid density");
    if (!density)
    {
      append_curve();
      continue;
    }
    density_min = std::min(density_min, density);
    density_max = std::max(density_max, density);
    graph_keys.push_back(quantile_value);
    graph_vals.push_back(density);
  }
  append_curve();
  if (!density_max)
    return;
  plot_object.y1Axis().scaleType(wasp::CustomPlot::stLogarithmic);
  plot_object.y1Axis().rangeMin(
      std::max(density_min / 2, std::numeric_limits<double>::denorm_min()));
  plot_object.y1Axis().rangeMax(density_max < std::numeric_limits<double>::max() / 2
                                    ? density_max * 2
                                    : std::numeric_limits<double>::max());
  addInputPlotResponse(plotting_responses, plot_object);
}

void
buildComparisonPlotResponses(MooseServer & moose_server,
                             wasp::DataArray & plotting_responses,
                             FEProblemBase & problem,
                             const std::string & object_name)
{
  // place the configured distribution before three other active input laws
  auto * selected_dist = getInputPlotObject<Distribution>(problem, object_name, "Distribution");
  if (!selected_dist)
    return;
  std::vector<Distribution *> all_distributions;
  problem.theWarehouse()
      .query()
      .condition<AttribSystem>("Distribution")
      .queryInto(all_distributions);
  std::sort(all_distributions.begin(),
            all_distributions.end(),
            [](const auto * a, const auto * b) { return a->name() < b->name(); });
  std::vector<Distribution *> plot_distributions = {selected_dist};
  for (auto * dist_ptr : all_distributions)
    if (dist_ptr != selected_dist && plot_distributions.size() < 4)
      plot_distributions.push_back(dist_ptr);
  if (plot_distributions.size() < 2)
    return;
  auto density_plot = buildInputPlot(
      object_name + " configured PDF comparisons", "input value", "probability density");
  auto cumulative_plot = buildInputPlot(
      object_name + " configured CDF comparisons", "input value", "cumulative probability");
  for (const auto * dist_ptr : plot_distributions)
  {
    std::vector<double> probabilities, quantiles;
    try
    {
      getDistributionPlotData(*dist_ptr, probabilities, quantiles);
    }
    catch (const std::exception &)
    {
      continue;
    }
    try
    {
      std::vector<double> graph_vals;
      for (const auto quantile_value : quantiles)
      {
        graph_vals.push_back(dist_ptr->pdf(quantile_value));
        if (!std::isfinite(graph_vals.back()) || graph_vals.back() < 0)
          throw std::invalid_argument("Invalid comparison density");
      }
      addInputPlotCurve(moose_server, density_plot, dist_ptr->name(), quantiles, graph_vals, true);
    }
    catch (const std::exception &)
    {
    }
    try
    {
      addInputPlotCurve(moose_server,
                        cumulative_plot,
                        dist_ptr->name(),
                        quantiles,
                        getDistributionPlotCDF(*dist_ptr, quantiles),
                        true);
    }
    catch (const std::exception &)
    {
    }
  }
  for (auto * plot_object : {&density_plot, &cumulative_plot})
    if (plot_object->series().size() > 1)
    {
      plot_object->title().text(plot_object->title().text() + " (" +
                                std::to_string(plot_object->series().size()) + "/" +
                                std::to_string(plot_distributions.size()) + " curves available)");
      if (plot_object == &cumulative_plot)
      {
        plot_object->y1Axis().rangeMin(0);
        plot_object->y1Axis().rangeMax(1);
      }
      addInputPlotResponse(plotting_responses, *plot_object);
    }
}

void
buildPriorPlotResponses(MooseServer & moose_server,
                        wasp::DataArray & plotting_responses,
                        FEProblemBase & problem,
                        const std::string & object_name)
{
  // read configured prior distributions without requesting sampler samples
  const auto * sampler_ptr = getInputPlotObject<Sampler>(problem, object_name, "Sampler");
  if (!sampler_ptr)
    return;
  std::vector<DistributionName> dist_names;
  if (sampler_ptr->isParamValid("prior_distributions"))
  {
    dist_names = sampler_ptr->getParam<std::vector<DistributionName>>("prior_distributions");
    if (sampler_ptr->isParamValid("prior_variance"))
      dist_names.push_back(sampler_ptr->getParam<DistributionName>("prior_variance"));
  }
  else if (!isSamplerInputPlotSupported(problem, *sampler_ptr) &&
           sampler_ptr->parameters().have_parameter<std::vector<DistributionName>>(
               "distributions") &&
           sampler_ptr->isParamValid("distributions"))
    dist_names = sampler_ptr->getParam<std::vector<DistributionName>>("distributions");
  const auto * initial_values = sampler_ptr->isParamValid("initial_values")
                                    ? &sampler_ptr->getParam<std::vector<Real>>("initial_values")
                                    : nullptr;
  for (const auto column : make_range(std::min<std::size_t>(4, dist_names.size())))
  {
    const auto & dist = problem.getDistribution(dist_names[column]);
    std::vector<double> probabilities, quantiles;
    try
    {
      getDistributionPlotData(dist, probabilities, quantiles);
    }
    catch (const std::exception &)
    {
      continue;
    }
    for (const std::string plot_type : {"PDF", "CDF"})
      try
      {
        std::vector<double> graph_vals;
        if (plot_type == "CDF")
          graph_vals = getDistributionPlotCDF(dist, quantiles);
        else
          for (const auto initial_value : quantiles)
          {
            graph_vals.push_back(dist.pdf(initial_value));
            if (!std::isfinite(graph_vals.back()) || graph_vals.back() < 0)
              throw std::invalid_argument("Invalid configured prior density");
          }
        auto plot_object = buildInputPlot(
            object_name + " configured sampling distribution " + std::to_string(column) + "/" +
                std::to_string(dist_names.size()) + ": " + dist_names[column] + " " + plot_type,
            "input value",
            plot_type == "PDF" ? "probability density" : "cumulative probability");
        addInputPlotCurve(
            moose_server, plot_object, dist_names[column], quantiles, graph_vals, true);
        if (initial_values && column < initial_values->size() &&
            std::isfinite((*initial_values)[column]))
        {
          const auto initial_value = (*initial_values)[column];
          const auto graph_value =
              plot_type == "PDF" ? dist.pdf(initial_value) : dist.cdf(initial_value);
          if (std::isfinite(graph_value) && graph_value >= 0 &&
              (plot_type == "PDF" || graph_value <= 1))
            addInputPlotCurve(moose_server,
                              plot_object,
                              "configured initial value",
                              {initial_value},
                              {graph_value});
        }
        if (plot_type == "CDF")
        {
          plot_object.y1Axis().rangeMin(0);
          plot_object.y1Axis().rangeMax(1);
        }
        addInputPlotResponse(plotting_responses, plot_object);
      }
      catch (const std::exception &)
      {
      }
  }
}

void
buildProposalPlotResponses(MooseServer & moose_server,
                           wasp::DataArray & plotting_responses,
                           FEProblemBase & problem,
                           const std::string & object_name)
{
  // use specified initial means and proposal scales for four input columns
  const auto * sampler_ptr = getInputPlotObject<Sampler>(problem, object_name, "Sampler");
  if (!sampler_ptr ||
      (sampler_ptr->type() != "IndependentGaussianMH" &&
       sampler_ptr->type() != "AdaptiveImportance" && sampler_ptr->type() != "AISActiveLearning"))
    return;
  const bool use_standard_normal = sampler_ptr->type() != "IndependentGaussianMH";
  const auto & initial_values = sampler_ptr->getParam<std::vector<Real>>("initial_values");
  const auto & std_deviations =
      sampler_ptr->getParam<std::vector<Real>>(use_standard_normal ? "proposal_std" : "std_prop");
  if (initial_values.size() != std_deviations.size())
    throw std::invalid_argument(
        "Initial proposal means and standard deviations have different sizes");
  const auto * lower_bounds = !use_standard_normal && sampler_ptr->isParamValid("lower_bound")
                                  ? &sampler_ptr->getParam<std::vector<Real>>("lower_bound")
                                  : nullptr;
  const auto * upper_bounds = !use_standard_normal && sampler_ptr->isParamValid("upper_bound")
                                  ? &sampler_ptr->getParam<std::vector<Real>>("upper_bound")
                                  : nullptr;
  if ((lower_bounds || upper_bounds) &&
      (!lower_bounds || !upper_bounds || lower_bounds->size() != initial_values.size() ||
       upper_bounds->size() != initial_values.size()))
    throw std::invalid_argument("Initial proposal bounds do not match the parameter count");
  for (const auto column : make_range(std::min<std::size_t>(4, initial_values.size())))
  {
    auto proposal_mean = initial_values[column];
    if (use_standard_normal)
    {
      const auto & dist_names =
          sampler_ptr->getParam<std::vector<DistributionName>>("distributions");
      if (dist_names.size() != initial_values.size())
        throw std::invalid_argument("Proposal distributions do not match the initial values");
      const auto initial_probability =
          problem.getDistribution(dist_names[column]).cdf(proposal_mean);
      if (!std::isfinite(initial_probability) || initial_probability <= 0 ||
          initial_probability >= 1)
        throw std::invalid_argument("Initial value has no finite standard-normal transform");
      proposal_mean = Normal::quantile(initial_probability, 0, 1);
    }
    if (!std::isfinite(proposal_mean) || !std::isfinite(std_deviations[column]) ||
        std_deviations[column] <= 0 ||
        (lower_bounds &&
         (!std::isfinite((*lower_bounds)[column]) || !std::isfinite((*upper_bounds)[column]) ||
          (*lower_bounds)[column] >= (*upper_bounds)[column])))
      throw std::invalid_argument("Initial proposal parameters are invalid");
    std::vector<double> graph_keys, graph_vals;
    for (const auto i : make_range(201))
    {
      const double probability = 0.001 + 0.998 * i / 200.0;
      const auto sample_value =
          lower_bounds ? TruncatedNormal::quantile(probability,
                                                   proposal_mean,
                                                   std_deviations[column],
                                                   (*lower_bounds)[column],
                                                   (*upper_bounds)[column])
                       : Normal::quantile(probability, proposal_mean, std_deviations[column]);
      graph_keys.push_back(sample_value);
      graph_vals.push_back(lower_bounds
                               ? TruncatedNormal::pdf(sample_value,
                                                      proposal_mean,
                                                      std_deviations[column],
                                                      (*lower_bounds)[column],
                                                      (*upper_bounds)[column])
                               : Normal::pdf(sample_value, proposal_mean, std_deviations[column]));
    }
    auto plot_object =
        buildInputPlot(object_name + " initial proposal density: column " + std::to_string(column),
                       use_standard_normal ? "standard-normal coordinate" : "parameter value",
                       "proposal density");
    addInputPlotCurve(moose_server,
                      plot_object,
                      "initial proposal (not the later adapted proposal)",
                      graph_keys,
                      graph_vals,
                      true);
    addInputPlotResponse(plotting_responses, plot_object);
  }
}

void
buildLikelihoodPlotResponses(MooseServer & moose_server,
                             wasp::DataArray & plotting_responses,
                             FEProblemBase & problem,
                             const std::string & object_name)
{
  // show at most 2000 configured observations including the two input ends
  const auto * likelihood_ptr =
      getInputPlotObject<Gaussian>(problem, object_name, "LikelihoodFunctionBase");
  if (!likelihood_ptr)
    return;
  const auto & observation_values = likelihood_ptr->getExperimentalValues();
  if (observation_values.empty())
    return;
  std::vector<double> graph_keys, graph_vals;
  const auto num_points = std::min<std::size_t>(2000, observation_values.size());
  for (const auto i : make_range(num_points))
  {
    const auto sample_row =
        num_points == 1
            ? 0
            : ((observation_values.size() - 1) / (num_points - 1)) * i +
                  ((observation_values.size() - 1) % (num_points - 1)) * i / (num_points - 1);
    graph_keys.push_back(sample_row);
    graph_vals.push_back(observation_values[sample_row]);
  }
  auto observations_plot =
      buildInputPlot(object_name + " input likelihood observations (" + std::to_string(num_points) +
                         "/" + std::to_string(observation_values.size()) + ")",
                     "observation index",
                     "observed value");
  addInputPlotCurve(
      moose_server, observations_plot, "configured observations", graph_keys, graph_vals);
  addInputPlotResponse(plotting_responses, observations_plot);
  const auto noise_value = likelihood_ptr->getNoise();
  // limit a 101-point three-sigma score sweep to at most 1000 observations
  if (!std::isfinite(noise_value) || noise_value <= 0 || observation_values.size() > 1000)
    return;
  std::vector<double> offset_values, score_values;
  auto model_values = observation_values;
  for (const auto i : make_range(101))
  {
    const auto value_offset = noise_value * (-3 + 6 * i / 100.0);
    for (const auto j : index_range(model_values))
      model_values[j] = observation_values[j] + value_offset;
    offset_values.push_back(value_offset);
    score_values.push_back(likelihood_ptr->function(model_values));
  }
  const auto y_axis_label = likelihood_ptr->type() == "LinearSum" ? "scaled squared error"
                            : likelihood_ptr->getParam<bool>("log_likelihood") ? "log likelihood"
                                                                               : "likelihood";
  auto profile_plot = buildInputPlot(object_name + " configured residual score",
                                     "common model-minus-observation offset",
                                     y_axis_label);
  addInputPlotCurve(moose_server,
                    profile_plot,
                    "all observations shifted by the same offset",
                    offset_values,
                    score_values,
                    true);
  addInputPlotResponse(plotting_responses, profile_plot);
}

#ifdef MOOSE_LIBTORCH_ENABLED
void
buildCovariancePlotResponses(MooseServer & moose_server,
                             wasp::DataArray & plotting_responses,
                             FEProblemBase & problem,
                             const std::string & object_name)
{
  // bound the kernel work to eight outputs and sixty-four input dimensions
  const auto * covariance_ptr =
      getInputPlotObject<CovarianceFunctionBase>(problem, object_name, "CovarianceFunction");
  if (!covariance_ptr)
    return;
  const auto num_outputs = covariance_ptr->getParam<unsigned int>("num_outputs");
  if (!num_outputs || num_outputs > 8)
    return;
  const CovarianceFunctionBase * coordinate_kernel = nullptr;
  std::vector<const CovarianceFunctionBase *> kernel_objects = {covariance_ptr};
  for (std::size_t i = 0; i < kernel_objects.size(); ++i)
  {
    const auto * kernel_ptr = kernel_objects[i];
    if (kernel_objects.size() > 9 || (i && kernel_ptr->type() == "LMC"))
      // bound composition to one LMC kernel and up to eight scalar kernels
      return;
    if (kernel_ptr->isParamValid("length_factor"))
    {
      if (!coordinate_kernel)
        coordinate_kernel = kernel_ptr;
      const auto & length_factors = kernel_ptr->getParam<std::vector<Real>>("length_factor");
      if (length_factors.empty() || length_factors.size() > 64 ||
          !std::all_of(length_factors.begin(),
                       length_factors.end(),
                       [](Real x) { return std::isfinite(x) && x > 0; }))
        return;
      for (const std::string parameter : {"signal_variance", "noise_variance"})
        if (kernel_ptr->isParamValid(parameter) &&
            (!std::isfinite(kernel_ptr->getParam<Real>(parameter)) ||
             kernel_ptr->getParam<Real>(parameter) < 0))
          return;
      if (kernel_ptr->isParamValid("p") && kernel_ptr->getParam<unsigned int>("p") > 8)
        return;
    }
    const auto & kernel_names = kernel_ptr->dependentCovarianceNames();
    if (kernel_ptr->type() == "LMC" &&
        kernel_ptr->getParam<unsigned int>("num_latent_funcs") > kernel_names.size())
      return;
    for (const auto & kernel_name : kernel_names)
    {
      const auto * child_kernel =
          getInputPlotObject<CovarianceFunctionBase>(problem, kernel_name, "CovarianceFunction");
      if (!child_kernel || child_kernel->getParam<unsigned int>("num_outputs") != 1)
        return;
      kernel_objects.push_back(child_kernel);
    }
  }
  if (!coordinate_kernel)
    return;
  const auto & length_factors = coordinate_kernel->getParam<std::vector<Real>>("length_factor");
  // sample 101 lags within three length factors without fitting the kernel
  constexpr int64_t num_points = 101;
  const auto num_dimensions = static_cast<int64_t>(length_factors.size());
  const auto tensor_options = torch::TensorOptions().dtype(at::kDouble);
  const auto origin_values = torch::zeros({1, num_dimensions}, tensor_options);
  torch::Tensor self_covariance;
  covariance_ptr->computeCovarianceMatrix(self_covariance, origin_values, origin_values, true);
  self_covariance = self_covariance.contiguous();
  if (self_covariance.dim() != 2 || self_covariance.size(0) != num_outputs ||
      self_covariance.size(1) != num_outputs)
    throw std::invalid_argument("Unexpected configured self-covariance dimensions");
  for (const auto output_component : make_range(std::min(2u, num_outputs)))
  {
    auto plot_object = buildInputPlot(object_name + " input covariance kernel, output " +
                                          std::to_string(output_component),
                                      "coordinate displacement (input units)",
                                      "configured covariance");
    for (const auto coordinate : make_range(std::min<std::size_t>(4, length_factors.size())))
    {
      auto input_values = torch::zeros({num_points, num_dimensions}, tensor_options);
      auto tensor_values = input_values.accessor<Real, 2>();
      std::vector<double> graph_keys;
      for (const auto i : make_range(num_points))
      {
        graph_keys.push_back(3 * length_factors[coordinate] * i / double(num_points - 1));
        tensor_values[i][coordinate] = graph_keys.back();
      }
      torch::Tensor covariance_values;
      covariance_ptr->computeCovarianceMatrix(
          covariance_values, input_values, origin_values, false);
      covariance_values = covariance_values.contiguous();
      if (covariance_values.dim() != 2 || covariance_values.size(0) != num_points * num_outputs ||
          covariance_values.size(1) != num_outputs)
        throw std::invalid_argument("Unexpected configured covariance dimensions");
      const auto * covariance_data = covariance_values.data_ptr<Real>();
      std::vector<double> graph_vals;
      for (const auto i : make_range(num_points))
        graph_vals.push_back(
            covariance_data[(output_component * num_points + i) * num_outputs + output_component]);
      addInputPlotCurve(moose_server,
                        plot_object,
                        "coordinate " + std::to_string(coordinate),
                        graph_keys,
                        graph_vals,
                        true);
    }
    addInputPlotCurve(
        moose_server,
        plot_object,
        "self covariance including diagonal noise",
        {0},
        {self_covariance.data_ptr<Real>()[output_component * num_outputs + output_component]});
    addInputPlotResponse(plotting_responses, plot_object);
  }
}
#endif

void
buildRewardPlotResponse(MooseServer & moose_server,
                        wasp::DataArray & plotting_responses,
                        FEProblemBase & problem,
                        const std::string & object_name)
{
  // get the configured reward function without running its control process
  if (!problem.hasFunction(object_name))
    return;
  const auto * reward_func =
      dynamic_cast<const ScaledAbsDifferenceDRLRewardFunction *>(&problem.getFunction(object_name));
  if (!reward_func)
    return;
  std::vector<double> graph_keys, graph_vals;
  // sample 201 signed errors on a unit interval to illustrate reward curve
  for (const auto i : make_range(201))
  {
    graph_keys.push_back(-1 + 2 * i / 200.0);
    graph_vals.push_back(reward_func->computeReward(graph_keys.back()));
  }
  auto plot_object = buildInputPlot(
      object_name + " input reward shape (unit error interval)", "signed tracking error", "reward");
  addInputPlotCurve(
      moose_server, plot_object, "configured reward rule", graph_keys, graph_vals, true);
  addInputPlotResponse(plotting_responses, plot_object);
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
  buildMorrisPlotResponse(moose_server, plotting_responses, *sampler_ptr, column_values);
  buildSobolPlotResponse(moose_server, plotting_responses, *sampler_ptr, column_values);
  buildQuadraturePlotResponse(
      moose_server, plotting_responses, *sampler_ptr, row_indices, plot_title);
  buildPolynomialPlotResponses(moose_server, plotting_responses, problem, *sampler_ptr);
  buildMarginalPlotResponses(moose_server, plotting_responses, column_values, plot_title);
  buildCorrelationPlotResponse(moose_server, plotting_responses, column_values, plot_title);
  if (column_values.size() > 1)
  {
    auto scatter_plot = buildInputPlot(plot_title + " paired columns", "column 0", "column 1");
    addInputPlotCurve(
        moose_server, scatter_plot, "input pairs", column_values[0], column_values[1]);
    addInputPlotResponse(plotting_responses, scatter_plot);
  }
}

void
buildParameterStudyPlotResponses(MooseServer & moose_server,
                                 wasp::DataArray & plotting_responses,
                                 FEProblemBase & problem,
                                 const std::string &)
{
  // find the sampler created by the configured parameter study action type
  for (const auto * action :
       problem.getMooseApp().actionWarehouse().getActions<ParameterStudyAction>())
  {
    buildSamplerPlotResponses(moose_server, plotting_responses, problem, action->samplerName());
    break;
  }
}
}

void
StochasticTools::registerInputPlots()
{
  MooseServer::registerInputPlotter("Samplers", "input-design", buildSamplerPlotResponses);
  MooseServer::registerInputPlotter(
      "ParameterStudy", "input-design", buildParameterStudyPlotResponses);
  MooseServer::registerInputPlotter("Samplers", "sampler-priors", buildPriorPlotResponses);
  MooseServer::registerInputPlotter("Samplers", "initial-proposals", buildProposalPlotResponses);
  MooseServer::registerInputPlotter("Distributions", "quantiles", buildQuantilePlotResponses);
  MooseServer::registerInputPlotter("Distributions", "tails", buildTailPlotResponse);
  MooseServer::registerInputPlotter("Distributions", "log-density", buildLogDensityPlotResponse);
  MooseServer::registerInputPlotter("Distributions", "comparison", buildComparisonPlotResponses);
  MooseServer::registerInputPlotter("Likelihood", "observations", buildLikelihoodPlotResponses);
  MooseServer::registerInputPlotter("Functions", "reward-shape", buildRewardPlotResponse);
#ifdef MOOSE_LIBTORCH_ENABLED
  MooseServer::registerInputPlotter("Covariance", "kernels", buildCovariancePlotResponses);
#endif
}
