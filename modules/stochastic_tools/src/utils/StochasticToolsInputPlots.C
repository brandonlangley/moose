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
#include "ConstantReporter.h"
#include "TheWarehouse.h"
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
}

void
StochasticTools::registerInputPlots()
{
  MooseServer::registerInputPlotter("Samplers", "input-design", buildSamplerPlotResponses);
  MooseServer::registerInputPlotter("Distributions", "quantiles", buildQuantilePlotResponses);
  MooseServer::registerInputPlotter("Distributions", "tails", buildTailPlotResponse);
  MooseServer::registerInputPlotter("Distributions", "log-density", buildLogDensityPlotResponse);
}
