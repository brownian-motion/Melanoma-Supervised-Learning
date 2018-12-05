"""Methods for plotting reliability curve."""

import matplotlib.pyplot as pyplot
import numpy

DEFAULT_NUM_BINS = 20
DEFAULT_LINE_COLOUR = numpy.array([228., 26., 28.]) / 255
DEFAULT_LINE_WIDTH = 3.
DEFAULT_PERFECT_LINE_COLOUR = numpy.array([152., 152., 152.]) / 255
DEFAULT_PERFECT_LINE_WIDTH = 2.

FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15

FONT_SIZE = 30
pyplot.rc('font', size=FONT_SIZE)
pyplot.rc('axes', titlesize=FONT_SIZE)
pyplot.rc('axes', labelsize=FONT_SIZE)
pyplot.rc('xtick', labelsize=FONT_SIZE)
pyplot.rc('ytick', labelsize=FONT_SIZE)
pyplot.rc('legend', fontsize=FONT_SIZE)
pyplot.rc('figure', titlesize=FONT_SIZE)


def _get_histogram(input_values, num_bins, min_value, max_value):
    """Creates histogram with uniform bin-spacing.

    N = number of input values
    K = number of bins

    :param input_values: length-N numpy array.
    :param num_bins: Number of bins.
    :param min_value: Minimum value for histogram.  Any input value <
        `min_value` will be assigned to the first bin.
    :param max_value: Max value for histogram.  Any input value >
        `max_value` will be assigned to the last bin.
    :return: input_to_bin_indices: length-N numpy array of bin indices.  If
        input_values[i] = j, the [i]th input value belongs to the [j]th bin.
    """

    bin_cutoffs = numpy.linspace(min_value, max_value, num=num_bins + 1)
    input_to_bin_indices = numpy.digitize(
        input_values, bin_cutoffs, right=False) - 1
    input_to_bin_indices[input_to_bin_indices < 0] = 0
    input_to_bin_indices[input_to_bin_indices > num_bins - 1] = num_bins - 1

    return input_to_bin_indices


def _get_points_in_relia_curve(
        observed_labels, forecast_probabilities, num_bins):
    """Creates points for reliability curve.

    N = number of examples
    K = number of bins

    :param observed_labels: length-N numpy array of observed labels (must be
        integers in 0...1).
    :param forecast_probabilities: length-N numpy array with forecast
        probabilities of label = 1.
    :param num_bins: Number of bins.
    :return: mean_observation_by_bin: length-K numpy array with mean observation
        (frequency of label = 1) for each bin.
    :return: mean_forecast_by_bin: length-K numpy array with mean forecast
        probability (probability of label = 1) for each bin.
    """

    assert numpy.all(numpy.logical_or(
        observed_labels == 0, observed_labels == 1))
    assert numpy.all(numpy.logical_and(
        forecast_probabilities >= 0, forecast_probabilities <= 1))
    assert num_bins > 1

    input_to_bin_indices = _get_histogram(
        input_values=forecast_probabilities, num_bins=num_bins, min_value=0.,
        max_value=1.)

    mean_observation_by_bin = numpy.full(num_bins, numpy.nan)
    mean_forecast_by_bin = numpy.full(num_bins, numpy.nan)

    for k in range(num_bins):
        these_example_indices = numpy.where(input_to_bin_indices == k)[0]

        mean_observation_by_bin[k] = numpy.mean(
            observed_labels[these_example_indices].astype(float)
        )
        mean_forecast_by_bin[k] = numpy.mean(
            forecast_probabilities[these_example_indices])

    return mean_observation_by_bin, mean_forecast_by_bin


def plot_reliability_curve(
        observed_labels, forecast_probabilities, num_bins=DEFAULT_NUM_BINS,
        line_colour=DEFAULT_LINE_COLOUR, line_width=DEFAULT_LINE_WIDTH,
        perfect_line_colour=DEFAULT_PERFECT_LINE_COLOUR,
        perfect_line_width=DEFAULT_PERFECT_LINE_WIDTH):
    """Plots reliability curve.

    N = number of examples

    :param observed_labels: length-N numpy array of observed labels (must be
        integers in 0...1).
    :param forecast_probabilities: length-N numpy array with forecast
        probabilities of label = 1.
    :param num_bins: Number of bins.
    :param line_colour: Colour (in any format accepted by `matplotlib.colors`).
    :param line_width: Line width (real positive number).
    :param perfect_line_colour: Colour of reference line (reliability curve with
        reliability = 0).
    :param perfect_line_width: Width of reference line.
    """

    mean_observation_by_bin, mean_forecast_by_bin = _get_points_in_relia_curve(
        observed_labels=observed_labels,
        forecast_probabilities=forecast_probabilities, num_bins=num_bins)

    _, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )

    perfect_x_coords = numpy.array([0, 1])
    perfect_y_coords = numpy.array([0, 1])
    axes_object.plot(
        perfect_x_coords, perfect_y_coords, color=perfect_line_colour,
        linestyle='dashed', linewidth=perfect_line_width)

    not_nan_indices = numpy.where(numpy.invert(numpy.logical_or(
        numpy.isnan(mean_forecast_by_bin),
        numpy.isnan(mean_observation_by_bin)
    )))[0]

    axes_object.plot(
        mean_forecast_by_bin[not_nan_indices],
        mean_observation_by_bin[not_nan_indices], color=line_colour,
        linestyle='solid', linewidth=line_width)

    axes_object.set_xlabel('Forecast probability')
    axes_object.set_ylabel('Conditional event frequency')
    axes_object.set_xlim(0., 1.)
    axes_object.set_ylim(0., 1.)


if __name__ == '__main__':
    observed_labels = numpy.random.random_integers(low=0, high=1, size=1000)

    forecast_probabilities = numpy.full(observed_labels.size, numpy.nan)
    negative_indices = numpy.where(observed_labels == 0)[0]
    forecast_probabilities[negative_indices] = numpy.random.normal(
        loc=0., scale=0.5, size=len(negative_indices))

    positive_indices = numpy.where(observed_labels == 1)[0]
    forecast_probabilities[positive_indices] = 1. - numpy.random.normal(
        loc=0., scale=0.5, size=len(positive_indices))

    forecast_probabilities[forecast_probabilities < 0.] = 0.
    forecast_probabilities[forecast_probabilities > 1.] = 1.

    plot_reliability_curve(
        observed_labels=observed_labels,
        forecast_probabilities=forecast_probabilities)

    pyplot.show()
