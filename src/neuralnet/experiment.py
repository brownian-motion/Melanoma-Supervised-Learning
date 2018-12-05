import random

from graphs import *
from images import *
from neuralnet import *


def main():
    samples = load_training_samples()
    train_samples, valid_samples = samples[: 3 * len(samples) // 4], samples[3 * len(samples) // 4:]
    train_ys = numpy.array(get_sample_observations(train_samples))
    valid_ys = numpy.array(get_sample_observations(valid_samples))

    net = make_neural_net()
    net.fit(get_sample_images_gen(train_samples), train_ys)  # Takes a few minutes!!

    show_conv_net_filter_output(net, samples)

    valid_yhats = net.predict(get_sample_images_gen(valid_samples))

    # show_prediction_offsets(valid_yhats, valid_ys)
    # print_results(valid_ys, valid_yhats)

    plot_performance_diagram(valid_ys, valid_yhats)
    plt.title("Performance diagram for CNN")
    plt.show()

    plot_reliability_curve(valid_ys, valid_yhats)
    plt.title("Reliability diagram for CNN")
    plt.show()

    plot_roc_curve(valid_ys, valid_yhats, "CNN")


def print_results(true_observations, predictions):
    print("Observations:")
    print(true_observations)
    print("Predictions:")
    print(predictions)


def show_prediction_offsets(yhats, ys):
    for i in range(len(yhats)):
        plt.plot([i, i], [ys[i], yhats[i]], 'r-' if ys[i] == 1 else 'b-')
    plt.title("Predictions vs. observations")
    plt.xlabel("Sample number")
    plt.ylabel("Confidence sample is melanoma")
    plt.show()


def show_conv_net_filter_output(net, samples):
    net._process(next(get_sample_images_gen(samples)), remember_inputs=True)  # to populate layer._last_outputs
    for i in range(len(net.layers)):
        layer = net.layers[i]
        if type(layer) is ConvolutionalLayer:
            filter_outs = layer._last_outputs[-1]
            filter_img = filter_outs[0]
            plt.imshow((filter_img + 1) * 128, cmap='gray')
            plt.title("Last output from trained conv layer %d, filter %d" % (i, 0))
            plt.show()


def get_sample_images_gen(samples):
    """
    Returns a generator (for for loops) of images for each sample, appropriate for this experiment's neural net
    :return: an iterable of images from each sample, appropriate for this experiment's neural net
    """
    # TODO: upsample melanoma, maybe somewhere else tho
    # for sample in samples:
    #     print("(%4d, %4d) - %s" % (sample.image_dim[0], sample.image_dim[1], sample.diagnosis))
    # this normalizes each color from 0~256 to -1~1, and makes the color channel the primary axis.
    return (s.get_grayscale_image() / 128 - 1 for s in samples)
    # return (move_color_channel_to_first_axis(s.get_image() / 128 - 1) for s in samples)


def load_training_samples():
    samples = Sample.get_samples("C:/Users/JJ/Documents/GitHub/Melanoma-Supervised-Learning/ISIC-images/UDA-1")
    random.shuffle(samples)
    return samples


def get_sample_observations(samples):
    sample_ys = [1 if s.diagnosis == "melanoma" else 0 for s in samples]
    return sample_ys


def make_neural_net():
    net = SimpleNeuralBinaryClassifier()
    dim = STANDARD_IMAGE_LENGTH
    num_layers = 1
    net.add_layer(ConvolutionalLayer((5, 5), num_filters=5, training_rate=0.001))
    dim = dim - 5 + 1
    num_layers *= 5
    net.add_layer(MeanpoolLayer((4, 4), overlap_tiles=False))
    dim //= 4
    net.add_layer(ConvolutionalLayer((3, 3), num_filters=3, training_rate=0.001))
    dim = dim - 3 + 1
    num_layers *= 3
    net.add_layer(MeanpoolLayer((3, 3), overlap_tiles=False))
    dim //= 3
    net.add_layer(ConvolutionalLayer((4, 4), num_filters=4, training_rate=0.001))
    dim = dim - 4 + 1
    num_layers *= 4
    net.add_layer(MeanpoolLayer((4, 4), overlap_tiles=False))
    dim //= 4
    net.add_layer(ConvolutionalLayer((4, 4), num_filters=4, training_rate=0.001))
    dim = dim - 4 + 1
    num_layers *= 4
    net.add_layer(MeanpoolLayer((4, 4), overlap_tiles=False))
    dim //= 4
    num_pixels = num_layers * dim * dim
    net.add_layer(FullyConnectedLayer(num_pixels, 12, training_rate=0.01, activation_function_name='relu'))
    net.add_layer(FullyConnectedLayer(12, 12, training_rate=0.01, activation_function_name='relu'))
    net.add_layer(FullyConnectedLayer(12, 1, training_rate=0.01,
                                      activation_function_name='identity'))  # to allow for full range of inputs into sigmoid output layer
    net.add_layer(SigmoidLayer())
    return net


# plt.plot(pos_sample_nums, pos_entropies, 'r-', neg_sample_nums, neg_entropies, 'b-')
# plt.legend(handles=[Patch(color='red', label='Melanoma'), Patch(color='blue', label='Not Melanoma')])
# plt.show()


if __name__ == '__main__':
    main()
