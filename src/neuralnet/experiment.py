import random

from graphs import *
from images import *
from neuralnet import *


def main():
    samples = load_training_samples()
    sample_ys = numpy.array(get_sample_observations(samples))
    net = make_neural_net()
    net.fit(get_training_sample_images_gen(samples), sample_ys)
    yhats = net.predict(get_training_sample_images_gen(samples))
    for i in range(len(yhats)):
        plt.plot([i, i], [sample_ys[i], yhats[i]], 'r-' if sample_ys[i] == 1 else 'b-')
    plt.title("Predictions vs. observations")
    plt.xlabel("Sample number")
    plt.ylabel("Confidence sample is melanoma")
    plt.show()

    print("Observations:")
    print(sample_ys)
    print("Predictions:")
    print(yhats)

    plot_performance_diagram(sample_ys, yhats)
    plt.title("Performance diagram for CNN")
    plt.show()

    plot_reliability_curve(sample_ys, yhats)
    plt.title("Reliability diagram for CNN")
    plt.show()

    plot_roc_curve(sample_ys, yhats, "CNN")


def load_images_iter(samples):
    for s in samples:
        yield move_color_channel_to_first_axis(s.get_image() / 128 - 1)


def get_training_sample_images_gen(samples):
    """
    Returns a generator (for for loops) of images for each sample, appropriate for this experiment's neural net
    :return: an iterable of images from each sample, appropriate for this experiment's neural net
    """
    # TODO: upsample melanoma
    # for sample in samples:
    #     print("(%4d, %4d) - %s" % (sample.image_dim[0], sample.image_dim[1], sample.diagnosis))
    # this normalizes each color from 0~256 to -1~1, and makes the color channel the primary axis.
    return (move_color_channel_to_first_axis(s.get_image() / 128 - 1) for s in samples)


def load_training_samples():
    samples = Sample.get_samples("../../ISIC-images/UDA-1")
    random.shuffle(samples)
    return samples[:3]


def get_sample_observations(samples):
    sample_ys = [1 if s.diagnosis == "melanoma" else 0 for s in samples]
    return sample_ys


def make_neural_net():
    net = SimpleNeuralBinaryClassifier()
    dim = STANDARD_IMAGE_LENGTH
    num_layers = 3
    net.add_layer(ConvolutionalLayer((5, 5), num_filters=6, training_rate=0.01))
    dim = dim - 5 + 1
    num_layers *= 6
    net.add_layer(MeanpoolLayer((4, 4), overlap_tiles=False))
    dim = dim // 4
    net.add_layer(ConvolutionalLayer((3, 3), num_filters=4, training_rate=0.01))
    dim = dim - 3 + 1
    num_layers *= 4
    net.add_layer(MeanpoolLayer((3, 3), overlap_tiles=False))
    dim = dim // 3
    net.add_layer(ConvolutionalLayer((4, 4), num_filters=4, training_rate=0.01))
    dim = dim - 4 + 1
    num_layers *= 4
    net.add_layer(MeanpoolLayer((4, 4), overlap_tiles=False))
    dim = dim // 4
    net.add_layer(ConvolutionalLayer((4, 4), num_filters=4, training_rate=0.01))
    dim = dim - 4 + 1
    num_layers *= 4
    net.add_layer(MeanpoolLayer((4, 4), overlap_tiles=False))
    dim = dim // 4
    num_pixels = num_layers * dim * dim
    net.add_layer(FullyConnectedLayer(num_pixels, 12, training_rate=0.1, activation_function_name='relu'))
    net.add_layer(FullyConnectedLayer(12, 12, training_rate=0.1, activation_function_name='relu'))
    net.add_layer(FullyConnectedLayer(12, 1, training_rate=0.1, activation_function_name='relu'))
    # todo: make dense layer activation function for sigmoid -1~1 so that output can be below .5 ever
    net.add_layer(SigmoidLayer())
    return net


# plt.plot(pos_sample_nums, pos_entropies, 'r-', neg_sample_nums, neg_entropies, 'b-')
# plt.legend(handles=[Patch(color='red', label='Melanoma'), Patch(color='blue', label='Not Melanoma')])
# plt.show()


if __name__ == '__main__':
    main()
