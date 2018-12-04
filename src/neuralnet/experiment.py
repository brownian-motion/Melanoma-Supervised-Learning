import random

from matplotlib.patches import Patch
from sklearn.metrics import roc_curve

from graphs.performance_diagram_stuff import plot_performance_diagram
from graphs.reliability_curve_stuff import plot_reliability_curve
from images import *
from neuralnet import *


def main():
    sample_images_iter, sample_ys = load_training_samples()
    sample_ys = numpy.array(sample_ys)
    net = make_neural_net()
    net.fit(sample_images_iter, sample_ys)
    yhats = net.predict(sample_images_iter)
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
    plt.title("Performance diagram")
    plt.show()

    plot_reliability_curve(sample_ys, yhats)
    plt.title("Reliability diagram")
    plt.show()

    fpr, tpr, _ = roc_curve(sample_ys, yhats)
    plt.plot([0, 1], [0, 1], '--')
    plt.plot(fpr, tpr, 'r-')
    plt.xlabel("Probability of false detection")
    plt.ylabel("Probability of detection")
    plt.title("ROC curve for CNN")
    plt.legend(handles=[Patch(color='red', label='CNN'), Patch(color='gray', label="Random predictor")])
    plt.show()


def load_training_samples():
    '''
    Returns two items:
    an iterable (currently a list, soon a generator expression / iterator for for loops) of images for each sample,
    and a list of observed classes for each sample (1 for melanoma, 0 otherwise)
    :return: an iterable of images (currently a list), and a list of classes for each image
    '''
    samples = Sample.get_samples("../../ISIC-images/UDA-1")[:20]
    # TODO: upsample melanoma
    random.shuffle(samples)
    # for sample in samples:
    #     print("(%4d, %4d) - %s" % (sample.image_dim[0], sample.image_dim[1], sample.diagnosis))

    # TODO: make this return a generator expression. That is, make the neural net accept iterables (that can be iterated only once)
    # this normalizes each color from 0~256 to -1~1, and makes the color channel the primary axis.
    sample_images = [move_color_channel_to_first_axis(s.get_image() / 128 - 1) for s in samples]
    sample_ys = [1 if s.diagnosis == "melanoma" else 0 for s in samples]
    return sample_images, sample_ys


def make_neural_net():
    net = SimpleNeuralBinaryClassifier()
    # net.add_layer(MinpoolLayer((2, 2), overlap_tiles=False))  # take out border
    # net.add_layer(MeanpoolLayer((2, 2), overlap_tiles=False))  # make the image smaller
    net.add_layer(ConvolutionalLayer((5, 5), num_filters=6, training_rate=0.001))
    net.add_layer(MeanpoolLayer((4, 4), overlap_tiles=False))
    net.add_layer(ConvolutionalLayer((3, 3), num_filters=4, training_rate=0.001))
    net.add_layer(MeanpoolLayer((3, 3), overlap_tiles=False))
    net.add_layer(ConvolutionalLayer((4, 4), num_filters=4, training_rate=0.001))
    net.add_layer(MeanpoolLayer((4, 4), overlap_tiles=False))
    dim = (((STANDARD_IMAGE_LENGTH - 5 + 1) // 4 - 3 + 1) // 3 - 4 + 1) // 4
    num_pixels = 3 * 6 * 4 * 4 * dim * dim
    net.add_layer(FullyConnectedLayer(num_pixels, 12, training_rate=0.1, activation_function_name='relu'))
    net.add_layer(FullyConnectedLayer(12, 12, training_rate=0.01, activation_function_name='relu'))
    net.add_layer(FullyConnectedLayer(12, 1, training_rate=0.01, activation_function_name='relu'))
    net.add_layer(SigmoidLayer())
    return net


# plt.plot(pos_sample_nums, pos_entropies, 'r-', neg_sample_nums, neg_entropies, 'b-')
# plt.legend(handles=[Patch(color='red', label='Melanoma'), Patch(color='blue', label='Not Melanoma')])
# plt.show()


if __name__ == '__main__':
    main()
