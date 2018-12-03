import random

from images import *
from neuralnet import *


def main():
    samples = Sample.get_samples("../../ISIC-images/UDA-1")

    # for sample in samples:
    #     print("(%4d, %4d) - %s" % (sample.image_dim[0], sample.image_dim[1], sample.diagnosis))
    #
    net = SimpleNeuralBinaryClassifier()
    # net.add_layer(MinpoolLayer((2, 2), overlap_tiles=False))  # take out border
    # net.add_layer(MeanpoolLayer((2, 2), overlap_tiles=False))  # make the image smaller
    net.add_layer(ConvolutionalLayer((5, 5), num_filters=6, training_rate=0.5))
    net.add_layer(MeanpoolLayer((4, 4), overlap_tiles=False))
    net.add_layer(ConvolutionalLayer((3, 3), num_filters=4, training_rate=0.5))
    net.add_layer(MeanpoolLayer((3, 3), overlap_tiles=False))
    dim = ((STANDARD_IMAGE_LENGTH - 5 + 1) // 4 - 3 + 1) // 3
    net.add_layer(FullyConnectedLayer(6 * 4 * dim * dim, 12, training_rate=0.1, activation_function_name='relu'))
    net.add_layer(FullyConnectedLayer(12, 12, training_rate=0.01, activation_function_name='relu'))
    net.add_layer(FullyConnectedLayer(12, 2, training_rate=0.01, activation_function_name='relu'))
    net.add_layer(SoftmaxLayer())

    random.shuffle(samples)

    pos_entropies = []
    pos_sample_nums = []
    neg_entropies = []
    neg_sample_nums = []

    sample_images = [s.get_grayscale_image() / 256 for s in samples]
    sample_ys = [1 if s.diagnosis == "melanoma" else 0 for s in samples]

    net.fit(sample_images, sample_ys)
    yhats = net.predict(sample_images)[:, 1]
    for i in range(len(yhats)):
        plt.plot([i, i], [sample_ys[i], yhats[i]], 'r-' if sample_ys[i] == 1 else 'b-')
    plt.show()


# plt.plot(pos_sample_nums, pos_entropies, 'r-', neg_sample_nums, neg_entropies, 'b-')
# plt.legend(handles=[Patch(color='red', label='Melanoma'), Patch(color='blue', label='Not Melanoma')])
# plt.show()


if __name__ == '__main__':
    main()
