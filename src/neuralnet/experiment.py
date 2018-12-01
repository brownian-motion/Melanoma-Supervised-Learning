import random

import matplotlib.pyplot as plt

from cross_entropy import binary_cross_entropy
from images import *
from neuralnet import *

APPROVED_SIZE = (1504, 1129)


def main():
    samples = Sample.get_samples("../../ISIC-images/UDA-1")
    samples = [sample for sample in samples if sample.image_dim == APPROVED_SIZE]
    # for sample in samples:
    #     print("(%4d, %4d) - %s" % (sample.image_dim[0], sample.image_dim[1], sample.diagnosis))
    #
    net = LinearNeuralNetwork()
    # net.add_layer(MinpoolLayer((2, 2), overlap_tiles=False))  # take out border
    # net.add_layer(MeanpoolLayer((2, 2), overlap_tiles=False))  # make the image smaller
    net.add_layer(ConvolutionalLayer((3, 3), 0.01))
    net.add_layer(FullyConnectedLayer((1129 - 3 + 1) * (1129 - 3 + 1), 4, 0.001, 'sigmoid'))
    net.add_layer(FullyConnectedLayer(4, 2, 0.01, 'relu'))
    net.add_layer(SoftmaxLayer())

    random.shuffle(samples)

    pos_entropies = []
    pos_sample_nums = []
    neg_entropies = []
    neg_sample_nums = []

    for sample_num in range(len(samples) // 2):
        sample = samples[sample_num]
        print("Processing sample %s (%s)" % (sample_num, sample.diagnosis), end=" ")
        prediction = net.process(sample.get_grayscale_image())

        if sample.diagnosis == "melanoma":
            net.backpropagate(to_column_vector([1, 0]))
            entropy = binary_cross_entropy(prediction[0], 1)
            pos_entropies.append(entropy)
            pos_sample_nums.append(sample_num)
        else:
            net.backpropagate(to_column_vector([0, 1]))
            entropy = binary_cross_entropy(prediction[0], 0)
            neg_entropies.append(entropy)
            neg_sample_nums.append(sample_num)
        print(entropy)

    plt.plot(pos_sample_nums, pos_entropies, 'r-', neg_sample_nums, neg_entropies, 'b-')
    plt.show()


if __name__ == '__main__':
    main()
