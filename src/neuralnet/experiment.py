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
    net.add_layer(ConvolutionalLayer((10, 10), 0.01))
    net.add_layer(FullyConnectedLayer((1504 - 10 + 1) * (1128 - 10 + 1), 4, 0.001, 'sigmoid'))
    net.add_layer(FullyConnectedLayer(4, 2, 0.01, 'relu'))
    net.add_layer(SoftmaxLayer())

    random.shuffle(samples)

    entropies = []

    for sample_num in range(len(samples)):
        print("Processing sample %s" % sample_num, end=" ")
        sample = samples[sample_num]
        prediction = net.process(sample.get_grayscale_image()[:-1, :])

        if sample.diagnosis == "melanoma":
            net.backpropagate(to_column_vector([1, 0]))
            entropy = binary_cross_entropy(prediction[0], 1)
        else:
            net.backpropagate(to_column_vector([0, 1]))
            entropy = binary_cross_entropy(prediction[0], 0)
        entropies.append(entropy)
        print(entropy)

    plt.plot(entropies)
    plt.show()


if __name__ == '__main__':
    main()
