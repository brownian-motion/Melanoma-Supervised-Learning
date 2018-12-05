from decisiontree import DecisionTree
from decisiontree.DecisionTree import *

from images.featureextraction import FeatureExtractor as FE, plot
from images import *
from graphs import *

fe = FE.make_extractor()
# import samples, directory can change as needed
samples = Sample.get_samples("C:/Users/osage/Desktop/SL Short Project/ISIC-images/UDA-1")
test_samples = Sample.get_samples("C:/Users/osage/Desktop/SL Short Project/ISIC-images/UDA-2")
training_samples = samples[:3 * len(samples)//4]
valid_samples = samples[3 * len(samples)//4:]


def extract_examples(samples):
    examples = []  # fill with [X1,X2,...,Xn, Y]
    for sample in samples:
        # create grey-image
        grey_image = fe.convert_image_to_greyscale(sample)
        features = fe.get_features(grey_image)
        obs = 1 if sample.diagnosis == "melanoma" else 0
        features = numpy.append(features, obs)

        examples.append(features)
    return numpy.array(examples)


training_examples = extract_examples(training_samples)
valid_examples = extract_examples(valid_samples)

real_training = training_examples[:, -1]
real_valid = valid_examples[:, -1]

best_max_depth, best_entropy, best_tree = None, math.inf, None

for max_depth in range(2, 50):
    tree = growTree(training_examples, max_depth)
    valid_predict = tree.predict(valid_examples)
    valid_real = valid_examples[:, -1]
    avg_entropy = total_cross_entropy(valid_real, valid_real) / len(valid_real)

    if best_entropy > avg_entropy:
        best_max_depth, best_entropy, best_tree = max_depth, best_entropy, tree

test_examples = extract_examples(test_samples)
test_real = test_examples[:, -1 ].flatten()
test_predict = best_tree.predict(test_examples).flatten()

plot_roc_curve(test_real, test_predict, "Decision Tree: Test - ROC Curve")
plot_reliability_curve(test_real, test_predict)
plt.show()
plot_performance_diagram(test_real, test_predict)
plt.show()