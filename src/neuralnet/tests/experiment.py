import unittest

from neuralnet.experiment import *


class ExperimentDataIntegrationTest(unittest.TestCase):
    def test_net_learns_to_increase_prediction_given_only_melanoma(self):
        samples = [s for s in load_training_samples() if s.diagnosis == "melanoma"]
        samples = samples[:10]  # at most 10, for fast processing
        ys = get_sample_observations(samples)
        net = make_neural_net()
        initial_predictions = net.predict(get_sample_images_gen(samples))
        net.fit(get_sample_images_gen(samples), ys)
        final_predictions = net.predict(get_sample_images_gen(samples))

        print("Changes were %s" % ["%.1E" % (final_predictions[i] - initial_predictions[i])
                                   for i in range(len(initial_predictions))])
        print("Average change was %.1E" % numpy.average(final_predictions - initial_predictions))
        expected_increase = 0.005
        for i in range(len(initial_predictions)):
            self.assertGreater(final_predictions[i], initial_predictions[i] + expected_increase,
                               msg="Given only positive examples, net should "
                                   "learn to increase its prediction for sample %d by %.1E" % (i, expected_increase))

    def test_net_learns_to_decrease_prediction_given_only_benign(self):
        samples = [s for s in load_training_samples() if s.diagnosis != "melanoma"]
        samples = samples[:10]  # at most 10, for fast processing
        ys = get_sample_observations(samples)
        net = make_neural_net()
        initial_predictions = net.predict(get_sample_images_gen(samples))
        net.fit(get_sample_images_gen(samples), ys)
        final_predictions = net.predict(get_sample_images_gen(samples))

        print("Changes were %s" % ["%.1E" % (final_predictions[i] - initial_predictions[i])
                                   for i in range(len(initial_predictions))])
        print("Average change was %.1E" % numpy.average(final_predictions - initial_predictions))
        expected_decrease = 0.005
        for i in range(len(initial_predictions)):
            self.assertLess(final_predictions[i], initial_predictions[i] - expected_decrease,
                            msg="Given only negative examples, net should "
                                "learn to decrease its prediction for sample %d by %.1E" % (i, expected_decrease))

    def test_net_learns_to_increase_prediction_given_evenly_mixed_group(self):
        pos_samples = [s for s in load_training_samples() if s.diagnosis == "melanoma"]
        neg_samples = [s for s in load_training_samples() if s.diagnosis != "melanoma"]
        samples = pos_samples[:5] + neg_samples[:5]  # at most 10, for fast processing
        numpy.random.shuffle(samples)
        ys = get_sample_observations(samples)
        net = make_neural_net()
        initial_predictions = net.predict(get_sample_images_gen(samples))
        net.fit(get_sample_images_gen(samples), ys)
        final_predictions = net.predict(get_sample_images_gen(samples))

        print("True ys were %s" % ["%9d" % y for y in ys])
        print("Changes were %s" % ["%9.1E" % (final_predictions[i] - initial_predictions[i])
                                   for i in range(len(initial_predictions))])
        expected_change = 0.005
        for i in range(len(initial_predictions)):
            if samples[i].diagnosis == "melanoma":
                self.assertGreater(final_predictions[i], initial_predictions[i] + expected_change,
                                   msg="Given evenly mixed examples, net should "
                                       "learn to increase its prediction for positive sample %d by %.1E" % (
                                           i, expected_change))
            else:
                self.assertLess(final_predictions[i], initial_predictions[i] - expected_change,
                                msg="Given only evenly mixed examples, net should "
                                    "learn to decrease its prediction for negative sample %d by %.1E" % (
                                        i, expected_change))
