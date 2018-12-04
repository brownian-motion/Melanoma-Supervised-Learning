from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from sklearn.metrics import roc_curve

from graphs.performance_diagram_stuff import plot_performance_diagram
from graphs.reliability_curve_stuff import plot_reliability_curve


def plot_roc_curve(true_ys, forecast_predictions, model_name):
    fpr, tpr, _ = roc_curve(true_ys, forecast_predictions)
    plt.plot([0, 1], [0, 1], '--')
    plt.plot(fpr, tpr, 'r-')
    plt.xlabel("Probability of false detection")
    plt.ylabel("Probability of detection")
    plt.title("ROC curve for " + model_name)
    plt.legend(handles=[Patch(color='red', label=model_name), Patch(color='gray', label="Random predictor")])
    plt.show()
