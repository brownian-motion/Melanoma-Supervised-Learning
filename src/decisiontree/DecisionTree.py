import math
import numpy as np
from cross_entropy import total_cross_entropy


class LeafNode:
    def __init__(self, labels):
        self.label = getAverageProbablity(labels)

    def predict(self, arr_examples):
        if type(arr_examples) is np.ndarray and len(arr_examples.shape) == 2:
            return np.full(len(arr_examples),self.label)
        else:
            return [self.label]

class BranchNode:
    def __init__(self, best_column, best_threshold):
        self.best_column = best_column
        self.best_threshold = best_threshold
        self.left = None
        self.right = None

    def predict(self, arr_examples):
        predictions = []
        for row in arr_examples:
            if is_row_below_threshold(row,self.best_column, self.best_threshold):
                prediction = self.left.predict([row])
            else:
                prediction = self.right.predict([row])
            for p in prediction:
                predictions.append(p)
        return np.array(predictions)

def is_row_below_threshold(row, col_number, threshold):
    return row[col_number] < threshold

def get_rows_below_threshold(mat, col_number, threshold):
    row_numbers_below_threshold = np.asarray(np.where(mat[:, col_number] < threshold)).flatten()
    return mat[row_numbers_below_threshold, :]


def get_rows_above_threshold(mat, col_number, threshold):
    row_numbers_above_threshold = np.asarray(np.where(mat[:, col_number] >= threshold)).flatten()
    return mat[row_numbers_above_threshold]


# create method to return a leaf node given label
def getLabelLeafNode(label):
    return LeafNode([label])


# Create method to create a leaf node given multiple labels
def createNewLeafNode(labels):
    return LeafNode(labels)


# Create branch node using the best column and best threshold
def createBranchNode(best_column, best_threshold):
    return BranchNode(best_column, best_threshold)


# Compute entropies
def get_entropy(mat):
    obs = mat[:, -1]
    prediction = getAverageProbablity(obs)
    yhat = np.full(obs.size, prediction)

    return total_cross_entropy(yhat, obs)


def getAverageProbablity(obs):
    prob_one = np.average(obs)
    prediction = prob_one
    return prediction


def chooseBestAttribute(arr_examples):
    best_column, best_threshold, best_entropy = None, None, math.inf
    num_cols = arr_examples.shape[1]
    initial_entropy = get_entropy(arr_examples)

    for col in range(num_cols - 1):
        uniq = np.unique(arr_examples[:, col])
        for threshold in uniq:
            left = get_rows_below_threshold(arr_examples, col, threshold)
            right = get_rows_above_threshold(arr_examples, col, threshold)

            entropy = get_entropy(left) + get_entropy(right)

            if best_column is None:
                # compare against initial entropy
                if entropy < initial_entropy:
                    best_column, best_threshold, best_entropy = col, threshold, entropy
            else:
                # compare against against best_entropy
                if entropy < best_entropy:
                    best_column, best_threshold, best_entropy = col, threshold, entropy

    if best_column is None:
        return None, None
    else:
        return best_column, best_threshold


# Split into left and right side of a tree
def split(arr_examples, best_column, best_treshold):
    left = get_rows_below_threshold(arr_examples, best_column, best_treshold)
    right = get_rows_above_threshold(arr_examples, best_column, best_treshold)
    return left, right


def growTree(arr_examples, max_depth=5):
    if len(arr_examples) == 0:
        raise ValueError()
    if np.all(arr_examples[:, -1] == arr_examples[0, -1]):
        return getLabelLeafNode(arr_examples[0, -1])
    if arr_examples.shape[1] == 1:
        raise ValueError()
    if max_depth <= 0:
        return createNewLeafNode(arr_examples[:, -1])
    else:
        best_column, best_threshold = chooseBestAttribute(arr_examples)
        if best_column is None:
            return createNewLeafNode(arr_examples[:, -1])
        tree = createBranchNode(best_column, best_threshold)
        # subset into left and right nodes
        left, right = split(arr_examples, best_column, best_threshold)
        tree.left = growTree(left, max_depth - 1)
        tree.right = growTree(right, max_depth - 1)
        return tree
