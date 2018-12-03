# based on work from 
# https://machinelearningmastery.com/naive-bayes-classifier-scratch-python/

import csv
import random

# load a csv file
def loadData(file):
    data = list()
    with open(file) as csv:
        reader = csv.reader(csv)
        for row in reader:
            data.append(row)
    return data

# splits data into training and testing sets
def splitData(dataset, ratio):
    train_size = int(len(dataset) * ratio)
    train_data = []
    test_data = list(dataset)
    while len(train_data) < train_size:
        index = random.randrange(len(test_data))
        train_data.append(test_data.pop(index))
    
    return [train_data, test_data]

# returns a map of class values to lists of samples
def classifier(data):
    separated = {}
    for i in range(len(data)):
        sample = data[i]
        if sample[-1] not in separated:
            separated[sample[-1]] = []
        separated[sample[-1]].append(sample)
    return separated

# returns a vector of the mean and stdev
def summarize(data):
    summaries = [(mean(attr), stdev(attr)) for attr in zip(*data)]
    del summaries[-1]
    return summaries

def summarizeByClass(data):
    separated = classifier(data)
    summaries = {}
    for classVal, instances in separated.iteritems():
        summaries[classVal] = summarize(instances)
    return summaries

def probability(x, mean, stdev):
    exponent = math.exp(-(math.pow(x-mean,2) / (2*math.pow(stdev,2))))
    return exponent * (1 / (math.sqrt(2*math.pi) * stdev))

def calculateClassProbabilities(summaries, inputVector):
	probabilities = {}
	for classValue, classSummaries in summaries.iteritems():
		probabilities[classValue] = 1
		for i in range(len(classSummaries)):
			mean, stdev = classSummaries[i]
			x = inputVector[i]
			probabilities[classValue] *= calculateProbability(x, mean, stdev)
	return probabilities

def predict(summaries, inputVector):
	probabilities = calculateClassProbabilities(summaries, inputVector)
	bestLabel, bestProb = None, -1
	for classValue, probability in probabilities.iteritems():
		if bestLabel is None or probability > bestProb:
			bestProb = probability
			bestLabel = classValue
	return bestLabel

def getPredictions(summaries, testSet):
	predictions = []
	for i in range(len(testSet)):
		result = predict(summaries, testSet[i])
		predictions.append(result)
	return predictions

def getAccuracy(testSet, predictions):
	correct = 0
	for x in range(len(testSet)):
		if testSet[x][-1] == predictions[x]:
			correct += 1
	return (correct/float(len(testSet))) * 100.0