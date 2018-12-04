import naive_bayes as nb

filename = 'sample_data.csv'
splitRatio = 0.67
dataset = nb.loadCsv(filename)
trainingSet, testSet = nb.splitData(dataset, splitRatio)
print('Split {0} rows into train={1} and test={2} rows'.format(len(dataset), len(trainingSet), len(testSet)))
summaries = nb.summarizeByClass(trainingSet)
predictions = nb.getPredictions(summaries, testSet)
accuracy = nb.getAccuracy(testSet, predictions)
print('Accuracy: {0}%'.format(accuracy))