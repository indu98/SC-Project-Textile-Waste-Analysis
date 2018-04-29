import csv
import random
import math
from sklearn.metrics import confusion_matrix,mean_squared_error
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

def loadCsv(filename):
	lines = csv.reader(open(filename, "rb"))
	dataset = list(lines)
	del dataset[0]
	maxY = dataset[0][6]
	maxY = float(maxY)
	maxY = maxY * 100
	minY = dataset[0][6]
	minY = float(minY)
	minY = minY * 100

	for i in range(len(dataset)):
		dataset[i] = [float(x) for x in dataset[i]]
		dataset[i][6] = dataset[i][6] * 100
		if dataset[i][6] > maxY:
			maxY = dataset[i][6]
		if dataset[i][6] < minY:
			minY = dataset[i][6]
		dataset[i][6] = int(dataset[i][6])
	maxY = int(maxY)
	minY = int(minY)
	return dataset, maxY, minY

def splitDataset(dataset, splitRatio):
	trainSize = int(len(dataset) * splitRatio)
	testSize = len(dataset) - trainSize
	testSet = []
	trainSet = list(dataset)
	while len(testSet) < testSize:
		index = random.randrange(len(trainSet))
		testSet.append(trainSet.pop(index))
	return [trainSet, testSet]

def separateByClass(dataset):
	separated = {}
	for i in range(len(dataset)):
		vector = dataset[i]
		if (vector[-1] not in separated):
			separated[vector[-1]] = []
		separated[vector[-1]].append(vector)
	return separated

def mean(numbers):
	if len(numbers) == 0:
		return 0
	return sum(numbers)/float(len(numbers))
from sklearn.metrics import confusion_matrix
def stdev(numbers):
	if len(numbers) == 0:
		return 0
	avg = mean(numbers)
	variance = sum([pow(x-avg,2) for x in numbers])/float(len(numbers))
	return math.sqrt(variance)

def summarize(dataset):
	summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
	del summaries[-1]
	return summaries

def summarizeByClass(dataset, maxY, minY):
	separated = separateByClass(dataset)
	summaries = {}
	for classValue, instances in separated.iteritems():
		summaries[classValue] = summarize(instances)
	for i in range(minY, maxY, 1):
		if i not in separated:
			summaries[i] = [(0,0),(0,0),(0,0),(0,0),(0,0),(0,0)]
	return summaries

def calculateProbability(x, mean, stdev):
	if stdev == 0:
		return 1
	exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
	return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent

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
	for i in range(len(testSet)):
		if testSet[i][-1] == predictions[i]:
			correct += 1
	return (correct/float(len(testSet))) * 100.0

def getConfusionMatrix(y_true, y_pred):

	y_exp = []
	for i in range(len(y_true)):
		#print(y_true[i][6], "," , y_pred[i])
		y_exp.append(y_true[i][6])
	y_exp = np.array(y_exp)
	print(y_exp.shape)
	y_pred = np.array(y_pred)
	print(y_pred.shape)

	confusion_mat = confusion_matrix(y_exp, y_pred, labels=None, sample_weight=None)
	return confusion_mat

def getPrecisionRecall(y_true, y_pred):

	y_exp = []
	for i in range(len(y_true)):
		#print(y_true[i][6], "," , y_pred[i])
		y_exp.append(y_true[i][6])
	y_exp = np.array(y_exp)
	y_pred = np.array(y_pred)

	prerecall = precision_recall_fscore_support(y_exp, y_pred, beta=1.0, labels=None, pos_label=1, average=None, warn_for=('precision', 'recall', 'f-score'), sample_weight=None)
	return prerecall

def get_mean_squared_error(y_true, y_pred):

	y_exp = []
	for i in range(len(y_true)):
		#print(y_true[i][6], "," , y_pred[i])
		y_exp.append(y_true[i][6])
	y_exp = np.array(y_exp)
	y_pred = np.array(y_pred)

	mse = mean_squared_error(y_exp, y_pred, sample_weight=None, multioutput='uniform_average')
	return float(mse)/10000.0

def main():	
	filename = 'sc_5.csv'
	dataset, maxY, minY = loadCsv(filename)
	print('Loaded data file {0} with {1} rows').format(filename, len(dataset))
	splitRatio = 0.67
	train, test = splitDataset(dataset, splitRatio)
	print('Split {0} rows into train with split ratio {1} and test with {2} with Split Ratio {3}').format(len(dataset), len(train), len(test), splitRatio)
	summaries = summarizeByClass(train, maxY, minY)
	predictions = getPredictions(summaries, test)
	print(len(predictions))
	print(len(test))
	accuracy = getAccuracy(test, predictions)
	print('Accuracy: {0}%').format(accuracy)
	print('Confusion  matrix')
	confusion_mat = getConfusionMatrix(test,predictions)
	print(confusion_mat)
	print('Precision recall')
	prerecall = getPrecisionRecall(test,predictions)
	mse = get_mean_squared_error(test, predictions)


	for item in prerecall:
		print(item)

	print("mean squared error", mse)
main()
