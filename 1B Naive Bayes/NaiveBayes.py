"""
ML Assignment 1
Naive Bayes Classifier

Team Details:
1. Rohan Maheshwari		2017B4A70965H
2. Keshav Kabra	
3. Meganaa Reddy
""" 

import math
import random
import os
import re
import numpy as np
import pandas as pd


def createInvertedIndex(data):
	"""
	Creating the inverted index. Dictioary from word to data items it appears in.

	Parameters:  
	data(list): List containing the data points  

	Returns:  
	Dictionary: The inverted index
	"""
	inverted_index = {}
	for i,line in enumerate(data):
		for word in line.split():
			if len(word)>=3 and word!="and" and word!="the":
				if inverted_index.get(word,False):  #value to return if specified key DNE
					inverted_index[word].append(i)
				else:
					inverted_index[word] = [i]
	return inverted_index


def checkTrainSet(arr, low, high, x):
	"""
	Do binary search to find value in array
	"""
	if high >= low:

		mid = (high + low) // 2

		if arr[mid] == x:
			return mid
		elif arr[mid] > x:
			return checkTrainSet(arr, low, mid - 1, x)
		else:
			return checkTrainSet(arr, mid + 1, high, x)

	else:
		return -1


def create_Model(data,labels,vocab,inverted_index):
	"""
	Create and fit the Naive Bayes model

	Parameters:  
	data(list): 	Containing the data points  
	labels(list): 	Containing the labels for the data points
	vocab(set): 	Contains the unique words in the dataset
	inverted_index(dict): 	Contains the word to data point number matching

	Returns:  
	float: Average accuracy
	"""
	datalen = len(data)
	datanums = list(range(0,datalen))
	random.shuffle(datanums)
	testlen = (int) ((1/7) * datalen)
	trainlen = datalen - testlen

	accuracies = []

	#k fold cross validation, k = 7
	for i in range(7):
		train_nums1 = datanums[0 : i*testlen]
		test_nums = datanums[i*testlen : (i+1)*testlen]
		train_nums2 = datanums[(i+1)*testlen : datalen]
		train_nums = train_nums1 + train_nums2
		# print(i," - ", len(test_nums), " ", len(train_nums))
		train_nums.sort()

		word_probabilities = {}
		train_positive = 0
		train_negative = 0
		for j in train_nums:
			if labels[j] == '1':
				train_positive += 1
			else:
				train_negative += 1
		# print("Total positive examples in training set are = ", train_positive)
		# print("Total negative examples in training set are = ", train_negative)

		for word in vocab:
			posnumer = 0
			negnumer = 0
			for j in inverted_index[word]:
				if checkTrainSet(train_nums, 0, len(train_nums)-1, j) != -1:
					if labels[j] == '1':
						posnumer += 1
					else:
						negnumer += 1
			word_probabilities[word] = [(posnumer+1)/(train_positive+1), (negnumer+1)/(train_negative+1)]

		correct_classified = 0
		for j in test_nums:
			pos_result = train_positive / len(train_nums)
			neg_result = train_negative / len(train_nums)
			for word in data[j].split():
				if len(word)>=3 and word!="and" and word!="the":
					pos_result *= word_probabilities[word][0]
					neg_result *= word_probabilities[word][1]

			# print("P(1|input) = ", pos_result)
			# print("P(0|input) = ", neg_result)

			if pos_result > neg_result and labels[j] == '1':
				correct_classified += 1
			elif pos_result < neg_result and labels[j] == '0':
				correct_classified += 1

		accuracies.append((correct_classified*100)/len(test_nums))
	print("Accuracy over the 7 fold cross validation:\n",list(np.around(np.array(accuracies),4)))
	return sum(accuracies) / len(accuracies)


if __name__ == '__main__':
	file = open("dataset_NB.txt","r")

	vocab = set()
	data = []
	labels = []

	for line in file:
		line = re.sub(r'[^\w\s]', '', line)		#remove punctuations 
		line = line.lower()						#lower case
		data.append(line)
		labels.append(line[-2])
		for word in line.split():
			# print(word)
			if len(word)>=3 and word!="and" and word!="the":
				vocab.add(word)

	file.close()

	inverted_index = createInvertedIndex(data)	
	vocablen = len(vocab)

	# print(inverted_index["bad"])
	# print(data[0])
	# print(vocablen)
	# print(labels)
	# print("Number of labels are = ",len(labels))

	accuracy = create_Model(data,labels,vocab,inverted_index)
	print("Average accuracy of the model is = ", round(accuracy,4))