

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from random import random

import random

import pickle

import time 


class NN:

	# create MLP with x inputs, h hidden layer nodes, y outputs
	def __init__(self, x,h,y):

		# intialization class attributes 
		self.num_inputs = x
		self.num_hidden = h
		self.num_outputs = y

		# network consists of hidden layer and output layer
		self.network = []

		# radomly initialize all weights 
		hidden_layer = {'weights': np.random.rand(self.num_hidden,self.num_inputs+1) - 0.5}
		output_layer = {'weights': np.random.rand(self.num_outputs,self.num_hidden+1) - 0.5}
		
		self.network.append(hidden_layer)
		self.network.append(output_layer)

		# for layer in self.network:
		# 	print (layer)

		self.loss = []
		
	# SIGMOID ACTIVATION FUNCTION
	def sigmoid(self, x):
		return 1.0/(1.0 + np.exp(-x))
	
	# SOFTMAX ACTIVATION FUNCTION
	def softmax(self,x):
		e_x = np.exp(x - np.max(x))
		return e_x / e_x.sum()

	# CALCULATE WEIGHTED SUM
	def weighted_sum(self,w,x):

		w_sum= w[-1]
		
		for i in range(len(w)-1):
			w_sum += w[i]*x[i]
		
		return w_sum

	# FORWARD PROPAGATE TO FIND OUTPUT FROM NETWORK
	def forward_propagate(self, data_point):

		j = 0

		# for each layer in network (hidden,output)
		for layer in self.network:

			outputs = []
			w_sum_list = []
			
			# for each neuron in current layer
			for i in range(len(layer['weights'])):
				
				# if hidden layer
				if (j == 0):
				
					# calculate weighted sum between input nodes and current hidden neuron
					#w_sum = self.weighted_sum(layer['weights'][i],data_point)
					w_sum = np.dot(layer['weights'][i][:-1],data_point)
					w_sum += layer['weights'][i][-1]
					# print ("SIG WEIGHTED SUM - ", w_sum)
					
					# apply sigmoid activation function on weighted sum and save
					outputs.append(self.sigmoid(w_sum))

				# if output layer
				if (j == 1):

					# calculate weighted sum between output from hidden node and current output neuron
					#w_sum = self.weighted_sum(layer['weights'][i],data_point)
					#w_sum = np.dot(layer['weights'][i],data_point)
					w_sum = np.dot(layer['weights'][i][:-1],data_point)
					w_sum += layer['weights'][i][-1]
					
					# save the weighted sum in a list
					w_sum_list.append(w_sum)
				
			# save outs from hidden neuron
			if (j == 0):
				layer['outputs'] = outputs

			if (j==1):

				# apply softmax on the weighted sum vector
				layer['outputs'] = self.softmax(w_sum_list)
				outputs = layer['outputs']

			# input value for next later will be output from current layer
			data_point = outputs

			j = j + 1

		# return output from output neurons
		return data_point

	
	# BACKPROPAGATE ON NETWORK TO UPDATE WEIGHTS
	def back_propagate(self,x,y,output,learning_rate):

		
		delta_ho_list = []
		delta_hob_list = []
		sigma_list = []

		
		# --- UPDATING HIDDEN TO OUTPUT LAYER WEIGHTS --- #

		# for each output neuron
		for i in range(len(self.network[1]['weights'])):

			# error signal for current output node
			sigma = y[i] - self.network[1]['outputs'][i]
			sigma_list.append(sigma)

			delta_w_list = []

			for j in range(len(self.network[1]['weights'][i][:-1])):

				# delta for output node weight
				delta_w = learning_rate * sigma * self.network[0]['outputs'][j]
				delta_w_list.append(delta_w)
				

			# delta for output node bias weight
			delta_b = learning_rate * sigma * (-1)

			delta_ho_list.append(delta_w_list)
			delta_hob_list.append(delta_b)


		# save all the error signals from output node
		self.network[1]['sigma'] = np.asarray(sigma_list)

		# deltas for all output_hidden weights 
		total_delta_ohweights = np.append(np.asarray(delta_ho_list),np.asarray(delta_hob_list).reshape(self.num_outputs,1),axis=1)
		


		# --- UPDATING INPUT TO HIDDEN LAYER WEIGHTS --- #

		delta_ih_list = []
		delta_ihb_list = []

		# for each hidden neuron
		for i in range(len(self.network[0]['weights'])):

			delta_w_list = []

			hidden_output = self.network[0]['outputs'][i]

			# error signal for current hidden node 
			sigma = (hidden_output*(1-hidden_output))*np.dot(self.network[1]['weights'][:,i],self.network[1]['sigma'])

				
			for j in range(len(self.network[0]['weights'][i][:-1])):
				
				# delta for hidden node weight
				delta_w = learning_rate * sigma * x[j]
				delta_w_list.append(delta_w)

			
			# delta for hidden node bias weight
			delta_b = learning_rate * sigma * (-1)
		

			delta_ih_list.append(delta_w_list)
			delta_ihb_list.append(delta_b)


		# deltas for all input_hidden weights 
		total_delta_ihweights = np.append(np.asarray(delta_ih_list),np.asarray(delta_ihb_list).reshape(self.num_hidden,1),axis=1)
		
		# update all weights
		self.network[1]['weights'] = self.network[1]['weights'] + total_delta_ohweights
		
		self.network[0]['weights'] = self.network[0]['weights'] + total_delta_ihweights

	def cross_entropy_loss(self,target,output):
		return np.sum(-target * np.log(output))

	# TRAIN NEURAL NETWORK	
	def train(self,data_points,labels):

		error_val = []

		# run each point through network
		for i in range(len(data_points)):
			
			# take output from forward_prof and apply back_prop
			output = self.forward_propagate(data_points[i])
			learning_rate = 0.01
			self.back_propagate(data_points[i],labels[i],output,learning_rate)
			if (i%1000 == 0):
				print ("Done so far - ",i)
			if (i%100 == 0):
				error = self.cross_entropy_loss(labels[i],output)
				error_val.append(error)

		data = np.asarray(error_val)
		np.savetxt('error_plot_15.csv', data, delimiter=',')

	
	# PREDICT CLASS GIVEN DATA POINT
	def predict(self, data_point):
		
		# feed data point through the network
		outputs = self.forward_propagate(data_point)
		
		# value with the highest probability is 1, rest are zeros
		prediction = np.zeros((4,),dtype=int)
		prediction[np.argmax(outputs)] = 1

		return prediction

def main():
	
	# read in data points
	df = pd.read_csv('train_data.csv', sep=',',header=None)
	data_points = df.values
	
	# read in labels
	df = pd.read_csv('train_labels.csv', sep=',',header=None)
	labels = df.values

	# seperate data set into training and testing set
	indices = list(range(data_points.shape[0]))
	
	#80:20 split of data
	num_training_instances = int(0.8 * data_points.shape[0]) 
	
	random.seed(10)
	np.random.shuffle(indices)
	
	train_indices = indices[:num_training_instances]
	test_indices = indices[num_training_instances:]

	# save training data and testing data
	x_data_train, y_data_train = data_points[train_indices],labels[train_indices]
	x_data_test, y_data_test = data_points[test_indices],labels[test_indices]

	# create neural net with 15 hidden nodes
	classifier = NN(784,100,4)

	# train network on the training data
	classifier.train(x_data_train, y_data_train)

	# save network using pickle
	outfile = open("saved_model",'wb')
	pickle.dump(classifier.network,outfile)
	outfile.close()

	#### performing predictions now #####
	sum_correct = 0
	for i in range(len(x_data_test)):
		pred = classifier.predict(x_data_test[i])
		if ((pred == y_data_test[i]).all()):
			sum_correct +=1

	accuracy = (sum_correct/len(x_data_test))*100
	print ("Accuracy of MLP Network is - ", accuracy")

    
if __name__ == '__main__':
    main()



