import csv
import itertools
import operator
import numpy as np
import nltk
import sys
from datetime import datetime
import matplotlib.pyplot as plt

#from model.rnn import *
from utils.utils import *

#remove metaclass because it cann't be seriazlied by PySpark 2.2 (Oct 4th 2017 Fiona)
#https://issues.apache.org/jira/browse/SPARK-21439

class RNNNumpy:
    
    def __init__(self, word_dim, hidden_dim=100, bptt_truncate=4):

        # Assign instance variables
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate
        # Randomly initialize the network parameters
        self.U = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (hidden_dim, word_dim))
        self.V = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (word_dim, hidden_dim))
        self.W = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (hidden_dim, hidden_dim))
        self.training_loss=sys.maxsize

    def get_parameter(self):

        return [self.U,self.V,self.W]

    def forward_propagation(self, x):
        """
        forward propagation in recurrent neural network

        :param x: input of training set
        :return: output from rnn which includes probability for each input words
        """

        # The total number of time steps
        T = len(x)
        # During forward propagation we save all hidden states in s because need them later.
        # We add one additional element for the initial hidden, which we set to 0
        s = np.zeros((T + 1, self.hidden_dim))
        s[-1] = np.zeros(self.hidden_dim)
        # The outputs at each time step. Again, we save them for later.
        o = np.zeros((T, self.word_dim))
        # For each time step...
        for t in np.arange(T):
            # Note that we are indxing U by x[t]. This is the same as multiplying U with a one-hot vector.
            s[t] = np.tanh(self.U[:,x[t]] + self.W.dot(s[t-1]))
            o[t] = softmax(self.V.dot(s[t]))
        return [o, s]

    def predict(self, x):
        """
        Perform forward propagation and return index of the highest score
        :param x: input of training set
        :return: words with highest probability to happen next
        """

        o, s = self.forward_propagation(x)
        return np.argmax(o, axis=1)

    def calculate_total_loss(self, x, y):
        L = 0
        # For each sentence...
        for i in np.arange(len(y)):
            o, s = self.forward_propagation(x[i])
            # We only care about our prediction of the "correct" words
            correct_word_predictions = o[np.arange(len(y[i])), y[i]]
            # Add to the loss based on how off we were
            L += -1 * np.sum(np.log(correct_word_predictions))
        return L

    def sgd_step(self, x, y, learning_rate):
        """
         Performs one step of stochastic gradient decress
        :param x: input of training set
        :param y: target of training set
        :param learning_rate: learning_rate
        :return: None
        """

        # Calculate the gradients
        dLdU, dLdV, dLdW = self.bptt(x, y)
        # Change parameters according to gradients and learning rate
        self.U -= learning_rate * dLdU
        self.V -= learning_rate * dLdV
        self.W -= learning_rate * dLdW

    def calculate_loss(self, x, y):
        # Divide the total loss by the number of training examples
        N = np.sum((len(y_i) for y_i in y))
        self.training_loss = self.calculate_total_loss(x,y)/N
        return self.training_loss

    def bptt(self, x, y):
        """
        back propagation through time which is needed for model parameters estimation

        :param x: input from training set
        :param y: target from training set
        :return: gradient of loss to model parameters
        """
        T = len(y)
        # Perform forward propagation
        o, s = self.forward_propagation(x)
        # We accumulate the gradients in these variables
        dLdU = np.zeros(self.U.shape)
        dLdV = np.zeros(self.V.shape)
        dLdW = np.zeros(self.W.shape)
        delta_o = o
        delta_o[np.arange(len(y)), y] -= 1.
        # For each output backwards...
        for t in np.arange(T)[::-1]:
            dLdV += np.outer(delta_o[t], s[t].T)
            # Initial delta calculation
            delta_t = self.V.T.dot(delta_o[t]) * (1 - (s[t] ** 2))
            # Backpropagation through time (for at most self.bptt_truncate steps)
            for bptt_step in np.arange(max(0, t-self.bptt_truncate), t+1)[::-1]:
                # print "Backpropagation step t=%d bptt step=%d " % (t, bptt_step)
                dLdW += np.outer(delta_t, s[bptt_step-1])
                dLdU[:,x[bptt_step]] += delta_t
                # Update delta for next step
                delta_t = self.W.T.dot(delta_t) * (1 - s[bptt_step-1] ** 2)
        return [dLdU, dLdV, dLdW]

    def gradient_check(self, x, y, h=0.001, error_threshold=0.01):
        """
        Calculate the gradients using backpropagation. We want to checker if these are correct.
        :param x: input of training set
        :param y: target of training set
        :param h: step for finite difference which is an approximation of derivative
        :param error_threshold:
        :return:
        """

        bptt_gradients = self.bptt(x, y)
        # List of all parameters we want to check.
        model_parameters = ['U', 'V', 'W']
        # Gradient check for each parameter
        for pidx, pname in enumerate(model_parameters):
            # Get the actual parameter value from the mode, e.g. model.W
            parameter = operator.attrgetter(pname)(self)
            print("Performing gradient check for parameter %s with size %d." % (pname, np.prod(parameter.shape)))
            # Iterate over each element of the parameter matrix, e.g. (0,0), (0,1), ...
            it = np.nditer(parameter, flags=['multi_index'], op_flags=['readwrite'])
            while not it.finished:
                ix = it.multi_index
                # Save the original value so we can reset it later
                original_value = parameter[ix]
                # Estimate the gradient using (f(x+h) - f(x-h))/(2*h)
                parameter[ix] = original_value + h
                gradplus = self.calculate_total_loss([x],[y])
                parameter[ix] = original_value - h
                gradminus = self.calculate_total_loss([x],[y])
                estimated_gradient = (gradplus - gradminus)/(2*h)
                # Reset parameter to original value
                parameter[ix] = original_value
                # The gradient for this parameter calculated using backpropagation
                backprop_gradient = bptt_gradients[pidx][ix]
                # calculate The relative error: (|x - y|/(|x| + |y|))
                relative_error = np.abs(backprop_gradient - estimated_gradient)/(np.abs(backprop_gradient) + np.abs(estimated_gradient))
                # If the error is to large fail the gradient check
                if relative_error > error_threshold:
                    print("Gradient Check ERROR: parameter=%s ix=%s" % (pname, ix))
                    print("+h Loss: %f" % gradplus)
                    print("-h Loss: %f" % gradminus)
                    print("Estimated_gradient: %f" % estimated_gradient)
                    print("Backpropagation gradient: %f" % backprop_gradient)
                    print("Relative Error: %f" % relative_error)
                    return
                it.iternext()
            print("Gradient check for parameter %s passed." % (pname))


    def train_with_sgd(self,X_train, y_train, learning_rate=0.005, nepoch=10, evaluate_loss_after=5):
        # Outer SGD Loop
        # - model: The RNN model instance
        # - X_train: The training data set
        # - y_train: The training data labels
        # - learning_rate: Initial learning rate for SGD
        # - nepoch: Number of times to iterate through the complete dataset
        # - evaluate_loss_after: Evaluate the loss after this many epochs
        # We keep track of the losses so we can plot them later
        losses = []
        num_examples_seen = 0

        for epoch in range(nepoch):
            # Optionally evaluate the loss
            if (epoch % evaluate_loss_after == 0):
                loss = self.calculate_loss(X_train, y_train)
                losses.append((num_examples_seen, loss))
                time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print("%s: Loss after num_examples_seen=%d epoch=%d: %f" % (time, num_examples_seen, epoch, loss))

                if (len(losses) > 1 ):
                    if losses[-1][1] > losses[-2][1]:
                        # Adjust the learning rate if loss increases
                        learning_rate = learning_rate * 0.5
                        print("Setting learning rate to %f" % learning_rate)
                    else:
                        # Adjust the learning rate if loss decreases
                        learning_rate = learning_rate * 1.05
                        print("Setting learning rate to %f" % learning_rate)
                sys.stdout.flush()
            # For each training example...
            for i in range(len(y_train)):
                # One SGD step
                self.sgd_step(X_train[i], y_train[i], learning_rate)
                num_examples_seen += 1

        return losses

