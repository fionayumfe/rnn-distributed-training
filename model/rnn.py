__Author__ = "Fiona Yu"
__Email__ = "fionayumfe@gmail.com"

import abc
import sys

class rnn(abc.ABC):
    """
    abstract base class for recurrent neural network
    """

    def __init__(self, word_dim, hidden_dim=100, bptt_truncate=4):
        # Assign instance variables
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate
        # Randomly initialize the network parameters
        self.U = None
        self.V = None
        self.W = None
        self.training_loss=sys.maxsize

    @abc.abstractmethod
    def predict(self,x,y):
        pass

    @abc.abstractmethod
    def forward_propagation(self, x):
        pass

    @abc.abstractmethod
    def calculate_total_loss(self, x, y):
        pass

    @abc.abstractmethod
    def bptt(self, x, y):
        pass


