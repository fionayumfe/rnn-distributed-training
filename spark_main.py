__Author__ = "Fiona Yu"
__Email__ = "fionayumfe@gmail.com"

import csv
import itertools
import operator
import numpy as np
import nltk
import sys
from datetime import datetime
import matplotlib.pyplot as plt
import timeit
import time

from pyspark.sql import functions as F
from pyspark.sql import SQLContext, Row, SparkSession
from pyspark.sql.functions import udf,struct
from pyspark import SparkContext, StorageLevel, SparkConf
from pyspark.sql.types import IntegerType, TimestampType, StringType,DoubleType,StructType,StructField,DateType,DataType,BooleanType,LongType


from utils.utils import *
from model.rnn_numpy import RNNNumpy

class CalcEngine(object):

    """
     calculation engine to preprocess training data and train models in parallel by PySpark

    """

    def __init__(self):

        self.vocabulary_size = 8000
        self.unknown_token = "UNKNOWN_TOKEN"
        self.sentence_start_token = "SENTENCE_START"
        self.sentence_end_token = "SENTENCE_END"

    def preprocess(self,input_file):

        # Read the data by nltk package and append SENTENCE_START and SENTENCE_END tokens
        print("Reading CSV file...")
        with open(input_file, 'r') as f:
            reader = csv.reader(f, skipinitialspace=True)
            next(reader)
            # Split full comments into sentences
            sentences = itertools.chain(*[nltk.sent_tokenize(x[0].lower()) for x in reader])
            # Append SENTENCE_START and SENTENCE_END
            sentences = ["%s %s %s" % (self.sentence_start_token, x, self.sentence_end_token) for x in sentences]
        print("Parsed %d sentences." % (len(sentences)))

        # Tokenize the sentences into words
        tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]

        # Count the word frequencies
        word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
        print("Found %d unique words tokens." % len(word_freq.items()))

        # Get the most common words and build index_to_word and word_to_index vectors
        vocab = word_freq.most_common(self.vocabulary_size-1)
        index_to_word = [x[0] for x in vocab]
        index_to_word.append(self.unknown_token)
        word_to_index = dict([(w,i) for i,w in enumerate(index_to_word)])

        print("Using vocabulary size %d." % self.vocabulary_size)
        print("The least frequent word in our vocabulary is '%s' and appeared %d times." % (vocab[-1][0], vocab[-1][1]))

        # Replace all words not in our vocabulary with the unknown token
        for i, sent in enumerate(tokenized_sentences):
            tokenized_sentences[i] = [w if w in word_to_index else self.unknown_token for w in sent]

        print("\nExample sentence: '%s'" % sentences[0])

        # Create the training data
        X_train = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in tokenized_sentences])
        y_train = np.asarray([[word_to_index[w] for w in sent[1:]] for sent in tokenized_sentences])
        print("\nExample sentence after Pre-processing: '%s'" % tokenized_sentences[0])

        # Print an training data example

        # x_example, y_example = X_train[17], y_train[17]
        # print("x:\n%s\n%s" % (" ".join([index_to_word[x] for x in x_example]), x_example))
        # print("\ny:\n%s\n%s" % (" ".join([index_to_word[x] for x in y_example]), y_example))

        return [X_train,y_train]


    def distributed_training(self,X_train, y_train,vocabulary_size,num_core=4,rate=0.005):

        """
        train rnn in parallel by Spark 2.2

        :param X_train: input training set
        :param y_train: target in training set
        :param vocabulary_size: size of vocabulary only frequent words will be modelled
        :param num_core: number of core to run in parallel default is 4 for standalone PC. It needs to change for EMR clusters.
        :param rate: learng rate default at 0.005
        :return: model parameters U,V,W matrix with minimum loss
        """

        global spark_context

        learning_rate = [ rate*np.random.random() for i in range(num_core)]
        print(learning_rate)

        learning_rate_rdd = spark_context.parallelize(learning_rate)

        #run map-reduce to find model parameters with mininum loss in training data
        result = learning_rate_rdd.map(lambda rate:train_model(rate,vocabulary_size,X_train,y_train)).reduceByKey(min)

        return result

def train_model(rate,vocabulary_size,X_train,y_train):

    """
     work in Spark map-reduce framework
    :param rate: learning rate, different worker has different starting learning rate. I tried to find global optimal paramers by differing learning rate
    :param vocabulary_size: size of the vocabulary
    :param X_train: input of training set which is the same across workers
    :param y_train: target of training set which is the same across workers
    :return: locally optimized parameters in each worker
    """

    model = RNNNumpy(vocabulary_size)
    loss =  model.train_with_sgd(X_train, y_train, rate)
    return {loss[-1]:model.get_parameter()}


def initialize_spark():

    """
    In Spark 2.2, Spark session is recommended(for Spark SQL dataframe operations).
    :return:
    """

    global debug
    global spark_context
    global spark_session
    global sql_context
    global log4j_logger
    global logger


    spark_session = SparkSession \
        .builder \
        .appName("tnn-distributed-training") \
        .getOrCreate()
    sql_context = SQLContext(spark_session.sparkContext)
    spark_context = spark_session.sparkContext
    log4j_logger = spark_context._jvm.org.apache.log4j
    logger = log4j_logger.LogManager.getLogger(__name__)

    log('SPARK INITIALIZED')

def log(s):
    global log_enabled
    if log_enabled:
        logger.warn(s)

# if __name__ == "__main__":

debug = False
spark_context = None
spark_session = None
sql_context = None
log4j_logger = None
logger = None
log_enabled = False


initialize_spark()
my_engine = CalcEngine()

X_train,y_train = my_engine.preprocess('data/reddit-comments-2015-08.csv')
vocabulary_size=8000


np.random.seed(10)
model = RNNNumpy(vocabulary_size)
o, s = model.forward_propagation(X_train[10])
print(o.shape)
print(o)


predictions = model.predict(X_train[10])
print(predictions.shape)
print(predictions)


# Limit to 1000 examples to save time
print("Expected Loss for random predictions: %f" % np.log(vocabulary_size))
print("Actual loss: %f" % model.calculate_loss(X_train[:1000], y_train[:1000]))


# To avoid performing millions of expensive calculations we use a smaller vocabulary size for checking.
grad_check_vocab_size = 100
np.random.seed(10)
model = RNNNumpy(grad_check_vocab_size, 10, bptt_truncate=1000)
model.gradient_check([0,1,2,3], [1,2,3,4])

np.random.seed(10)
model = RNNNumpy(vocabulary_size)
start_time = time.time()
model.sgd_step(X_train[10], y_train[10], 0.005)
end_time = time.time()
print("it take {0} seconds to do one step sgd".format(end_time - start_time))

np.random.seed(10)
# Train on a small subset of the data to see what happens
model = RNNNumpy(vocabulary_size)
losses = model.train_with_sgd(X_train[:100], y_train[:100])

print("starting to train model in parallel...")
optimal_parameter = my_engine.distributed_training(X_train[:100], y_train[:100],vocabulary_size)