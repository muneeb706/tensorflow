# # Sequence to Sequence (seq2seq) Recurrent Neural Network (RNN) for Time Series Prediction
#
# The goal of this project of mine is to bring users to try and experiment with the seq2seq neural network architecture. This is done by solving different simple toy problems about signal prediction. Normally, seq2seq architectures may be used for other more sophisticated purposes than for signal prediction, let's say, language modeling, but this project is an interesting tutorial in order to then get to more complicated stuff.
#
# In this project are given 4 exercises of gradually increasing difficulty. I take for granted that the public already have at least knowledge of basic RNNs and how can they be shaped into an encoder and a decoder of the most simple form (without attention). To learn more about RNNs in TensorFlow, you may want to visit this other project of mine about that: https://github.com/guillaume-chevalier/LSTM-Human-Activity-Recognition
#
# The current project is a series of example I have first built in French, but I haven't got the time to generate all the charts anew with proper English text. I have built this project for the practical part of the third hour of a "master class" conference that I gave at the WAQ (Web At Quebec) in March 2017:
# https://webaquebec.org/classes-de-maitre/deep-learning-avec-tensorflow
#
# You can find the French, original, version of this project in the French Git branch: https://github.com/guillaume-chevalier/seq2seq-signal-prediction/tree/francais
#
# ## How to use this ".ipynb" Python notebook ?
#
# Except the fact I made available an ".py" Python version of this tutorial within the repository, it is more convenient to run the code inside the notebook. The ".py" code exported feels a bit raw as an exportation.
#
# To run the notebook, you must have installed Jupyter Notebook or iPython Notebook. To open the notebook, you must write `jupyter notebook` or `iPython notebook` in command line (from the folder containing the notebook once downloaded, or a parent folder). It is then that the notebook application (IDE) will open in your browser as a local server and it will be possible to open the `.ipynb` notebook file and to run code cells with `CTRL+ENTER` and `SHIFT+ENTER`, it is also possible to restart the kernel and run all cells at once with the menus. Note that this is interesting since it is possible to make that IDE run as hosted on a cloud server with a lot of GPU power while you code through the browser.
#
# ## Exercises
#
# Note that the dataset changes in function of the exercice. Most of the time, you will have to edit the neural networks' training parameter to succeed in doing the exercise, but at a certain point, changes in the architecture itself will be asked and required. The datasets used for this exercises are found in `datasets.py`.
#
# ### Exercise 1
#
# In theory, it is possible to create a perfect prediction of the signal for this exercise. The neural network's parameters has been set to acceptable values for a first training, so you may pass this exercise by running the code without even a change. Your first training might get predictions like that (in yellow), but it is possible to do a lot better with proper parameters adjustments.
#
# Note: the neural network sees only what is to the left of the chart and is trained to predict what is at the right (predictions in yellow).
#
# We have 2 time series at once to predict, which are tied together. That means our neural network processes multidimensional data. A simple example would be to receive as an argument the past values of multiple stock market symbols in order to predict the future values of all those symbols with the neural network, which values are evolving together in time. That is what we will do in the exercise 6.
#
#
# ### Exercise 2
#
# Here, rather than 2 signals in parallel to predict, we have only one, for simplicity. HOWEVER, this signal is a superposition of two sine waves of varying wavelenght and offset (and restricted to a particular min and max limit of wavelengts).
#
# In order to finish this exercise properly, you will need to edit the neural network's hyperparameters. As an example, here is what is possible to achieve as a predction with those better (but still unperfect) training hyperparameters:
#
# - `num_iters = 2500`
# - `batch_size = 50`
# - `hidden_dim = 35`
# <img src="images/E2.png" />
#
# Note that it would be possible to obtain better results with a smaller neural network, provided better training hyperparameters and a longer training, adding dropout, and on.
#
# ### Exercise 3
#
# This exercise is similar to the previous one, except that the input data given to the encoder is noisy. The expected output is not noisy. This makes the task a bit harder.
#
# Therefore the neural network is brought to denoise the signal to interpret its future smooth values.
#
# Similarly as I said for the exercise 2, it would be possible here too to obtain better results. Note that it would also have been possible to ask you to predict to reconstruct the denoised signal from the noisy input (and not predict the future values of it). This would have been called a "denoising autoencoder", this type of architecture is also useful for data compression, such as manipulating images.
#
# ### Exercise 4
#
# This exercise is much harder than the previous ones and is built more as a suggestion. It is to predict the future value of the Bitcoin's price. We have here some daily market data of the bitcoin's value, that is, BTC/USD and BTC/EUR. This is not enough to build a good predictor, at least having data precise at the minute level, or second level, would be more interesting. Here is a prediction made on the actual future values, the neural network has not been trained on the future values shown here and this is a legitimate prediction, given a well-enough model trained on the task:
#
# <img src="images/E5.png" />
#
# Disclaimer: this prediction of the future values was really good and you should not expect predictions to be always that good using as few data as actually (side note: the other prediction charts in this project are all "average" except this one). Your task for this exercise is to plug the model on more valuable financial data in order to make more accurate predictions. Let me remind you that I provided the code for the datasets in "datasets.py", but that should be replaced for predicting accurately the Bitcoin.
#
# It would be possible to improve the input dimensions of your model that accepts (BTC/USD and BTC/EUR). As an example, you could create additionnal input dimensions/streams which could contain meteo data and more financial data, such as the S&P 500, the Dow Jones, and on. Other more creative input data could be sine waves (or other-type-shaped waves such as saw waves or triangles or two signals for `cos` and `sin`) representing the fluctuation of minutes, hours, days, weeks, months, years, moon cycles, and on. This could be combined with a Twitter sentiment analysis about the word "Bitcoin" in tweets in order to have another input signal which is more human-based and abstract. Actually, some libraries exists to convert text to a sentiment value, and there would also be the neural network end-to-end approach (but that would be a way more complicated setup). It is also interesting to know where is the bitcoin most used: http://images.google.com/search?tbm=isch&q=bitcoin+heatmap+world
#
# With all the above-mentionned examples, it would be possible to have all of this as input features, at every time steps: (BTC/USD, BTC/EUR, Dow_Jones, SP_500, hours, days, weeks, months, years, moons, meteo_USA, meteo_EUROPE, Twitter_sentiment). Finally, there could be those two output features, or more: (BTC/USD, BTC/EUR).
#
# This prediction concept can apply to many things, such as meteo prediction and other types of shot-term and mid-term statistical predictions.
#
# ## To change which exercise you are doing, change the value of the following "exercise" variable:
#

exercise = 4  # Possible values: 1, 2, 3, 4, or 5.

# In[1]:

#import os
#import pickle
import numpy as np
import requests
import sys
!pip install tensorflow==0.12.1
%tensorflow_version 0.12.1
import tensorflow as tf  # Version 1.0 or 0.12
# tf.disable_eager_execution()
# tf.disable_v2_behavior()

import matplotlib.pyplot as plt

import random
import math

__author__ = "Guillaume Chevalier"
__license__ = "MIT"
__version__ = "2017-03"


def generate_x_y_data_v1(isTest, batch_size, predict_days):
    """
    Data for exercise 1.

    returns: tuple (X, Y)
        X is a sine and a cosine from 0.0*pi to 1.5*pi
        Y is a sine and a cosine from 1.5*pi to 3.0*pi
    Therefore, Y follows X. There is also a random offset
    commonly applied to X an Y.

    The returned arrays are of shape:
        (seq_length, batch_size, output_dim)
        Therefore: (10, batch_size, 2)

    For this exercise, let's ignore the "isTest"
    argument and test on the same data.
    """
    seq_length = 10
    if (predict_days > seq_length): predict_days = seq_length

    batch_x = []
    batch_y = []

    for _ in range(batch_size):
        rand = random.random() * 2 * math.pi

        sig1 = np.sin(np.linspace(0.0 * math.pi + rand,
	                                  3.0 * math.pi + rand, seq_length * 2))
        sig2 = np.cos(np.linspace(0.0 * math.pi + rand,
						                                    3.0 * math.pi + rand, seq_length * 2))
        x1 = sig1[:seq_length]
        y1 = sig1[predict_days:predict_days+seq_length]
        x2 = sig2[:seq_length]
        y2 = sig2[predict_days:predict_days+seq_length]

        x_ = np.array([x1, x2])
        y_ = np.array([y1, y2])
        x_, y_ = x_.T, y_.T

        batch_x.append(x_)
        batch_y.append(y_)

    batch_x = np.array(batch_x)
    batch_y = np.array(batch_y)
	# shape: (batch_size, seq_length, output_dim)

    batch_x = np.array(batch_x).transpose((1, 0, 2))
    batch_y = np.array(batch_y).transpose((1, 0, 2))
	# shape: (seq_length, batch_size, output_dim)

    return batch_x, batch_y

def generate_x_y_data_two_freqs(isTrain, batch_size, seq_length, predict_days):
    if (predict_days > seq_length): predict_days = seq_length
    batch_x = []
    batch_y = []
    for _ in range(batch_size):
        offset_rand = random.random() * 2 * math.pi
        freq_rand = (random.random() - 0.5) / 1.5 * 15 + 0.5
        amp_rand = random.random() + 0.1

        sig1 = amp_rand * np.sin(np.linspace(
            seq_length / 15.0 * freq_rand * 0.0 * math.pi + offset_rand,
            seq_length / 15.0 * freq_rand * 3.0 * math.pi + offset_rand,
            seq_length * 2
        )
        )

        offset_rand = random.random() * 2 * math.pi
        freq_rand = (random.random() - 0.5) / 1.5 * 15 + 0.5
        amp_rand = random.random() * 1.2

        sig1 = amp_rand * np.cos(np.linspace(
            seq_length / 15.0 * freq_rand * 0.0 * math.pi + offset_rand,
            seq_length / 15.0 * freq_rand * 3.0 * math.pi + offset_rand,
            seq_length * 2
        )
        ) + sig1

        x1 = sig1[:seq_length]
        y1 = sig1[predict_days:predict_days+seq_length]

        x_ = np.array([x1])
        y_ = np.array([y1])
        x_, y_ = x_.T, y_.T

        batch_x.append(x_)
        batch_y.append(y_)

    batch_x = np.array(batch_x)
    batch_y = np.array(batch_y)
    # shape: (batch_size, seq_length, output_dim)

    batch_x = np.array(batch_x).transpose((1, 0, 2))
    batch_y = np.array(batch_y).transpose((1, 0, 2))
    # shape: (seq_length, batch_size, output_dim)

    return batch_x, batch_y


def generate_x_y_data_v2(isTest, batch_size, predict_days):
    """
    Similar the the "v1" function, but here we generate a signal with
    2 frequencies chosen randomly - and this for the 2 signals. Plus,
    the lenght of the examples is of 15 rather than 10.
    So we have 30 total values for past and future.
    """
    return generate_x_y_data_two_freqs(
	            isTest, batch_size, seq_length=15, predict_days=predict_days)


def generate_x_y_data_v3(isTrain, batch_size, predict_days):
    """
    Similar to the "v2" function, but here we generate a signal
    with noise in the X values. Plus,
    the lenght of the examples is of 30 rather than 10.
    So we have 60 total values for past and future.
    """
    seq_length = 30
    x, y = generate_x_y_data_two_freqs(
        isTrain, batch_size, seq_length=seq_length, predict_days=predict_days)
    noise_amount = random.random() * 0.15 + 0.10
    x = x + noise_amount * np.random.randn(seq_length, batch_size, 1)

    avg = np.average(x)
    std = np.std(x) + 0.0001
    x = x - avg
    y = y - avg
    x = x / std / 2.5
    y = y / std / 2.5

    return x, y

def loadCurrency(curr, window_size, predict_days):
    """
    Return the historical data for the USD or EUR bitcoin value. Is done with an web API call.
    curr = "USD" | "EUR"
    """
    # For more info on the URL call, it is inspired by :
    # https://github.com/Levino/coindesk-api-node
    r = requests.get(
        "http://api.coindesk.com/v1/bpi/historical/close.json?start=2010-07-17&end=2018-10-31&currency={}".format(
            curr
        ))
    data = r.json()
    time_to_values = sorted(data["bpi"].items())
    values = [val for key, val in time_to_values]
    #kept_values = values[1000:]
    #print("kept_values length =", len(kept_values))

    v = values[-1]
    for i in range(predict_days+1): values.append(v)
    
    X = []
    for i in range(len(values) - window_size):
        X.append(values[i:i + window_size])
    Y = X[predict_days:]
    X = X[:-predict_days]
    
    # To be able to concat on inner dimension later on:
    X = np.expand_dims(X, axis=2)
    Y = np.expand_dims(Y, axis=2)

    print("X.shape =", X.shape, "Y.shape =", Y.shape)
    return X, Y

input_data_mean = 0
input_data_std = 0

def normalize(X, Y=None):
    """
    Normalise X and Y according to the mean and standard deviation of the X values only.
    """
    # # It would be possible to normalize with last rather than mean, such as:
    # lasts = np.expand_dims(X[:, -1, :], axis=1)
    # assert (lasts[:, :] == X[:, -1, :]).all(), "{}, {}, {}. {}".format(lasts[:, :].shape, X[:, -1, :].shape, lasts[:, :], X[:, -1, :])
    mean = np.expand_dims(np.average(X, axis=1) + 0.00001, axis=1)
    stddev = np.expand_dims(np.std(X, axis=1) + 0.00001, axis=1)
    
    global input_data_mean
    global input_data_std

    input_data_mean = mean
    input_data_std = stddev
    # print (mean.shape, stddev.shape)
    # print (X.shape, Y.shape)
    X = X - mean
    X = X / (2.5 * stddev)
    if Y is not None:
        assert Y.shape == X.shape, (Y.shape, X.shape)
        Y = Y - mean
        Y = Y / (2.5 * stddev)
        return X, Y
    return X


def fetch_batch_size_random(X, Y, batch_size):
    """
    Returns randomly an aligned batch_size of X and Y among all examples.
    The external dimension of X and Y must be the batch size (eg: 1 column = 1 example).
    X and Y can be N-dimensional.
    """
    assert X.shape == Y.shape, (X.shape, Y.shape)
    idxes = np.random.randint(X.shape[0], size=batch_size)
    X_out = np.array(X[idxes]).transpose((1, 0, 2))
    Y_out = np.array(Y[idxes]).transpose((1, 0, 2))
    return X_out, Y_out

X_train = []
Y_train = []
X_test = []
Y_test = []


def generate_x_y_data_v4(isTest, batch_size, predict_days):
    """Return financial data for the bitcoin.

    Features are USD and EUR, in the internal dimension.
    We normalize X and Y data according to the X only to not
    spoil the predictions we ask for.

    For every window (window or seq_length), Y is the prediction
    following X.  Train and test data are separated according to the
    90/10 rule.  Every example in X contains 40 points of USD and then
    EUR data in the feature axis/dimension.  It is to be noted that
    the returned X and Y has the same shape and are in a tuple.

    """
    # 40 step_num for encoder and decoder's backpropagation.
    seq_length = 40
    if (predict_days > seq_length): predict_days = seq_length
    
    global Y_train
    global X_train
    global X_test
    global Y_test
    
    # First load, with memoization:
    if len(Y_test) == 0:
        print("Bitcoin price prediction")
        # API call:
        X_usd, Y_usd = loadCurrency("USD", window_size=seq_length, predict_days=predict_days)
        X_eur, Y_eur = loadCurrency("EUR", window_size=seq_length, predict_days=predict_days)

        # All data, aligned:
        X = np.concatenate((X_usd, X_eur), axis=2)
        Y = np.concatenate((Y_usd, Y_eur), axis=2)
        X, Y = normalize(X, Y)  # X.shape == Y.shape

        # Split 90-10:  X (and Y) is a list of vectors (of length seq_length)
        m = int(len(X) * 0.9)
        X_train = X[ : m ]
        Y_train = Y[ : m ]
        X_test = X[ m : ]
        Y_test = Y[ m : ]
        print("Shapes of train_set and test_set:", X_train.shape, X_test.shape)

    if isTest == 0:
        # return a random set of batch_size items from (X_train, Y_train)
        return fetch_batch_size_random(X_train, Y_train, batch_size)
    elif isTest == 1:
        # return a random set of batch_size items from (X_test, Y_test)
        return fetch_batch_size_random(X_test,  Y_test,  batch_size)
    else:
        # return the last batch_size items in (X_test, Y_test)
        X_out = np.array(X_test[-batch_size : ]).transpose((1, 0, 2))
        Y_out = np.array(Y_test[-batch_size : ]).transpose((1, 0, 2))
        return X_out, Y_out


"""

# Get stoack quotes

import pandas_datareader.data as web
from datetime import datetime

def loadStock(stock, window_size, predict_days):
   #
    Return the historical stock market close prices. Is done with a web API call.
    stock = 'SPY' or 'QQQ' for this assignment.
   #
        filepath = stock + '.pkl'
	    if not os.path.exists(filepath):
	            # start = datetime(2008, 3, 31) # 3/31/2008 is Monday
		            start = datetime(2013, 3, 25) # 3/25/2013 is Monday
			            end = datetime(2018, 3, 28)
				            # get data from morningstar API
					            #data = web.DataReader(stock, 'morningstar', start, end)
						            data = web.DataReader(stock, 'robinhood', start, end)
							            data = data.values
								            print("Loading successful: data.shape =", data.shape)
									            # data[:, 0] = Close value
										            # data[:, 1] = High value
											            # data[:, 2] = Low value
												            # data[:, 3] = Open value
													            # data[:, 4] = Volume; if Volume = 0, the market is closed that day
														            values = data[:, 0] # get stock's closed values
															            f = open(filepath, 'wb')
																            pickle.dump(values, f)
																	            f.close()
																		            print("File", filepath, ".pkl is created; if you change the data, remove this file first.")
																			        else:
																				        f = open(filepath, 'rb')
																					        values = pickle.load(f)

    print("closed values.shape =", values.shape)
        X = []
	    for i in range(len(values) - window_size + 1):
	            X.append(values[i:i + window_size])
		        Y = X[predict_days:]
			    X = X[:-predict_days]

    # To be able to concat on inner dimension later on:
        X = np.expand_dims(X, axis=2)
	    Y = np.expand_dims(Y, axis=2)

    print("X.shape =", X.shape, "Y.shape =", Y.shape)
        return X, Y

def generate_x_y_data_v5(isTest, batch_size, predict_days):
    #
    Return financial data for the stock symbol SPY.

    For every window (i.e, seq_length), Y is the prediction following X.
    Train and test data are separated according to the 90/10 rule.
    Every example in X contains seq_length points of SPY data 
    in the feature axis/dimension.
    It is to be noted that the returned X and Y has the same shape
    and are in a tuple.
    
    #
    # step_length is the number for encoder and decoder's backpropagation.
	# A small number is used for demo.
	seq_length = 60
	if (predict_days > seq_length): predict_days = seq_length

    # to be completed
"""



#from datasets import generate_x_y_data_v1, generate_x_y_data_v2, generate_x_y_data_v3, generate_x_y_data_v4, generate_x_y_data_v5

# We choose which data function to use below, in function of the exericse.
if exercise == 1: generate_x_y_data = generate_x_y_data_v1
if exercise == 2: generate_x_y_data = generate_x_y_data_v2
if exercise == 3: generate_x_y_data = generate_x_y_data_v3
if exercise == 4: generate_x_y_data = generate_x_y_data_v4
#if exercise == 5: generate_x_y_data = generate_x_y_data_v5


# In[2]:

# This is for the notebook to generate inline matplotlib
# charts rather than to open a new window every time:
# get_ipython().magic('matplotlib inline')


# ## Neural network's hyperparameters

# In[3]:
# Configuration of Prediction:
num_predictions = 4
predict_days = 5  # prediction in the next predict_days

# Configuration of Optmizer:
learning_rate = 0.001  # Small lr helps not to diverge during training.
num_iters = 500  # How many times we perform a training step (therefore how many times we show a batch).
lr_decay = 0.92  # default: 0.9 . Simulated annealing.
momentum = 0.5  # default: 0.0 . Momentum technique in weights update
lambda_l2_reg = 0.003  # L2 regularization of weights - avoids overfitting
batch_size = 20  # Low value used for live demo purposes - 100 and 1000 would be possible too, crank that up!

# Neural network parameters
hidden_dim = 20  # Count of hidden neurons in the recurrent units.
layers_stacked_count = 2  # Number of stacked recurrent cells, on the neural depth axis.

sample_x, sample_y = generate_x_y_data(isTest=0, batch_size=batch_size, predict_days = predict_days)
print("Dimensions of X and Y training examples: ")
print("  (seq_length, batch_size, output_dim) = ", sample_x.shape, sample_y.shape)

# Dependent neural network parameters
seq_length = sample_x.shape[0]  # Time series for backpropagation
output_dim = input_dim = sample_x.shape[-1]  # Output dimension (e.g.: multiple signals at once, tied in time)


# ## Definition of the seq2seq neuronal architecture
#
# <img src="https://www.tensorflow.org/images/basic_seq2seq.png" />
#
# Comparatively to what we see in the image, our neural network deals with signal rather than letters. Also, we don't have the feedback mechanism yet.


# Backward compatibility for TensorFlow's version 0.12:
try:
    tf.nn.seq2seq = tf.contrib.legacy_seq2seq
    tf.nn.rnn_cell = tf.contrib.rnn
    tf.nn.rnn_cell.GRUCell = tf.contrib.rnn.GRUCell
    # print("TensorFlow's version : 1.0 (or more)")
except: 
    print("TensorFlow's version : 0.12")

tf.reset_default_graph()
# sess.close()
sess = tf.InteractiveSession()

with tf.variable_scope('Seq2seq'):

    # Encoder: inputs
    enc_inp = [
        tf.placeholder(tf.float32, shape=(None, input_dim), name="inp_{}".format(t))
           for t in range(seq_length)
    ]

    # Decoder: expected outputs
    expected_sparse_output = [
        tf.placeholder(tf.float32, shape=(None, output_dim), name="expected_sparse_output_".format(t))
          for t in range(seq_length)
    ]
    
    # Give a "GO" token to the decoder. 
    # Note: we might want to fill the encoder with zeros or its own feedback rather than with "+ enc_inp[:-1]"
    dec_inp = [ tf.zeros_like(enc_inp[0], dtype=np.float32, name="GO") ] + enc_inp[:-1]
    # dec_inp = enc_inp

    # Create a `layers_stacked_count` of stacked RNNs (GRU cells here). 
    cells = []
    for i in range(layers_stacked_count):
        with tf.variable_scope('RNN_{}'.format(i)):
            cells.append(tf.nn.rnn_cell.GRUCell(hidden_dim))
            # cells.append(tf.nn.rnn_cell.BasicLSTMCell(...))
    cell = tf.nn.rnn_cell.MultiRNNCell(cells)
    
    # Here, the encoder and the decoder uses the same cell, HOWEVER,
    # the weights aren't shared among the encoder and decoder, we have two
    # sets of weights created under the hood according to that function's def. 
    dec_outputs, dec_memory = tf.nn.seq2seq.basic_rnn_seq2seq(
        enc_inp, 
        dec_inp, 
        cell
    )
    
    # For reshaping the output dimensions of the seq2seq RNN: 
    w_out = tf.Variable(tf.random_normal([hidden_dim, output_dim]))
    b_out = tf.Variable(tf.random_normal([output_dim]))
    
    # Final outputs: with linear rescaling for enabling possibly large and unrestricted output values.
    output_scale_factor = tf.Variable(1.0, name="Output_ScaleFactor")
    
    reshaped_outputs = [output_scale_factor*(tf.matmul(i, w_out) + b_out) for i in dec_outputs]


# Training loss and optimizer

with tf.variable_scope('Loss'):
    # L2 loss
    output_loss = 0
    for _y, _Y in zip(reshaped_outputs, expected_sparse_output):
        output_loss += tf.reduce_mean(tf.nn.l2_loss(_y - _Y))
        
    # L2 regularization (to avoid overfitting and to have a  better generalization capacity)
    reg_loss = 0
    for tf_var in tf.trainable_variables():
        if not ("Bias" in tf_var.name or "Output_" in tf_var.name):
            reg_loss += tf.reduce_mean(tf.nn.l2_loss(tf_var))
            
    loss = output_loss + lambda_l2_reg * reg_loss

with tf.variable_scope('Optimizer'):
    optimizer = tf.train.RMSPropOptimizer(learning_rate, decay=lr_decay, momentum=momentum)
    train_op = optimizer.minimize(loss)


# Training of the neural net

def train_batch(batch_size):
    """
    Training step that optimizes the weights 
    provided some batch_size X and Y examples from the dataset. 
    """
    X, Y = generate_x_y_data(isTest=0, batch_size=batch_size, predict_days=predict_days)
    feed_dict = {enc_inp[t]: X[t] for t in range(len(enc_inp))}
    feed_dict.update({expected_sparse_output[t]: Y[t] for t in range(len(expected_sparse_output))})
    _, loss_t = sess.run([train_op, loss], feed_dict)
    return loss_t

def test_batch(batch_size):
    """
    Test step, does NOT optimizes. Weights are frozen by not
    doing sess.run on the train_op. 
    """
    X, Y = generate_x_y_data(isTest=1, batch_size=batch_size, predict_days=predict_days)
    feed_dict = {enc_inp[t]: X[t] for t in range(len(enc_inp))}
    feed_dict.update({expected_sparse_output[t]: Y[t] for t in range(len(expected_sparse_output))})
    loss_t = sess.run([loss], feed_dict)
    return loss_t[0]


# Training
train_losses = []
test_losses = []

sess.run(tf.global_variables_initializer())
for t in range(num_iters+1):
    train_loss = train_batch(batch_size)
    train_losses.append(train_loss)
    
    if t % 10 == 0: 
        # Tester
        test_loss = test_batch(batch_size)
        test_losses.append(test_loss)
        sys.stdout.flush()
        #sys.stdout.write("\rStep %d/%d, train loss: %.2f, \tTEST loss: %.2f" % (t, num_iters, train_loss, test_loss))
        #print("Step {}/{}, train loss: {}, \tTEST loss: {}".format(t, num_iters, train_loss, test_loss))

print("\nFinal train loss: {}, \tTEST loss: {}".format(train_loss, test_loss))

# Plot loss over time:
plt.figure(figsize=(12, 6))
plt.plot(
    np.array(range(0, len(test_losses)))/float(len(test_losses)-1)*(len(train_losses)-1), 
    np.log(test_losses), 
    label="Test loss"
)
plt.plot(
    np.log(train_losses), 
    label="Train loss"
)
plt.title("Training errors over time (on a logarithmic scale)")
plt.xlabel('Iteration')
plt.ylabel('log(Loss)')
plt.legend(loc='best')
plt.show()



def visualize(isTest=1):
    X, Y = generate_x_y_data(isTest=isTest, batch_size=num_predictions, predict_days=predict_days)
    feed_dict = {enc_inp[t]: X[t] for t in range(seq_length)}
    outputs = np.array(sess.run([reshaped_outputs], feed_dict)[0])

    for j in range(num_predictions): 
        plt.figure(figsize=(12, 3))
    
        for k in range(output_dim):
            past = X[:,j,k]
            past = past[:predict_days]
            expected = Y[:,j,k]
            pred = outputs[:,j,k]
 
            label1 = "Seen (past) values" if k==0 else "_nolegend_"
            label2 = "True future values" if k==0 else "_nolegend_"
            label3 = "Predictions" if k==0 else "_nolegend_"
            plt.plot(range(len(past)), past, "o--b", label=label1)
            plt.plot(range(len(past), len(expected)+len(past)), expected, "x--b", label=label2)
            plt.plot(range(len(past), len(pred)+len(past)), pred, "o--y", label=label3)
    
        plt.legend(loc='best')
        plt.title("Predictions v.s. true values")
        plt.show()

# Suppose x0 is the current day price of bitcoin and x1, x2, x3, x4, and x5 are the price prediction for the next five days,
# using the model. We may use these values x0, ..., x5 to design buy/sell rules:
# Buy rule: if x0 < (x1+x2)/2 < (x3+x4)/2 < x5, then place a buy order at the price of the next day;
# Sell rule: if x0 > (x1+x2)/2 > (x3+x4)/2 > x5, then place a sell order at the price of the next day.
# To test the above buy/sell rules, you create two accounts: cash and coins.
# Initially, cash=$1,000,000 and coins=0.
# For each buy order, if your cash account allows, buy upto $100,000 worth of bitcoins,
# add them into your coins account, and subtract the value of these coins from the cash account.
# For each sell order, sell one third of the total bitcoins and add the values of these coins into your cash account.
# You can place at most one order a day. Use prices for all trading and ignore trading fees.
#
# Following function tests the above trading strategy for the prices of bitcoin of 200 consecutive days (in the test set)
# and reports result (how many trades, changes of values in your cash and coins accounts).

def start_trading(consecutive_days):
    
    # cash account
    cash=1000000
    # coin account
    coins=0
    
    X, Y = generate_x_y_data(isTest=1, batch_size=consecutive_days, predict_days=predict_days)
   
    feed_dict = {enc_inp[t]: X[t] for t in range(seq_length)}
    outputs = np.array(sess.run([reshaped_outputs], feed_dict)[0])

    # denormalizing for getting exact values of bitcoin
    # cloning X
    P = X
    P = P * (2.5 * input_data_std[len(input_data_std) - len(X):])
    P = P + input_data_mean[len(input_data_mean) - len(X):]        
   
    total_trades = 0
    for j in range(consecutive_days): 

        past = X[:,j,1]   
        past = past[:predict_days]
            
        expected = Y[:,j,1]
        pred = outputs[:,j,1]
        x0 = past[-1]
        x1 = pred[0]
        x2 = pred[1]
        x3 = pred[2]
        x4 = pred[3]
        x5 = pred[4]
        
        bitcoin_value = P[:,j,1]   
        bitcoin_value = bitcoin_value[:predict_days]
        bitcoin_value = bitcoin_value[-1]
        
        # Buy Rule
        if x0 < (x1+x2)/2 and (x1+x2)/2 < (x3+x4)/2 and (x3+x4)/2 < x5:
          if cash > bitcoin_value:  
            print("\nBuying Bitcoins.")
            if cash > 100000:
              coins += 100000 / bitcoin_value
              cash -= 100000
            else:
              coins += cash / bitcoin_value
              cash = 0
            print(f"After Buying, cash={cash}, coins={coins}")
            total_trades += 1

            # Sell Rule
        elif x0 > (x1+x2)/2 and (x1+x2)/2 > (x3+x4)/2 and (x3+x4)/2 > x5:
          if coins > 0:
            print("\nSelling Bitcoins.")
            coins_to_sell = coins / 3
            cash += bitcoin_value*coins_to_sell
            coins -= coins_to_sell
            print(f"After Selling, cash={cash}, coins={coins}")
            total_trades += 1

    print(f"\nTotal Trades: {total_trades}")
# Test
#print("Let's visualize {} predictions with our signals:".format(num_predictions))
#visualize(isTest=1)

# Test
#print("Let's visualize {} predictions on the latest data:".format(num_predictions))
#visualize(isTest=2)

start_trading(200)

print("Reminder: the signal can contain many dimensions at once.")
print("In that case, signals have the same color.")
print("In reality, we could imagine multiple stock market symbols evolving,")
print("tied in time together and seen at once by the neural network.")


# ## Author
# 
# Guillaume Chevalier
# - https://ca.linkedin.com/in/chevalierg
# - https://twitter.com/guillaume_che
# - https://github.com/guillaume-chevalier/
# 
# ## License
# 
# This project is free to use according to the [MIT License](https://github.com/guillaume-chevalier/seq2seq-signal-prediction/blob/master/LICENSE) as long as you cite me and the License (read the License for more details). You can cite me by pointing to the following link: 
# - https://github.com/guillaume-chevalier/seq2seq-signal-prediction