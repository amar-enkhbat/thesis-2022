import os
from utils import *
from cnn_class import cnn
from TfRnnAttention.attention import attention
import tensorflow as tf
import time
import random
import numpy as np
import pandas as pd
import pickle
from convert_to_graphs import *

random_seed = 33
tf.set_random_seed(random_seed)
random.seed(random_seed)
np.random.seed(random_seed)

os.environ["CUDA_VISIBLE_DEVICES"]='0'

file_num = 1

num_node = 64

model = 'ng_cram'

# data = sio.loadmat("./cross_subject_data_"+str(file_num)+".mat")
# data = pickle.load(open("../dataset/train/cross_subject_data_0.pickle", "rb"))
X_train = np.random.randn(1024, 64)
y_train = np.random.randint(0, 4, 1024)

X_test = np.random.randn(1024, 64)
y_test = np.random.randint(0, 4, 1024)

data = {'X_train': X_train, 'y_train': y_train, 'X_test': X_test, 'y_test': y_test}
print(data.keys())
test_X	= data["X_test"]
train_X	= data["X_train"]
print("X_test shape:")
print(test_X.shape)
print("X_train shape:")
print(train_X.shape)

test_y	= data["y_test"].ravel()
train_y = data["y_train"].ravel()
print("test_y shape:")
print(test_y.shape)
print("train_X shape:")
print(train_X.shape)

train_y = np.asarray(pd.get_dummies(train_y.ravel()), dtype = np.int8)
test_y = np.asarray(pd.get_dummies(test_y.ravel()), dtype = np.int8)
print("Y after pd.get_dummies:")
print("train_y shape:", train_y.shape)
print("test_y shape:", test_y.shape)

tmp_df = pd.read_csv("../dataset/physionet.org_csv/S001/S001R01.csv")
ch_names = tmp_df.columns[2:]
ch_pos_1010_dist, ch_pos_1010_names = get_sensor_pos(ch_names)

if model == 'ng_cram':
	adj = n_graph()

elif model == 'dg_cram':
	adj = d_graph(num_node, ch_pos_1010_dist)

elif model == 'sg_cram':
	adj= s_graph(num_node, ch_pos_1010_dist)

print("Adjacency matrix shape:", adj.shape)

test_X = np.matmul(np.expand_dims(adj, 0), np.expand_dims(test_X, 2))
train_X = np.matmul(np.expand_dims(adj, 0), np.expand_dims(train_X, 2))
# test_X = test_X.dot(adj)
# train_X = train_X.dot(adj)
print("test_X shape after expand and matmul with adj:")
print(test_X.shape)
print("train_X shape after expand and matmul with adj:")
print(train_X.shape)

window_size = 400
step = 10

train_raw_x = np.transpose(train_X, [0, 2, 1])
test_raw_x = np.transpose(test_X, [0, 2, 1])
print("train_raw_x shape after np.transpose(test_X, [0, 2, 1]):")
print(train_raw_x.shape)
print("test_raw_x shape after np.transpose(test_X, [0, 2, 1]):")
print(test_raw_x.shape)



def segment_dataset(X, window_size, step):
    windows = (i for i in range(0, len(X), step))
    win_x = []
    i = next(windows)
    while i + window_size < X.shape[0]:
        # print(i)
        win_x.append(X[i:i+window_size])
        i = next(windows)
    return np.array(win_x)

# train_win_x = segment_dataset(train_raw_x, window_size, step)
# test_win_x = segment_dataset(test_raw_x, window_size, step)

# pickle.dump(train_win_x, open("../dataset/train/train_win_x.pickle", "wb"))
# pickle.dump(test_win_x, open("../dataset/train/test_win_x.pickle", "wb"))

train_win_x = segment_dataset(train_raw_x, window_size, window_size)
test_win_x = segment_dataset(test_raw_x, window_size, window_size)
train_y = train_y[::window_size]
test_y = test_y[::window_size]

print("train_raw_x shape after segment_dataset:")
print(train_win_x.shape)
print("test_raw_x shape after segment_dataset:")
print(test_win_x.shape)
print("Train y shape:")
print(train_y.shape)
print("Test y shape:")
print(test_y.shape)
# exit()
# [trial, window, channel, time_length]
train_win_x = np.transpose(train_win_x, [0, 2, 3, 1])
test_win_x = np.transpose(test_win_x, [0, 2, 3, 1])
print("train_raw_x shape after np.transpose(train_win_x, [0, 1, 3, 2]):")
print(train_win_x.shape)
print("test_raw_x shape after np.transpose(train_win_x, [0, 1, 3, 2]):")
print(test_win_x.shape)
# exit()

features_train = train_win_x
features_test = test_win_x
y_train = train_y
y_test = test_y

print("features_train shape:", features_train.shape)
print("features_test shape:", features_test.shape)

features_train = np.expand_dims(features_train, axis = -1)
features_test = np.expand_dims(features_test, axis = -1)
print("train_raw_x shape after np.expand_dims(features_test, axis = -1):")
print(features_train.shape)
print("test_raw_x shape after np.expand_dims(features_test, axis = -1):")
print(features_test.shape)

num_timestep = features_train.shape[1]
###########################################################################
# set model parameters
###########################################################################
# kernel parameter
kernel_height_1st	= 64
kernel_width_1st 	= 45

kernel_stride		= 1

conv_channel_num	= 40
# pooling parameter
pooling_height_1st 	= 1
pooling_width_1st 	= 75

pooling_stride_1st = 10
# full connected parameter
fc_size = 512
attention_size = 512
n_hidden_state = 64

n_fc_in = "None"

n_fc_out = "None"
###########################################################################
# set dataset parameters
###########################################################################
# input channel
input_channel_num = 1

# input height 
input_height = features_train.shape[2]

# input width
input_width = features_train.shape[3]

# prediction class
num_labels = 4
###########################################################################
# set training parameters
###########################################################################
# set learning rate
learning_rate = 1e-5

# set maximum traing epochs
training_epochs = 5000

# set batch size
batch_size = 10

# set dropout probability
dropout_prob = 0.5

# set train batch number per epoch
batch_num_per_epoch = features_train.shape[0]//batch_size

# instance cnn class
cnn_2d = cnn(padding='VALID')

# input placeholder
X = tf.placeholder(tf.float32, shape=[None, input_height, input_width, input_channel_num], name='X')
Y = tf.placeholder(tf.float32, shape=[None, num_labels], name='Y')
train_phase = tf.placeholder(tf.bool, name='train_phase')
keep_prob = tf.placeholder(tf.float32, name='keep_prob')

print('X size:', X)
print('Y size:', Y)
# first CNN layer
conv_1 = cnn_2d.apply_conv2d(X, kernel_height_1st, kernel_width_1st, input_channel_num, conv_channel_num, kernel_stride, train_phase)
print('Conv_1 size:', conv_1)
pool_1 = cnn_2d.apply_max_pooling(conv_1, pooling_height_1st, pooling_width_1st, pooling_stride_1st)
print('Pool_1 size:', pool_1)
pool1_shape = pool_1.get_shape().as_list()
pool1_flat = tf.reshape(pool_1, [-1, pool1_shape[1]*pool1_shape[2]*pool1_shape[3]])
print('Pool1_flat shape:', pool1_flat)
fc_drop = tf.nn.dropout(pool1_flat, keep_prob)	

if (n_fc_in == 'None'):
	print("fc_in is None\n")
	lstm_in = tf.reshape(fc_drop, [-1, num_timestep, pool1_shape[1]*pool1_shape[2]*pool1_shape[3]])
else:
	lstm_in = tf.reshape(fc_drop, [-1, num_timestep, n_fc_in])
print('lstm_in shape:', lstm_in)
########################## RNN ########################
output = lstm_in
for layer in range(2):
	with tf.variable_scope('rnn_{}'.format(layer),reuse=False):
		cell_fw = tf.contrib.rnn.LSTMCell(n_hidden_state)
		cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, input_keep_prob = keep_prob)

		cell_bw = tf.contrib.rnn.LSTMCell(n_hidden_state)
		cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw, input_keep_prob = keep_prob)

		outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, output, dtype=tf.float32)
		print('Outputs shape:', outputs)
		output = tf.concat(outputs,2)
		print('Output shape:', output)
		state = tf.concat(states,2)

rnn_op = output
print('rnn_op shape:', rnn_op)
print('Attention size:', attention_size)
########################## attention ########################
with tf.name_scope('Attention_layer'):
    attention_op, alphas = attention(rnn_op, attention_size, time_major = False, return_alphas=True)
print('Attention shape', attention_op)

attention_drop = tf.nn.dropout(attention_op, keep_prob)	
y_ = cnn_2d.apply_readout(attention_drop, rnn_op.shape[2].value, num_labels)

# probability prediction 
y_posi = tf.nn.softmax(y_, name = "y_posi")

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_, labels=Y), name = 'loss')

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
	# set training SGD optimizer
	optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

# get correctly predicted object
correct_prediction = tf.equal(tf.argmax(tf.nn.softmax(y_), 1), tf.argmax(Y, 1))

# calculate prediction accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name = 'accuracy')

total_parameters = 0
for variable in tf.trainable_variables():
    # shape is an array of tf.Dimension
    shape = variable.get_shape()
    print(shape)
    print(len(shape))
    variable_parameters = 1
    for dim in shape:
        print(dim)
        variable_parameters *= dim.value
    print(variable_parameters)
    total_parameters += variable_parameters
print(total_parameters)

exit(1)
###########################################################################
# train test and save result
###########################################################################
# run with gpu memory growth
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

history_train = []
history_test = []
# test_cm = np.zeros((num_labels, num_labels))

for epoch in range(training_epochs):
	true_test = []
	posi_test = []
	# training process
	for b in range(batch_num_per_epoch):
		offset = (b * batch_size) % (y_train.shape[0] - batch_size) 
		batch_x = features_train[offset:(offset + batch_size), :, :, :, :]
		batch_x = batch_x.reshape([len(batch_x)*num_timestep, num_node, window_size, 1])
		# batch_x = batch_x.reshape([-1, num_node, window_size, 1])
		batch_y = y_train[offset:(offset + batch_size), :]
		# print("Batch X shape:")
		# print(batch_x.shape)
		# print("Batch y shape:")
		# print(batch_y.shape)
		# exit()
		_, c = sess.run([optimizer, cost], feed_dict={X: batch_x, Y: batch_y, keep_prob: 1-dropout_prob, train_phase: True})
	# calculate train and test accuracy after each training epoch
	if(epoch%1 == 0):
		train_accuracy 	= np.zeros(shape=[0], dtype=float)
		test_accuracy	= np.zeros(shape=[0], dtype=float)
		train_l 		= np.zeros(shape=[0], dtype=float)
		test_l			= np.zeros(shape=[0], dtype=float)
		# calculate train accuracy after each training epoch
		for i in range(batch_num_per_epoch):
			offset = (i * batch_size) % (y_train.shape[0] - batch_size) 
			train_batch_x = features_train[offset:(offset + batch_size), :, :, :]
			train_batch_x = train_batch_x.reshape([len(train_batch_x)*num_timestep, num_node, window_size, 1])
			train_batch_y = y_train[offset:(offset + batch_size), :]

			train_a, train_c = sess.run([accuracy, cost], feed_dict={X: train_batch_x, Y: train_batch_y, keep_prob: 1.0, train_phase: True})
			
			train_l = np.append(train_l, train_c)
			train_accuracy = np.append(train_accuracy, train_a)
		print("("+time.asctime(time.localtime(time.time()))+") Epoch: ", epoch+1, " Training Cost: ", np.mean(train_l), "Training Accuracy: ", np.mean(train_accuracy))
		history_train.append([epoch, np.mean(train_l), np.mean(train_accuracy)])
		# calculate test accuracy after each training epoch
		ctr = 0
		for j in range(batch_num_per_epoch):
			# print("Test X shape:")
			# print(features_test.shape)
			# print("Test y shape:")
			# print(test_y.shape)
			offset = (j * batch_size) % (test_y.shape[0] - batch_size)
			# print(offset) 
			test_batch_x = features_test[offset:(offset + batch_size), :, :, :]
			test_batch_x = test_batch_x.reshape([len(test_batch_x)*num_timestep, num_node, window_size, 1])
			test_batch_y = y_test[offset:(offset + batch_size), :]
			if test_batch_x.shape[0] != test_batch_y.shape[0]:
				# print("Test batch X shape:")
				# print(test_batch_x)
				# print("Test batch y shape:")
				# print(test_batch_y)
				# print("Passed!", ctr)
				ctr += 1
				continue
			# print("Test batch X shape:")
			# print(test_batch_x.shape)
			# print("Test batch y shape:")
			# print(test_batch_y.shape)
			# exit()
			test_a, test_c, test_p = sess.run([accuracy, cost, y_posi], feed_dict={X: test_batch_x, Y: test_batch_y, keep_prob: 1.0, train_phase: True})
			
			test_accuracy = np.append(test_accuracy, test_a)
			test_l = np.append(test_l, test_c)
			true_test.append(test_batch_y)
			posi_test.append(test_p)
			# test_cm = test_cm + confusion_matrix(test_batch_y.argmax(axis=1), test_p.argmax(axis=1))
			# print(test_cm)
		print("Number of passed batches:", ctr)
		auc_roc_test = roc_auc_score(y_true=np.array(true_test).reshape([-1, 2]), y_score = np.array(posi_test).reshape([-1, 2]))
		print("("+time.asctime(time.localtime(time.time()))+") Epoch: ", epoch+1, "Test AUC: ", auc_roc_test, " Test Cost: ", np.mean(test_l), "Test Accuracy: ", np.mean(test_accuracy), "\n")
		
		history_test.append([epoch, np.mean(test_l), np.mean(test_accuracy), auc_roc_test])

# print(history_train)
# print(history_test)
# print(test_p.argmax(axis=1))
# print(test_batch_y.argmax(axis=1))


# print(cm)

# pickle.dump(cm, open("cm.pickle", "wb"))
pickle.dump(history_train, open("history_train.pickle", "wb"))
pickle.dump(history_test, open("history_test.pickle", "wb"))
