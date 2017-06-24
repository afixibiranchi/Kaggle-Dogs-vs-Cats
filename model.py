import tensorflow as tf

class Model:

	def weight_variable(self,shape):
		initial = tf.random_normal(shape, stddev=0.5)
		return tf.Variable(initial)

	def bias_variable(self,shape):
		initial = tf.random_normal(shape, stddev = 0.1)
		return tf.Variable(initial)

	def conv2d(self,x,W):
		return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

	def max_pool(self,x):
		return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")


	n_hidden_1 = 300
	n_input = 16000
	n_classes = 2

	def multilayer_perceptron(self,x, weights, biases):

	    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
	    layer_1 = tf.nn.relu(layer_1)

	    out_layer = tf.matmul(layer_1, weights['out']) + biases['out']
	    return out_layer

	weights = {
	    'h1' : tf.Variable(tf.random_normal([n_input,n_hidden_1])),
	    'out': tf.Variable(tf.random_normal([n_hidden_1, n_classes])) 
	}

	biases = {
	    'b1' : tf.Variable(tf.random_normal([n_hidden_1])),
	    'out': tf.Variable(tf.random_normal([n_classes]))
	}

	def __init__(self):

		
		self.x = tf.placeholder(tf.float32, shape=[None, 128,128])
		self.y_= tf.placeholder(tf.float32, shape=[None, 2])
		# keep_prob = tf.placeholder(tf.float32)


		W_conv1 = self.weight_variable([9,9,1,8])
		b_conv1 = self.bias_variable([8])

		x_image = tf.reshape(self.x, [-1, 128, 128, 1])
		h_conv1 = tf.nn.relu(self.conv2d(x_image ,W_conv1) + b_conv1)
		h_pool1 = self.max_pool(h_conv1)

		W_conv2 = self.weight_variable([7,7,8,16])
		b_conv2 = self.bias_variable([16])

		h_conv2 = tf.nn.relu(self.conv2d(h_pool1, W_conv2) + b_conv2)
		h_pool2 = self.max_pool(h_conv2)

		W_conv3 = self.weight_variable([5,5,16,32])
		b_conv3 = self.bias_variable([32])

		h_conv3 = tf.nn.relu(self.conv2d(h_pool2, W_conv3) + b_conv3)
		h_pool3 = self.max_pool(h_conv3)

		W_conv4 = self.weight_variable([3,3,32,64])
		b_conv4 = self.bias_variable([64])

		h_conv4 = tf.nn.relu(self.conv2d(h_pool3, W_conv4) + b_conv4)
		h_pool4 = self.max_pool(h_conv4)

		W_conv5 = self.weight_variable([3,3,64,250])
		b_conv5 = self.bias_variable([250])

		h_conv5 = tf.nn.relu(self.conv2d(h_pool4, W_conv5) + b_conv5)

		h_pool5_flat = tf.reshape(h_conv5, [-1,8*8*250])

		

		self.pred = self.multilayer_perceptron(h_pool5_flat, self.weights, self.biases)

		cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.pred, labels=self.y_))
		self.train_step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
		correct_prediction = tf.equal(tf.argmax(self.pred, 1), tf.argmax(self.y_, 1))
		self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
		

		self.prediction = tf.argmax(self.pred,1)





