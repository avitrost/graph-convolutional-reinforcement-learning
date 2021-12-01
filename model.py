import tensorflow as tf
import numpy as np
from tensorflow import keras

class Encoder(keras.Model):

    def __init__(self, hidden_dim=128):
        super(Encoder, self).__init__()
        self.fc = keras.layers.Dense(hidden_dim, activation='relu')
    
    def call(self, x):
        return self.fc(x)

class AttModel(keras.Model):
    def __init__(self, hidden_dim, output_dim):
        super(AttModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.fcv = keras.layers.Dense(hidden_dim, activation='relu')
        self.fck = keras.layers.Dense(hidden_dim, activation='relu')
        self.fcq = keras.layers.Dense(hidden_dim, activation='relu')
        self.fcout = keras.layers.Dense(output_dim, activation='relu')
        
    def call(self, x, mask):
        v = self.fcv(x)
        q = self.fcq(x)
        k = tf.transpose(self.fck(x), perm=(0,2,1))
        att = tf.nn.softmax(tf.matmul(tf.matmul(q, k), mask) - np.inf * (1 - mask), axis=2)
        
        out = tf.matmul(att, v) / np.sqrt(self.hidden_dim) # Might not be needed
		# out = torch.add(out,v)
        out = self.fcout(out)
        return out

class Q_Net(keras.Model):
	def __init__(self, output_dim):
		super(Q_Net, self).__init__()
		self.fc = keras.layers.Dense(output_dim)

	def call(self, x):
		q = self.fc(x)
		return q

class DGN(keras.Model):
	def __init__(self, hidden_dim, num_actions):
		super(DGN, self).__init__()
		
		self.encoder = Encoder(hidden_dim)
		self.att_1 = AttModel(hidden_dim, hidden_dim)
		self.att_2 = AttModel(hidden_dim, hidden_dim)
		self.q_net = Q_Net(num_actions)
		
	def call(self, x, mask):
		h1 = self.encoder(x)
		h2 = self.att_1(h1, mask)
		h3 = self.att_2(h2, mask)
		q = self.q_net(h3)
		return q 
