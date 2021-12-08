import tensorflow as tf
import numpy as np
from tensorflow import keras

class Encoder(keras.Model):

    def __init__(self, hidden_dim=128):
        super(Encoder, self).__init__()
        self.flatten = keras.layers.Flatten()
        self.fc = keras.layers.Dense(hidden_dim, activation='relu')
    
    def call(self, x):
        flattened = self.flatten(x)
        return self.fc(flattened)

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
        k = self.fck(x)
        scores = tf.matmul(q, k, transpose_b=True)
        att = tf.nn.softmax(scores - np.inf * (1 - mask), axis=None)
        
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
    def __init__(self, hidden_dim, num_actions, lr = 0.0001):
        super(DGN, self).__init__()
        
        self.encoder = Encoder(hidden_dim)
        self.att_1 = AttModel(hidden_dim, hidden_dim)
        self.att_2 = AttModel(hidden_dim, hidden_dim)
        self.q_net = Q_Net(num_actions)
        self.optimizer = keras.optimizers.Adam(learning_rate=lr)
        
    def call(self, x, mask):
        h1 = self.encoder(x)
        h2 = self.att_1(h1, mask)
        h3 = self.att_2(h2, mask)
        q = self.q_net(tf.concat([h1, h2, h3], axis=0))
        return q
        
class AgentModel:
    def __init__(self, hidden_dim, num_actions, lr = 0.0001):
        self.model = self.build_model(hidden_dim, num_actions, lr)
        self.target_model = self.build_model(hidden_dim, num_actions, lr)
        self.update_target_model()

    def build_model(self, hidden_dim, num_actions, lr = 0.0001):
        model = DGN(hidden_dim, num_actions, lr)
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def load(self, name):
        self.model.load_weights(name)
        self.update_target_model()

    def save(self, name):
        self.target_model.save_weights(name)
