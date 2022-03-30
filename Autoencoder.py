import tensorflow as tf
import numpy as np

from Encoder import * 
from Decoder import *

class Autoencoder(tf.keras.Model):
    def __init__(self):
        super(Autoencoder, self).__init__()

        self.encoder = Encoder()
        self.decoder = Decoder()

        self.loss_function = tf.keras.losses.MeanSquaredError()

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)


    @tf.function
    def call(self, x, training=False):
        embedding = self.encoder(x, training)
        decoded = self.decoder(embedding, training)
        return decoded

    @tf.function
    def train_step(self, input, target):

        with tf.GradientTape() as tape:
            prediction = self(input, training=True)
            loss = self.loss_function(target, prediction)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return loss

    def test(self, test_data):
        # test over complete test data
        test_loss_aggregator = []
        for input, target in test_data: # ignore label            
            prediction = self(input)
            
            sample_test_loss = self.loss_function(target, prediction)
            test_loss_aggregator.append(sample_test_loss.numpy())

        test_loss = tf.reduce_mean(test_loss_aggregator)
        return test_loss