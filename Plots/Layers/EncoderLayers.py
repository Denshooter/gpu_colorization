import tensorflow as tf

class EncoderLayers(tf.keras.Model):
    def __init__(self):
        super(EncoderLayers, self).__init__()

        self.layer_list = [
            tf.keras.layers.Conv2D(75, kernel_size=(3, 3), strides=2, padding='same', name="Conv2D_0"),
            tf.keras.layers.BatchNormalization(name="BatchNormalization_0"),
            tf.keras.layers.Activation(tf.nn.tanh, name="tanh_0"),

            tf.keras.layers.Conv2D(90, kernel_size=(3, 3), strides=2, padding='same', name="Conv2D_1"),
            tf.keras.layers.BatchNormalization(name="BatchNormalization_1"),
            tf.keras.layers.Activation(tf.nn.tanh, name="tanh_1"),

            tf.keras.layers.Conv2D(105, kernel_size=(3, 3), strides=2, padding='same',name="Conv2D_2"),
            tf.keras.layers.BatchNormalization(name="BatchNormalization_2"),
            tf.keras.layers.Activation(tf.nn.tanh, name="tanh_2"),

            tf.keras.layers.Conv2D(3, kernel_size=(1, 1), strides=1, padding='same', name="Conv2D_3"),
            tf.keras.layers.BatchNormalization(name="BatchNormalization_3"),
            tf.keras.layers.Activation(tf.nn.tanh, name="tanh_3"),
        ]
        
    
    def call(self, x):
        for layer in self.layer_list:
            x = layer(x)
        return x