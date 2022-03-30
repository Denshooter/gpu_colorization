import tensorflow as tf

from EncoderLayers import *
from DecoderLayers import *

import sys
sys.path.append("../..")

from Colorful_Image_Colorization.model import *

def main():

    encoder_layers = EncoderLayers()
    decoder_layers = DecoderLayers()

    inputs = tf.keras.Input(shape=(256,256, 1), name="Grey image") 
    encoder = tf.keras.Model(inputs=[inputs],outputs=encoder_layers.call(inputs))

    embedding = tf.keras.Input(shape=(32,32, 3), name="Embedding") 
    decoder = tf.keras.Model(inputs=[embedding],outputs=decoder_layers.call(embedding))

    tf.keras.utils.plot_model(encoder,show_shapes=True, show_layer_names=True, to_file="EncoderLayer.png")
    tf.keras.utils.plot_model(decoder,show_shapes=True, show_layer_names=True, to_file="DecoderLayer.png")
    
    ModelToCompare_layers = build_model()
    modelToCompare = tf.keras.Model(inputs=[inputs],outputs=ModelToCompare_layers.call(inputs))
    tf.keras.utils.plot_model(modelToCompare,show_shapes=True, show_layer_names=True, to_file="ModelToCompare.png")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")