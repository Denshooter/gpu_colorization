import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np
import os

import sys
sys.path.append("..")
from Autoencoder import Autoencoder
from Training import prepare_data, getRGB

def main():

    labels_path = tf.keras.utils.get_file('ImageNetLabels.txt','https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
    imagenet_labels = np.array(open(labels_path).read().splitlines())

    data_dir = '/home/timwitte/Downloads/'
    write_dir = '../imagenet'

    # Construct a tf.data.Dataset
    download_config = tfds.download.DownloadConfig(
                        extract_dir=os.path.join(write_dir, 'extracted'),
                        manual_dir=data_dir
                    )
    download_and_prepare_kwargs = {
        'download_dir': os.path.join(write_dir, 'downloaded'),
        'download_config': download_config,
    }

    train_dataset, test_dataset= tfds.load('imagenet2012', 
                data_dir=os.path.join(write_dir, 'data'),         
                split=['train', 'validation'], 
                shuffle_files=True, 
                download=True,
                as_supervised=True,
                download_and_prepare_kwargs=download_and_prepare_kwargs)
    
    test_dataset = test_dataset.take(32).apply(prepare_data)

    autoencoder = Autoencoder()

    autoencoder.build((1, 256, 256, 1)) # need a batch size
    autoencoder.load_weights("../saved_models/trainied_weights_epoch_12")
    autoencoder.summary()
    
    autoencoder.encoder.summary()
    autoencoder.decoder.summary()

    for img_L, img_AB_orginal in test_dataset.take(1):

        img_AB_reconstructed = autoencoder(img_L)

        img_rgb_orginal = getRGB(img_L, img_AB_orginal)
        img_rgb_reconstructed = getRGB(img_L, img_AB_reconstructed)
        
        NUM_IMGS = 5
        fig, axs = plt.subplots(NUM_IMGS, 3)

        axs[0, 0].set_title("Input", fontsize=30)
        axs[0, 1].set_title("Output", fontsize=30)
        axs[0, 2].set_title("Ground Truth", fontsize=30)

        for i in range(NUM_IMGS):
            
            axs[i, 0].imshow(img_L[i], cmap="gray")
            axs[i, 0].set_axis_off()

            axs[i, 1].imshow(img_rgb_reconstructed[i])
            axs[i, 1].set_axis_off()

            axs[i, 2].imshow(img_rgb_orginal[i])
            axs[i, 2].set_axis_off()
        
        plt.tight_layout()

        fig.set_size_inches(15, 25)
        fig.savefig("ColoredImages.png")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")