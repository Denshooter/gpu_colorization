import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt


import sys
sys.path.append("..")

from Autoencoder import Autoencoder
from Training import prepare_data, getRGB

import numpy as np
import os


#from Training import prepare_data, getRGB

from Colorful_Image_Colorization.model import build_model
from Colorful_Image_Colorization.config import img_rows, img_cols
from Colorful_Image_Colorization.config import nb_neighbors, T, epsilon
import cv2 as cv

def main():

    # Create Imagenet
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

    # Load our model
    model_our = Autoencoder()
    model_our.build((1, 256, 256, 1)) # need a batch size
    model_our.load_weights("../saved_models/trainied_weights_epoch_12")

    # Load model to compare
    model_weights_path = '../Colorful_Image_Colorization/model.06-2.5489.hdf5'
    model_toCompare = build_model()
    model_toCompare.load_weights(model_weights_path)

    loss_function = tf.keras.losses.MeanSquaredError()

    for img_L, img_AB_orginal in test_dataset.take(1):

        img_rgb_orginal = getRGB(img_L, img_AB_orginal)

        img_AB_reconstructed_our = model_our.predict(img_L.numpy())
        img_rgb_reconstructed_our = getRGB(img_L, img_AB_reconstructed_our)

        NUM_IMGS = 5
        fig, axs = plt.subplots(NUM_IMGS, 4)

        axs[0, 0].set_title("Input", fontsize=30)
        axs[0, 1].set_title("Richard Zhang $\it{et\ al.}$", fontsize=30,)
        axs[0, 2].set_title("Ours", fontsize=30)
        axs[0, 3].set_title("Ground Truth", fontsize=30)
        losses1 = []
        losses2 = []
        for i in range(NUM_IMGS):

            img_AB_reconstructed_toCompare = getABFromModel(model_toCompare, img_L[i].numpy())
            img_rgb_reconstructed_toCompare = getRGB(img_L[i], img_AB_reconstructed_toCompare, batch_mode=False)

            axs[i, 0].imshow(img_L[i], cmap="gray")
            axs[i, 0].set_axis_off()

            axs[i, 1].imshow(img_rgb_reconstructed_toCompare)
            axs[i, 1].set_axis_off()

            axs[i, 2].imshow(img_rgb_reconstructed_our[i])
            axs[i, 2].set_axis_off()

            axs[i, 3].imshow(img_rgb_orginal[i])
            axs[i, 3].set_axis_off()

            loss_our = loss_function(img_rgb_orginal[i], img_rgb_reconstructed_our[i])
            loss_toCompare = loss_function(img_rgb_orginal[i], img_rgb_reconstructed_our)

            losses1.append(loss_our)
            losses2.append(loss_toCompare)
       

        plt.tight_layout()

        fig.set_size_inches(20, 25)
        fig.savefig("ColoredImages_compareModels.png")

        # Reset plot
        plt.clf()
        plt.cla()
        fig = plt.figure()

        # Create bar plot
        x_axis = np.arange(NUM_IMGS)
        width = 0.2
        plt.bar(x_axis - width/2., losses2, width=width/2, label = "Richard Zhang $\it{et\ al.}$")
        plt.bar(x_axis - width/2. + 1/float(2)*width, losses1, width=width/2, label = 'Ours')
       
        
        plt.xticks(x_axis,[f"No. {i}" for i in range(NUM_IMGS)])
        
        plt.title("Loss of colorized images")
        plt.xlabel("Image")
        plt.ylabel("Loss")

        plt.legend()
        plt.tight_layout()
        plt.savefig("ColorizedImagesLossPlot_comparedModels.png")

 


def getABFromModel(model, grey_img):
    # code taken from https://github.com/foamliu/Colorful-Image-Colorization/blob/master/demo.py
    q_ab = np.load("../Colorful_Image_Colorization/pts_in_hull.npy")
    nb_q = q_ab.shape[0]

    grey_img = np.expand_dims(grey_img, axis=0)
    
    X_colorized = model.predict((grey_img+1)/2)

    
    h, w = img_rows // 4, img_cols // 4
    X_colorized = X_colorized.reshape((h * w, nb_q))

    # Reweight probas
    X_colorized = np.exp(np.log(X_colorized + epsilon) / T)
    X_colorized = X_colorized / np.sum(X_colorized, 1)[:, np.newaxis]

    # Reweighted
    q_a = q_ab[:, 0].reshape((1, 313))
    q_b = q_ab[:, 1].reshape((1, 313))

    X_a = np.sum(X_colorized * q_a, 1).reshape((h, w))
    X_b = np.sum(X_colorized * q_b, 1).reshape((h, w))

    X_a = cv.resize(X_a, (img_rows, img_cols), cv.INTER_CUBIC)
    X_b = cv.resize(X_b, (img_rows, img_cols), cv.INTER_CUBIC)

    # Before: -90 <=a<= 100, -110 <=b<= 110
    # After: 38 <=a<= 228, 18 <=b<= 238
    X_a = X_a + 128
    X_b = X_b + 128
  
    out_lab = np.zeros((256, 256, 2), dtype=np.float32)
    grey_img = np.reshape(grey_img, newshape=(256,256))
  

    out_lab[:, :, 0] = X_a
    out_lab[:, :, 1] = X_b
 
    out_lab[:, :, 0] = -1.0 + 2*(out_lab[:, :, 0] - 38.0)/190
    out_lab[:, :, 1] = -1.0 + 2*(out_lab[:, :, 1] - 20.0)/203

    return out_lab

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")