from sklearn.utils import shuffle
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np
import tqdm

from Decoder import *

import os

from Autoencoder import Autoencoder
import tensorflow_io as tfio

def getRGB(L, AB, batch_mode=True):
    # Remove normalization
    L = (L + 1)*50
    AB = ((AB - 1)*255/2)+128

    if batch_mode:
        L = tf.reshape(L, (32, 256,256,1))
        LAB = tf.concat([L, AB], 3)
    else:
        L = tf.reshape(L, (256,256,1))
        LAB = tf.concat([L, AB], 2)
    rgb = tfio.experimental.color.lab_to_rgb(LAB)

    return rgb

def main():
    

    labels_path = tf.keras.utils.get_file('ImageNetLabels.txt','https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
    imagenet_labels = np.array(open(labels_path).read().splitlines())

    data_dir = '/home/timwitte/Downloads/'
    write_dir = './imagenet'

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
    
    train_dataset = train_dataset.apply(prepare_data)
    test_dataset = test_dataset.apply(prepare_data).take(500) # take 500 batches
    

    # for L, AB in train_dataset.take(1):
        
    #     print(L.shape)
    #     print(AB.shape)

    #     print(np.min(L[0]))
    #     print(np.max(L[0]))
    #     print("######################")
    #     print(np.min(AB[0]))
    #     print(np.max(AB[0]))

    #     rgb = getRGB(L, AB)
        
    #     plt.imshow(rgb[0])
    #     plt.show()
   
    #     exit()

    autoencoder = Autoencoder()
    num_epochs = 75

    file_path = "test_logs/test"
    summary_writer = tf.summary.create_file_writer(file_path)

    for img_L_tensorBoard, img_AB_tensorBoard in test_dataset.take(1):
        pass

    with summary_writer.as_default():

        tf.summary.image(name="grey_images",data = img_L_tensorBoard, step=0, max_outputs=32)
        img_RBG = getRGB(img_L_tensorBoard, img_AB_tensorBoard)
        tf.summary.image(name="colored_images",data = img_RBG, step=0, max_outputs=32)
        
        imgs = autoencoder(img_L_tensorBoard)
        tf.summary.image(name="recolored_images",data = imgs, step=0, max_outputs=32)

        autoencoder.summary()
      
        train_loss = autoencoder.test(train_dataset.take(100))
        
        tf.summary.scalar(name="Train loss", data=train_loss, step=0)

        test_loss = autoencoder.test(test_dataset)
        tf.summary.scalar(name="Test loss", data=test_loss, step=0)


        for epoch in range(num_epochs):
            
            print(f"Epoch {epoch}")
           
        
            for img_L, img_AB in tqdm.tqdm(train_dataset,position=0, leave=True): 
                autoencoder.train_step(img_L, img_AB)
               
             
            tf.summary.scalar(name="Train loss", data=autoencoder.metric_mean.result(), step=epoch+1)
            autoencoder.metric_mean.reset_states()

            test_loss = autoencoder.test(test_dataset)
            tf.summary.scalar(name="Test loss", data=test_loss, step=epoch+1)

            img_AB = autoencoder(img_L_tensorBoard)

            img_RBG = getRGB(img_L_tensorBoard, img_AB)

            tf.summary.image(name="recolored_images",data = img_RBG, step=epoch + 1, max_outputs=32)
            
            # save model
            autoencoder.save_weights(f"./saved_models/trainied_weights_epoch_{epoch}", save_format="tf")

def prepare_data(data):

    # Remove label 
    data = data.map(lambda img, label: img )

    # resize
    data = data.map(lambda img: tf.image.resize(img, [256,256]) )

    #convert data from uint8 to float32
    data = data.map(lambda img: tf.cast(img, tf.float32) )

    # tfio.experimental.color.rgb_to_lab expects its input to be a float normalized between 0 and 1.
    data = data.map(lambda img: (img/255.) )
    data = data.map(lambda img: tfio.experimental.color.rgb_to_lab(img) )

    # X = L channel
    # Y = (A,B) channel
    data = data.map(lambda img: (img[:, :, 0], tf.stack([img[:, :, 1], img[:, :, 2]], axis=2)))

    # Reshape R channel -> grey
    data = data.map(lambda L, AB: ( tf.reshape(L, shape=(256,256,1)) , AB))  
    
    # Normalize between [-1, 1]
    data = data.map(lambda L, AB: ( (L/50.0) - 1., 1 + (2*(AB - 128)/255) ))  
    
    # add gray scaled image
    #data = data.map(lambda img: (tf.image.rgb_to_grayscale(img), img))

    #cache this progress in memory, as there is no need to redo it; it is deterministic after all
    #data = data.cache("cachefile")

    #shuffle, batch, prefetch
    data = data.shuffle(7000) 
    data = data.batch(32)

    AUTOTUNE = tf.data.AUTOTUNE
    data = data.prefetch(AUTOTUNE)
    #return preprocessed dataset
    return data

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")