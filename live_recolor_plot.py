import cv2 as cv
from Autoencoder import Autoencoder
import helper
import tensorflow as tf
import numpy as np
import os

# Height and width of each frame
HEIGHT = 256
WIDTH = 256

# Text on the pictures
font = cv.FONT_HERSHEY_SIMPLEX
position = (0,HEIGHT-10)
fontScale = 1
fontColor = (255,0,255)


def load_model(weight_path):
    autoencoder = Autoencoder()
    # need a batch size
    autoencoder.build((1, 256, 256, 1))
    # load model weights
    autoencoder.load_weights(weight_path)
    return autoencoder


def addTextToFrame(frame, text):
    return (
        cv.putText(frame,text,
        position,
        font,
        fontScale,
        fontColor)
    )


def main(pipeline, autoencoder):
    # capture camera feed
    if pipeline:
        cap = cv.VideoCapture(pipeline, cv.CAP_GSTREAMER)
    else:
        cap = cv.VideoCapture(0)
    # array with 30 values for loss display
    loss_array = np.zeros(31)    
    # loop
    frame = 1
    while True:
        # get frame
        ret, frame = cap.read()
        if frame is None:
            break
        frame = cv.resize(frame, (256,256))

        # make grayscale image
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # add back the channel, so concatination works
        three_value_gray = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)
        
        L, AB = helper.to_LAB(frame)
        
        # add batch dimension
        L = tf.expand_dims(L, 0)
        # predict the colored image from gray
        ab_img = autoencoder(L)
        
        recolored_image = helper.suitable_rgb(L, ab_img)
        
        
        # add text to the frame
        ogFrame =  addTextToFrame(frame, 'Original')
        three_value_gray =  addTextToFrame(three_value_gray, 'Grayscale')
        recolored_image =  addTextToFrame(recolored_image, 'Recolored')

        #get loss
        loss = autoencoder.loss_function(AB, ab_img)
        # get only the loss value
        loss = loss.numpy()

        # add loss to array
        loss_array = loss_array[1:]
        loss_array = np.append(loss_array, loss)
        
        helper.create_Plot(loss_array)

        # grab fps counter at to the last frame
        fps = cap.get(cv.CAP_PROP_FPS)
        cv.putText(recolored_image, "FPS: {:.2f}".format(fps), (150, 20), font,0.5,fontColor)

        # load loss graph
        loss_graph = cv.imread('loss.png')
        # make 256,256 image
        loss_graph = cv.resize(loss_graph, (256,256))

        # connect the three frames into one
        im_h = cv.hconcat([ogFrame, three_value_gray, recolored_image, loss_graph])
        # display it
        cv.imshow("Live Recoloration", im_h)
        # quit on ESC
        if cv.waitKey(1) == 27 :
            break
    # delete loss graph
    os.remove('loss.png')
    # release camera
    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':

    # create the gstreamer pipeline for picking the camera's data
    pipeline = None

    autoencoder = load_model(weight_path="./saved_models/trainied_weights_epoch_12")
    try:
        main(pipeline, autoencoder)
    except KeyboardInterrupt:
        print("[!] Exiting program . . .")
        # delete loss graph
        if os.path.exists('loss.png'):
            os.remove('loss.png')
        exit(1)