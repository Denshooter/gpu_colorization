# Colorization of Grey Images by applying a Convolutional Autoencoder on the Jetson Nano
## by Dennis Konkol and Tim Niklas Witte

This repository contains an pretrained convolutional autoencoder on the `imagenet2012` dataset 
for colorization of grey images.
The live camera stream will be colorizatized in real time.
The architecture of the ANN is optimized to run on the Jetson Nano.
It has 300.000 parameters.
In total, 10 FPS can be archived on this embedded GPU.

![Example Video](videoPresentation.gif)

The `./Paper_GPU_colorization.pdf` file represents the entire documentation of this project in form of a paper.
All corresponding files including the `.tex` file and figures are stored in `./Paper`.

This project was created as a part of the GPU programming course of Mario Porrmann in the winter term 2021/22 at the Osnabrück University.

## Requirements

- TensorFlow 2
- OpenCV 3.3.1
- CSI camera must be plugged in (see code of `live_recolor[_plot].py`)

## Model

```bash
 Model: "autoencoder"
_______________________________________________________________
 Layer (type)                Output Shape            Param #   
===============================================================
 encoder (Encoder)           multiple                148155    
                                                               
 decoder (Decoder)           multiple                150145    
                                                                 
===============================================================
Total params: 298,302
Trainable params: 297,210
Non-trainable params: 1,092
_______________________________________________________________
```

### Hyperparameters

```python3
optimizer = Adam
learning rate = 0.0001
loss function = mean squared error
batch size = 32
```

## Usage

### Training

Run `Training.py` to start the training of the model on the `imagenet2012` dataset.
Each epoch the weights are stored into `./saved_models`.
Besides, in `./test_logs` are the corresponding trainings statistics (train and test loss and also a batch of colorized test images) logged.
Note that, the `imagenet2012` dataset must be stored in `./imagenet` as described in this [blog article](https://towardsdatascience.com/preparing-the-imagenet-dataset-with-tensorflow-c681916014ee).
This includes a change of the variable `data_dir` at line 36.

```bash
python3 Training.py
```

### Live colorization

The launch of `live_recolor_plot.py` opens a window as shown in the GIF at the start of this README.
Note that, the CSI camera must be plugged in.

```bash
python3 live_recolor.py
```

It has the following structure:

```bash
(1) | (2) | (3)

(1) = live RGB camera image
(2) = live grey camera image
(3) = live colorized image
```

In order to get additionally displayed a loss plot (mean squared error between `(1)` and `(3)`),
run `live_recolor_plot.py` instead.
The loss plot is presented right from `(3)`.

```bash
python3 live_recolor_plot.py
```

### Pretrained Model

The model was runned for 13 epochs on the `imagnet2012` dataset and its weights are stored in `./saved_models`.
Note that, grey images must have a shape of `(256,256,1)`.
The following code will load the pretrained model and colorized an image:

```python3
autoencoder = Autoencoder()
autoencoder.build((1, 256, 256, 1)) # need a batch size
autoencoder.load_weights("./saved_models/trainied_weights_epoch_12")
autoencoder.summary()

grey_img = ... # grey_img.shape = (256,256,1)
grey_img = np.expand_dims(grey_img, axis=0) # add batch dim
ab_img = autoencoder(grey_img) # get ab values

rbg_img = getRGB(grey_img, ab_img) # contained in Main.py
```
