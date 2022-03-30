# Colorization of Grey Images by applying a Convolutional Autoencoder on the Jetson Nano
## by Dennis Konkol and Tim Niklas Witte

This repository contains an pretrainied convolutional autoencoder for colorization of grey images.
The live camera stream will be colorizatized in real time.
The architecture of the ANN is optimized to run on the Jetson Nano.
In total, 10 FPS can be archived on this embedded GPU.

![Example Video](videoPresentation.gif)

## Requirements

- TensorFlow 2
- OpenCV 3.3.1

## Usage

### Training

Run `Training.py` to start the training of the model.
Each epoch the weights are stored into `./saved_models`.
Besides, in `./test_logs` are the corresponding trainings statistics (train and test loss and also a batch of colorized test images) logged.

```bash
python3 Training.py
```

### Live colorization

The launch of `live_recolor_plot.py` opens a window as shown in the GIF at the start of this README.

```bash
python3 live_recolor.py
```

It has the following structure:

```bash
(1) | (2) | (3) | (4)

(1) = live RGB camera image
(2) = live grey camera image
(3) = live colorized image
```

To get also displayed a loss plot (mean squared error between `(1)` and `(3)`),
run `live_recolor_plot.py` instead.
The loss plot is presented right from `(3)`.

```bash
python3 live_recolor_plot.py
```

### Pretrainied Model