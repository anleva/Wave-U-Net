# Wave-U-Net

Daniel Stoller, Sebastian Ewert and Simon Dixon proposed in [Wave-U-Net](https://arxiv.org/abs/1806.03185) to use the well-known U-Net to separate the singing voice from the musical accompaniment, directly in the raw audio domain.

The architecture is a succession downsamling blocks with convolutional layers, followed by a corresponding upsampling blocks, and with skip connections between the up- and downsampling sequences.

![wave_u_net_overview](https://github.com/anleva/Wave-U-Net/blob/master/img/wave_u_net.jpg)

The training data is from the [MUSDB18 dataset](https://sigsep.github.io/datasets/musdb.html), which consists of 150 professionally recorded music tracks. Each track has four separate stems; vocals, drums, bass and other. There is also the mixed version, where all stems have been merged into one track.

## Implementation details

While MUSDB18 has been created with the intention of source separation into four separate stems, the authors of Wave-U-Net notes that the performance was considerably better for the simpler task of separating vocals from the instrumental music. I therefore limit the model to only separating vocals from instrumental music, which means that two input channels are separated into four, as the data is in stereo. Changing the model to separate all four stems is easy; change Wnet.stems from 2 to 4.

In a first setup [wave_u_net.py](https://github.com/anleva/Wave-U-Net/blob/master/wave_u_net.py), I made a Keras implementation that is very close to the original, with two main differences.
* The original implementation used data in 22,050 Hz, whereas I use 16k.
* The original implementation does not use any regularization, whereas I use an L2 kernel regularization throughout the network.

Despite using a very similar setup and training schedule, the model described in wave_u_net.py did not converge to a satisfactory level.
Therefore, I did the following changes, which are available in [wave_u_net_shallow.py](https://github.com/anleva/Wave-U-Net/blob/master/wave_u_net_shallow.py).

* The U-Net was made shallower. I used a depth of 7 downsampling blocks + 7 upsampling blocks, instead of the 12+12 blocks used in the original version.
* In order to maintain the same receptive field when using fewer blocks, I used a pooling factor of 4 (with no pooling in the first layer), instead of 2.
* The kernel size was increased from 15 to 31 during down-sampling and from 5 to 15 during up-sampling.
* I switched to using max-pooling for the down-sampling. The original decimated by dropping every second data point in the time dimension.
* The number of features in each layer was doubled.
* Batch Normalization was added throughout the U-Net.

The wave_u_net_shallow.py did get much better results than wave_u_net.py. However, the author's implementation performed better than what I achieved on my two trainings. 
The sound source predictions produced by wave_u_net_shallow.py explained around 80% of the actual sound stems' standard deviation.

### Training
[wave_u_net_shallow](https://github.com/anleva/Wave-U-Net/blob/master/wave_u_net_shallow.py) has around 40m parameters. Training could not be done on my laptop, but was instead done on Google Cloud Platform, using their generous $300 free credit for new accounts.

### Prerequisites

See requirements.txt.

## Authors

* **Anders Levander** - *Draft version* - [anleva/Wave-U-Net](https://github.com/anleva/Wave-U-Net)

## License

This project is licensed under the MIT License.

## Acknowledgments

* Wave-U-Net is based on [Wave-U-Net: A Multi-Scale Neural Network for End-to-End Audio Source Separation](https://arxiv.org/abs/1806.03185), by Daniel Stoller, Sebastian Ewert and Simon Dixon.
* Data is from the [MUSDB18](https://doi.org/10.5281/zenodo.1117372) (or [alt url](https://sigsep.github.io/datasets/musdb.html)) corpus for music separation. Authors: Zafar Rafii, Antoine Liutkus, Fabian-Robert Stöter, Stylianos Ioannis Mimilakis, and Rachel Bittner.
