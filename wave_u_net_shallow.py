import os
import numpy as np
import random
import datetime
from keras.layers import MaxPooling1D, AveragePooling1D, Concatenate, UpSampling1D, Cropping1D
from keras.layers import Input, Conv1D, LeakyReLU, BatchNormalization
from keras.layers import Add, Subtract
from keras.regularizers import l2
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.losses import mean_squared_error
import keras.backend as K
import json


class Wnet:

    def __init__(self):
        self.wave_net_bn_shallow_model = None
        self.data_sources = None
        self.data_mixes = None
        self.data_sources_mini = None
        self.data_mixes_mini = None
        self.data_path = 'data_2_stems'
        self.models_path = 'models'
        self.examples_path = 'examples'
        self.sound_length = 16384 * 14  # approx 14 sec of 16k bit rate (but 16384 to have a multiple of 4)
        self.input_length = None
        self.stems = 2
        self.channels_per_stem = 2

        self.u_net = [
            {'f': 2*24*1, 'p': 1, 'k_down': 31, 'k_up': 15},
            {'f': 2*24*1, 'p': 4, 'k_down': 31, 'k_up': 15},
            {'f': 2*24*3, 'p': 4, 'k_down': 31, 'k_up': 15},
            {'f': 2*24*5, 'p': 4, 'k_down': 31, 'k_up': 15},
            {'f': 2*24*7, 'p': 4, 'k_down': 31, 'k_up': 15},
            {'f': 2*24*9, 'p': 4, 'k_down': 31, 'k_up': 15},
            {'f': 2*24*11, 'p': 4, 'k_down': 31, 'k_up': 15},
        ]

        self.u_net_bridge = [
            {'f': 2*24*13, 'k': 15},
        ]

    def build_models(self, batch_norm_momentum=0.99, verbose=True):

        def cropped_mean_squared_error(y_true, y_pred):
            y_true_c = Cropping1D(cropping=int(80000))(y_true)
            y_pred_c = Cropping1D(cropping=int(80000))(y_pred)
            # loss = self.loss_multiplier * mean_squared_error(y_true=y_true_c, y_pred=y_pred_c)
            loss = 100. * mean_squared_error(y_true=y_true_c, y_pred=y_pred_c)
            return loss

        ########################
        # U-Net: DownSample
        ########################
        x_mix = Input(shape=(self.input_length, 2), name='x_mix')
        x = x_mix
        horizontal_tensors = list()
        for i in range(len(self.u_net)):

            x = Conv1D(
                filters=self.u_net[i]['f'],
                kernel_size=self.u_net[i]['k_down'],
                strides=1,
                dilation_rate=1,
                padding='same',
                activation=None,
                use_bias=False,  # We use BatchNormalization
                name='U_Down_Conv_{}'.format(i),
                kernel_regularizer=l2(0.01),
            )(x)

            x = BatchNormalization(
                name='U_Down_BN_{}'.format(i),
                momentum=batch_norm_momentum,
                axis=-1
            )(x)

            x = LeakyReLU(
                alpha=0.2,
                name='U_Down_LeakyReLU_{}'.format(i),
            )(x)

            horizontal_tensors.append(x)

            if self.u_net[i]['p'] > 1:
                x = MaxPooling1D(
                    name='U_Down_Pool_{}'.format(i),
                    pool_size=self.u_net[i]['p'],
                    strides=self.u_net[i]['p'],
                    padding='same',
                )(x)

        ########################
        # U-Net: Bridge between down and up
        ########################
        for i in range(len(self.u_net_bridge)):

            x = Conv1D(
                filters=self.u_net_bridge[i]['f'],
                kernel_size=self.u_net_bridge[i]['k'],
                strides=1,
                dilation_rate=1,
                padding='same',
                activation=None,
                use_bias=False,  # We use BatchNormalization
                name='U_Bridge_Conv_{}'.format(i),
                kernel_regularizer=l2(0.01),
            )(x)

            x = BatchNormalization(
                name='U_Bridge_BN_{}'.format(i),
                momentum=batch_norm_momentum,
                axis=-1,
            )(x)

            x = LeakyReLU(
                alpha=0.2,
                name='U_Bridge_LeakyReLU_{}'.format(i),
            )(x)

        ########################
        # U-Net: UpSample. Note: UpSampling1D + AveragePooling1D gives linear interpolation
        ########################
        for i in reversed(range(len(self.u_net))):

            if self.u_net[i]['p'] > 1:
                x = UpSampling1D(
                    size=self.u_net[i]['p'],
                    name='U_Up_Upsample_{}'.format(i),
                )(x)
                x = AveragePooling1D(
                    pool_size=self.u_net[i]['p'],
                    strides=1,
                    padding='same',
                    name='U_Up_Average_{}'.format(i),
                )(x)

            x = Concatenate(
                name='U_Up_Concat_{}'.format(i),
            )([x, horizontal_tensors[i]])

            x = Conv1D(
                filters=self.u_net[i]['f'],
                kernel_size=self.u_net[i]['k_up'],
                strides=1,
                dilation_rate=1,
                padding='same',
                activation=None,
                use_bias=False,  # We use BatchNormalization
                name='U_Up_Conv_{}'.format(i),
                kernel_regularizer=l2(0.01),
            )(x)

            x = BatchNormalization(
                name='U_Up_BN_{}'.format(i),
                momentum=batch_norm_momentum,
                axis=-1,
            )(x)

            x = LeakyReLU(
                alpha=0.2,
                name='U_Up_LeakyReLU_{}'.format(i),
            )(x)

        ########################
        # Generate difference output
        ########################
        x = Concatenate(
            name='Concat_Unet_Mix',
        )([x, x_mix])

        stems = list()
        for stem in range(self.stems - 1):  # 'Other' is calculated as the residual...
            x_stem = Conv1D(
                filters=2,
                kernel_size=1,
                strides=1,
                padding='same',
                activation='tanh',
                name='Stem_{}'.format(stem),
            )(x)
            stems.append(x_stem)

        if self.stems > 2:
            x_other_0 = Add(name='Stem_Other_Temp')(stems)
        else:
            x_other_0 = x_stem
        x_other = Subtract(name='Stem_{}'.format(self.stems - 1))([x_mix, x_other_0])
        stems.append(x_other)
        x_sources = Concatenate(name='ConcatStems')(stems)

        ########################
        # Finally, define the model (using a cropped mean square error as loss function, to ensure proper context)
        ########################
        self.wave_net_bn_shallow_model = Model(x_mix, x_sources, name='wave_net_bn_shallow')
        self.wave_net_bn_shallow_model.compile(
            optimizer=Adam(lr=0.0001, amsgrad=True, epsilon=1e-8),
            loss=cropped_mean_squared_error
        )
        if verbose:
            self.wave_net_bn_shallow_model.summary()

    def load_data(self, nb_data_files=3):

        # Load the mini-data files
        self.data_sources_mini = np.load(os.path.join(self.data_path, 'wnet_sources_mini.npy'))
        self.data_mixes_mini = np.load(os.path.join(self.data_path, 'wnet_mix_mini.npy'))

        # Get a list of data file names
        if nb_data_files == 0 or nb_data_files is None:
            # Use a mini data file if nb_data_files set to 0 or None
            files = [{
                'mix': 'wnet_mix_mini.npy',
                'sources': 'wnet_sources_mini.npy'
            }]
        else:
            all_x_files = [fn for fn in os.listdir(self.data_path) if fn[:12] == 'wnet_sources']
            all_hashes = [h.replace('.', '_').split('_')[2] for h in all_x_files if h[-8:] != 'mini.npy']
            selected_hashes = random.sample(all_hashes, nb_data_files)
            files = [{'mix': 'wnet_mix_{}.npy'.format(h), 'sources': 'wnet_sources_{}.npy'.format(h)}
                     for h in selected_hashes]

        # Load these files
        mixes_list = list()
        sources_list = list()
        i = 0
        for data_file in files:
            i += 1
            # Load one of the data files from storage
            print('Loading file {} and {}. {} of {}'.format(data_file['mix'], data_file['sources'], i, nb_data_files))
            mixes_list.append(np.load(os.path.join(self.data_path, data_file['mix'])))
            sources_list.append(np.load(os.path.join(self.data_path, data_file['sources'])))

        print('Concatenating data...')
        self.data_sources = np.concatenate(sources_list, axis=0)
        self.data_mixes = np.concatenate(mixes_list, axis=0)
        if self.sound_length < self.data_sources.shape[1]:
            self.data_sources = self.data_sources[:, :self.sound_length, :]
            self.data_mixes = self.data_mixes[:, :self.sound_length, :]
            self.data_sources_mini = self.data_sources_mini[:, :self.sound_length, :]
            self.data_mixes_mini = self.data_mixes_mini[:, :self.sound_length, :]

        print('Data files loaded: {0}'.format(len(sources_list)))

    def load_model(self, model_file_name=''):
        model_file_path = os.path.join(self.models_path, model_file_name)
        self.wave_net_bn_shallow_model = load_model(model_file_path)
        print('Model loaded from {}'.format(model_file_path))

    def load_model_weights(self, weights_file_name='', last=True):

        if last or weights_file_name == '':
            weights_file_name = self._get_last_model()

        try:
            model_path = os.path.join(self.models_path, weights_file_name)
            self.wave_net_bn_shallow_model.load_weights(model_path, by_name=True)
            print('Weights loaded into model from {} in folder {}.'.format(
                weights_file_name, self.models_path))

        except Exception:
            print('Failed to load weights into model from {} in folder {}.'.format(
                weights_file_name, self.models_path))

    def save_model(self, save_examples=True):
        time_stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        file_name = 'wave_net_bn_shallow_model_{}.h5'.format(time_stamp)
        model_path = os.path.join(self.models_path, file_name)
        self.wave_net_bn_shallow_model.save(model_path)
        print('Saved: {}'.format(model_path))

        if save_examples:
            self.save_example_np(time_stamp=time_stamp, nb_examples=10)

    def save_example_np(self, time_stamp=None, nb_examples=10):
        # Generate examples
        print('Generating {} examples.'.format(nb_examples))
        x_mix = self.data_mixes_mini[0:nb_examples, :, :]
        x_sources = self.data_sources_mini[0:nb_examples, :, :]
        y_sources = self.wave_net_bn_shallow_model.predict(x_mix)

        # Save
        if time_stamp is None:
            time_stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        path_anfang = os.path.join(self.examples_path, 'wave_net_bn_shallow_model_{}'.format(time_stamp))

        np.save(file='{}_{}'.format(path_anfang, 'x_mix'), arr=x_mix)
        np.save(file='{}_{}'.format(path_anfang, 'x_sources'), arr=x_sources)
        np.save(file='{}_{}'.format(path_anfang, 'y_sources'), arr=y_sources)
        print('Example files saved in {}.'.format(self.examples_path))

    def train(self, epochs_between_saves=10, number_of_saves=1, batch_size=16):

        # Ensure data exists
        if self.data_sources is None or self.data_mixes is None:
            print('Data must be loaded before training. Please run load_data.')
            return None

        # Loop through the data files and train
        print('Training job for model wave_net_bn_shallow.')
        print('Launching training. {0} epochs in total will be done.'.format(number_of_saves * epochs_between_saves))

        for s in range(number_of_saves):

            # If a settings file is available, update learning rate
            try:
                with open('settings.json', 'r') as fp:
                    settings = json.load(fp)
                    if 'learning_rate' in settings.keys():
                        K.set_value(self.wave_net_bn_shallow_model.optimizer.lr, settings['learning_rate'])
                        print('Learning rate set to {} from settings.json'.format(settings['learning_rate']))
                    if 'batch_size' in settings.keys():
                        batch_size = settings['batch_size']
                        print('Batch Size set to {}.'.format(batch_size))

            except Exception:
                print('Failed to update settings from settings.json.')

            self.wave_net_bn_shallow_model.fit(
                self.data_mixes,
                self.data_sources,
                epochs=epochs_between_saves,
                batch_size=batch_size,
                shuffle=True
            )

            self.save_model()

    def _get_last_model(self):
        model_name = 'wave_net_bn_shallow'
        fn = None
        try:
            fn = sorted([fn for fn in os.listdir(self.models_path) if fn[:len(model_name)] == model_name])[-1]
        except Exception:
            print('Failed to find a model of type {}'.format(model_name))

        return fn


if __name__ == '__main__':
    print('Launching a Shallow U-Net with Batch Normalization...')
    wnet = Wnet()
    wnet.build_models()
    wnet.load_model_weights()
    wnet.load_data(nb_data_files=40)
    wnet.train(epochs_between_saves=10, number_of_saves=100000, batch_size=8)

