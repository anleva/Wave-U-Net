import numpy as np
import os
import soundfile as sf


def build_data_files(data_path_in='musdb', data_path_out='data'):

    # Get file names
    all_x_files = [fn for fn in os.listdir(data_path_in) if fn[:12] == 'wnet_sources']
    all_hashes = [h.replace('.', '_').split('_')[2] for h in all_x_files if h[-8:] != 'mini.npy']
    files = [{'mix': 'wnet_mix_{}.npy'.format(h), 'sources': 'wnet_sources_{}.npy'.format(h)} for h in all_hashes]

    i = 0
    for data_file in files:
        i += 1

        # Load one of the data file pairs
        print('Loading file {} and {}. {} of {}'.format(data_file['mix'], data_file['sources'], i, len(all_hashes)))
        mixes = np.load(os.path.join(data_path_in, data_file['mix']))
        sources = np.load(os.path.join(data_path_in, data_file['sources']))

        # Merge the non-vocal stems into one source
        vocals = sources[:, :, 0:2]
        drums = sources[:, :, 2:4]
        bass = sources[:, :, 4:6]
        other = sources[:, :, 6:8]
        accompaniment = drums + bass + other
        sources = np.dstack((vocals, accompaniment))

        # Save
        np.save(os.path.join(data_path_out, data_file['mix']), mixes)
        np.save(os.path.join(data_path_out, data_file['sources']), sources)
        print('Saved: {} and {}'.format(data_file['mix'], data_file['sources']))
        print('Mixes shape: {}'.format(mixes.shape))
        print('Sources shape: {}'.format(sources.shape))


def convert_np_to_wav(folder_path, example_name):
    """
    The Wave-U-Net neural nets produce NumPy arrays when predicting a source separation. 
    These arrays are of shape [batch_size, sound_length, nb_of_channels]
    This function loads the data in npy-file example_name in folder folder_path, and saves 
    :param folder_path: 
    :param example_name: 
    :return: 
    """

    file_path_npy = os.path.join(folder_path, example_name)

    np_array = np.load(file_path_npy)

    for b in range(np_array.shape[0]):
        for c in range(int(np_array.shape[2] / 2)):
            file_path_wav = os.path.join(folder_path, '{}_{}_{}.wav'.format(example_name[:-3], b, c))
            sf.write(
                file=file_path_wav,
                data=np_array[b, 80000:-80000, (2*c):(2*c+1)].squeeze(),
                samplerate=16000
            )

