import os
import numpy as np
import torch.utils.data as da


def pixel_to_dBZ_2(img):
    return img * 70.0


def dBZ_to_pixel_2(dBZ_img):
    return np.clip(dBZ_img / 70.0, a_min=0.0, a_max=1.0)


class radar_dataloader(da.Dataset):
    """radar dataset"""

    def __init__(self, data_file, sample_shape=(20, 1, 140, 140), input_len=10):
        self.data_file = data_file
        self.input_seq_length = input_len

        # Load the data
        with np.load(data_file) as f:
            d = f['input_raw_data']

        # transform dBZ to pixel, [0, 1]
        d = dBZ_to_pixel_2(d)
        # Reshape and select requested number of samples
        d = d.reshape((-1,) + sample_shape)
        # d = np.transpose(d, (0, 1, 3, 4, 2))

        # The original PredRNN++ code applies this patch transform which
        # breaks the image up into patch_size patches stacked as channels.
        """
        if is_train and patch_size > 1:
            d = reshape_patch(d, patch_size)
        """

        d = d.astype(np.float32)

        # Convert to Torch tensor
        # self.data = torch.tensor(d)
        self.data = d

    def __getitem__(self, index):
        """
        return (self.data[index, :self.input_seq_length],
                self.data[index, self.input_seq_length:])
        """
        return self.data[index]

    def __len__(self):
        return self.data.shape[0]
        # return len(self.data)


def get_datasets(data_dir, n_train=None, n_valid=None, **kwargs):
    data_dir = os.path.expandvars(data_dir)
    train_data = radar_dataloader(os.path.join(data_dir, 'moving-mnist-train.npz'), **kwargs)
    valid_data = radar_dataloader(os.path.join(data_dir, 'moving-mnist-valid.npz'), **kwargs)
    return train_data, valid_data, {}
