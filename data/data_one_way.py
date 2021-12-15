import numpy as np
import torch
import lmdb
import csv
import pyarrow as pa
import torchvision.transforms as transforms
import pickle
from skimage import transform

BAND_STATS = {
    'S2': {
        'mean': {
            'B01': 340.76769064,
            'B02': 429.9430203,
            'B03': 614.21682446,
            'B04': 590.23569706,
            'B05': 950.68368468,
            'B06': 1792.46290469,
            'B07': 2075.46795189,
            'B08': 2218.94553375,
            'B8A': 2266.46036911,
            'B09': 2246.0605464,
            'B11': 1594.42694882,
            'B12': 1009.32729131
        },
        'std': {
            'B01': 554.81258967,
            'B02': 572.41639287,
            'B03': 582.87945694,
            'B04': 675.88746967,
            'B05': 729.89827633,
            'B06': 1096.01480586,
            'B07': 1273.45393088,
            'B08': 1365.45589904,
            'B8A': 1356.13789355,
            'B09': 1302.3292881,
            'B11': 1079.19066363,
            'B12': 818.86747235
        }
    },
    'S1': {
        'mean': {
            'VV': -12.619993741972035,
            'VH': -19.29044597721542,
            'VV/VH': 0.6525036195871579,
        },
        'std': {
            'VV': 5.115911777546365,
            'VH': 5.464428464912864,
            'VV/VH': 30.75264076801808,
        },
        'min': {
            'VV': -74.33214569091797,
            'VH': -75.11137390136719,
            'R': 3.21E-2
        },
        'max': {
            'VV': 34.60696029663086,
            'VH': 33.59768295288086,
            'R': 1.08
        }
    }
}

bands_mean = {
    'bands10_mean': [429.9430203, 614.21682446, 590.23569706, 2218.94553375],
    'bands20_mean': [950.68368468, 1792.46290469, 2075.46795189, 2266.46036911, 1594.42694882, 1009.32729131],
    'bands60_mean': [340.76769064, 2246.0605464],
    'vv_mean': [-12.619993741972035],
    'vh_mean': [-19.29044597721542],
    "vv/vh_mean": [0.6525036195871579]
}

bands_std = {
    'bands10_std': [572.41639287, 582.87945694, 675.88746967, 1365.45589904],
    'bands20_std': [729.89827633, 1096.01480586, 1273.45393088, 1356.13789355, 1079.19066363, 818.86747235],
    'bands60_std': [554.81258967, 1302.3292881],
    'vv_std': [5.115911777546365],
    'vh_std': [5.464428464912864],
    "vv/vh_std": [30.75264076801808]
}


class dataGenBigEarthLMDB_joint:
    """

    output is a dictionary with keys bands_S1, bands_S2 and labels
    """

    def __init__(self, bigEarthPthLMDB_S2, bigEarthPthLMDB_S1, state='train',
                 train_csv=None, val_csv=None, test_csv=None):

        self.env2 = lmdb.open(bigEarthPthLMDB_S2, readonly=True, lock=False, readahead=False, meminit=False)
        self.env1 = lmdb.open(bigEarthPthLMDB_S1, readonly=True, lock=False, readahead=False, meminit=False)

        self.modality = None
        self.train_bigEarth_csv = train_csv
        self.val_bigEarth_csv = val_csv
        self.test_bigEarth_csv = test_csv
        self.state = state
        self.patch_names = []
        self.readingCSV()

    def readingCSV(self):
        """
        this function reads the csv files, but S1 Files have different names than the names in
        the csv

        row[0] contains the Sentinel 2 image name as a string
        An adjustment happens via end and img with string concatenation
        """

        if self.state == 'train':
            with open(self.train_bigEarth_csv, 'r') as f:
                csv_reader = csv.reader(f)
                for row in csv_reader:
                    self.patch_names.append(row)

        elif self.state == "val":
            with open(self.val_bigEarth_csv, 'r') as f:
                csv_reader = csv.reader(f)
                for row in csv_reader:
                    self.patch_names.append(row)
        elif self.state == "test":
            with open(self.test_bigEarth_csv, 'r') as f:
                csv_reader = csv.reader(f)
                for row in csv_reader:
                    self.patch_names.append(row)

    def __len__(self):

        return len(self.patch_names)

    def __getitem__(self, idx):

        patch_name = self.patch_names[idx]

        return self._getDataUp(patch_name, idx)

    def _getDataUp(self, patch_name, idx):
        normalize_S2 = Normalize(bands_mean, bands_std, modality="S2")
        normalize_S1 = Normalize(bands_mean, bands_std, modality="S1")

        to_tensor_S2 = transforms.Compose([ToTensor(modality="S2")])
        to_tensor_S1 = transforms.Compose([ToTensor(modality="S1")])

        with self.env2.begin(write=False) as txn:
            byteflow_S2 = txn.get(patch_name[0].encode()) # buf


        with self.env1.begin(write=False) as txn:
            byteflow_S1 = txn.get(patch_name[1].encode())



        # Load S2 bytflow and create upsampled S2 dictionary
        bands10, bands20, _, multiHots = loads_pyarrow(byteflow_S2) #pickle.loads(byteflow_S2)

        sample_S2 = {'bands10': bands10.astype(np.float32), 'bands20': bands20.astype(np.float32),
                     'label': multiHots.astype(np.float32), 'patch_name': patch_name[0]}

        sample_S2 = normalize_S2(sample_S2)
        sample_S2 = to_tensor_S2(sample_S2)
        sample_S2['bands20'] = interpolate(sample_S2["bands20"])

        # sample['bands20'] = interp_band(bands20).astype(np.float32)
        # sample = to_tensor(sample)

        vv, vh, multiHots = loads_pyarrow(byteflow_S1) #pickle.loads(byteflow_S1)
        sample_S1 = {'vv': vv.astype(np.float32), 'vh': vh.astype(np.float32),
                     'label': multiHots.astype(np.float32),  'patch_name': patch_name[1]}

        sample_S1 = normalize_S1(sample_S1)
        sample_S1 = to_tensor_S1(sample_S1)

        sample = dict_concat(sample_S1, sample_S2)

        return sample


def interpolate(bands, img10_shape=[120, 120]):
    """
    bands: three dim tensor (6,60,60)
    return:

    bands three dim tensor (6,120,120)

    """

    bands = torch.unsqueeze(bands, 1)  # input for bicubic must be 4 dimensional

    bands_interpolated = torch.nn.functional.interpolate(bands, [120, 120], mode="bicubic")

    return torch.squeeze(bands_interpolated, 1)


def dict_concat(sample_S1: dict, sample_S2: dict,alternativ = False) -> dict:
    """

    alternativ is just used for the network with 12 bands input, ResNet Backbone and cls loss
    sample: dict with keys bands10,bands20, label

    :return:
         concat_dict: dict with keys bands, label

    """

    concat_dict = {}  # dict where bands10 and bands20 and vv and vh are concatenated along there channel dimension, e.g. (4,120,120) and (6,120,120) -> (10,120,120), label stays the same
    keys_S1 = list(sample_S1.keys())
    keys_S2 = list(sample_S2.keys())

    bands_S2 = torch.cat((sample_S2[keys_S2[0]], sample_S2[keys_S2[1]]))
    bands_S1 = torch.cat((sample_S1[keys_S1[0]], sample_S1[keys_S1[1]]))

    if alternativ:
        concat_dict["bands"] = torch.cat((bands_S2,bands_S1))
        concat_dict["labels"] = sample_S2[keys_S2[2]]
        concat_dict["patch_name_S1"] = sample_S1[keys_S1[3]]
        concat_dict["patch_name_S2"] = sample_S2[keys_S2[3]]
    else:
        concat_dict["bands_S2"] = bands_S2
        concat_dict["bands_S1"] = bands_S1
        concat_dict["labels"] = sample_S2[keys_S2[2]]
        concat_dict["patch_name_S1"] = sample_S1[keys_S1[3]]
        concat_dict["patch_name_S2"] = sample_S2[keys_S2[3]]

    return concat_dict





def loads_pyarrow(buf):
    """
    Args:
        buf: the output of `dumps`.
    """
    return pa.deserialize(buf)


class Normalize(object):
    def __init__(self, bands_mean, bands_std, modality):
        self.modality = modality

        self.bands10_mean = bands_mean['bands10_mean']
        self.bands10_std = bands_std['bands10_std']

        self.bands20_mean = bands_mean['bands20_mean']
        self.bands20_std = bands_std['bands20_std']

        self.vv_mean = bands_mean['vv_mean']
        self.vv_std = bands_std['vv_std']

        self.vh_mean = bands_mean['vh_mean']
        self.vh_std = bands_std['vh_std']

    def __call__(self, sample):

        if self.modality == "S2":
            band10, band20, label,patch_name = sample['bands10'], sample['bands20'], sample['label'],sample['patch_name']
            band10_norm = np.zeros((4, 120, 120), np.float32)
            band20_norm = np.zeros((6, 60, 60), np.float32)

            for idx, (t, m, s) in enumerate(zip(band10, self.bands10_mean, self.bands10_std)):
                band10_norm[idx] = np.divide(np.subtract(t, m), s)

            for idx, (t, m, s) in enumerate(zip(band20, self.bands20_mean, self.bands20_std)):
                band20_norm[idx] = np.divide(np.subtract(t, m), s)

            return {'bands10': band10_norm, 'bands20': band20_norm, 'label': label, 'patch_name': patch_name}

        elif self.modality == "S1":
            vv, vh, label,patch_name = sample['vv'], sample['vh'], sample['label'],sample['patch_name']
            vv_norm = np.zeros((1, 120, 120), np.float32)
            vh_norm = np.zeros((1, 120, 120), np.float32)

            for idx, (t, m, s) in enumerate(zip(vv, self.vv_mean, self.vv_std)):
                vv_norm[idx] = (t - m) / s

            for idx, (t, m, s) in enumerate(zip(vh, self.vh_mean, self.vh_std)):
                vh_norm[idx] = (t - m) / s

            return {'vv': vv_norm, 'vh': vh_norm, 'label': label, 'patch_name': patch_name}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, modality):
        self.modality = modality

    def __call__(self, sample):
        if self.modality == "S2":
            band10, band20, label,patch_name = sample['bands10'], sample['bands20'], sample['label'],sample['patch_name']

            sample = {'bands10': torch.tensor(band10), 'bands20': torch.tensor(band20), 'label': label,'patch_name':patch_name}
            return sample
        elif self.modality == "S1":

            vv, vh, label,patch_name = sample['vv'], sample['vh'], sample['label'],sample['patch_name']

            sample = {'vv': torch.tensor(vv), 'vh': torch.tensor(vh), 'label': label,'patch_name':patch_name}
            return sample












