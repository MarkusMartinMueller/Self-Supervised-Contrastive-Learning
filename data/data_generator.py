import numpy as np
import torch
import lmdb
import csv
import torchvision.transforms as transforms
import pickle
from skimage import transform



BANDS_MEAN = {
    'bands10_mean': [429.9430203, 614.21682446, 590.23569706, 2218.94553375],
    'bands20_mean': [950.68368468, 1792.46290469, 2075.46795189, 2266.46036911, 1594.42694882, 1009.32729131],
    'bands60_mean': [340.76769064, 2246.0605464],
    'vv_mean': [-12.619993741972035],
    'vh_mean': [-19.29044597721542],
    "vv/vh_mean": [0.6525036195871579]
}

BANDS_STD = {
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
        this function reads the csv file
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

        return self._getDataUp(patch_name)

    def _getDataUp(self, patch_name):
        normalize_S2 = Normalize(BANDS_MEAN, BANDS_STD, modality="S2")
        normalize_S1 = Normalize(BANDS_MEAN, BANDS_STD, modality="S1")

        to_tensor_S2 = transforms.Compose([ToTensor(modality="S2")])
        to_tensor_S1 = transforms.Compose([ToTensor(modality="S1")])

        with self.env2.begin(write=False) as txn:
            byteflow_S2 = txn.get(patch_name[0].encode()) # buf


        with self.env1.begin(write=False) as txn:
            byteflow_S1 = txn.get(patch_name[1].encode())



        # Load S2 bytflow and create upsampled S2 dictionary
        bands10, bands20, _, multiHots = pickle.loads(byteflow_S2)

        sample_S2 = {'bands10': bands10.astype(np.float32), 'bands20': bands20.astype(np.float32),
                     'label': multiHots.astype(np.float32), 'patch_name': patch_name[0]}

        sample_S2 = normalize_S2(sample_S2)
        sample_S2 = to_tensor_S2(sample_S2)
        sample_S2['bands20'] = interpolate(sample_S2["bands20"])

        # sample['bands20'] = interp_band(bands20).astype(np.float32)
        # sample = to_tensor(sample)

        vv, vh, multiHots = pickle.loads(byteflow_S1)
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

    bands = torch.unsqueeze(bands, 1)  # input for bicubic must be 4 dimensional, e.g. from (6,60,60) to (6,1,60,60)

    bands_interpolated = torch.nn.functional.interpolate(bands, [120, 120], mode="bicubic",align_corners =False)

    return torch.squeeze(bands_interpolated, 1)


def dict_concat(sample_S1: dict, sample_S2: dict,alternativ = False) -> dict:
    """

    alternativ is just used for the network with 12 bands input, ResNet Backbone and cls loss
    sample: dict with keys bands10,bands20, label

    :return:
         concat_dict: dict with keys bands, label

    """

    concat_dict = {}  # dict where bands10 and bands20 and vv and vh are concatenated along there channel dimension, e.g. (4,120,120) and (6,120,120) -> (10,120,120), label stays the same


    bands_S2 = torch.cat((sample_S2["bands10"], sample_S2["bands20"]))
    bands_S1 = torch.cat((sample_S1["vv"], sample_S1["vh"]))

    if alternativ:
        concat_dict["bands"] = torch.cat((bands_S2,bands_S1))
        concat_dict["labels"] = sample_S2["label"]
        concat_dict["patch_name_S1"] = sample_S1["patch_name"]
        concat_dict["patch_name_S2"] = sample_S2["patch_name"]
    else:
        concat_dict["bands_S2"] = bands_S2
        concat_dict["bands_S1"] = bands_S1
        concat_dict["labels"] = sample_S2["label"]
        concat_dict["patch_name_S1"] = sample_S1["patch_name"]
        concat_dict["patch_name_S2"] = sample_S2["patch_name"]

    return concat_dict








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
                vv_norm[idx] = np.divide(np.subtract(t, m), s)

            for idx, (t, m, s) in enumerate(zip(vh, self.vh_mean, self.vh_std)):
                vh_norm[idx] = np.divide(np.subtract(t, m), s)

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












