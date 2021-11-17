import numpy as np
import torch
import lmdb
import csv
import pyarrow as pa
import torchvision.transforms as transforms
from skimage.transform import resize

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
                    'bands10_mean': [ 429.9430203 ,  614.21682446,  590.23569706, 2218.94553375],
                    'bands20_mean': [ 950.68368468, 1792.46290469, 2075.46795189, 2266.46036911, 1594.42694882, 1009.32729131],
                    'bands60_mean': [ 340.76769064, 2246.0605464 ],
                    'vv_mean' : [-12.619993741972035] ,
                    'vh_mean' : [-19.29044597721542],
                    "vv/vh_mean": [0.6525036195871579]
                }

bands_std = {
                    'bands10_std': [ 572.41639287,  582.87945694,  675.88746967, 1365.45589904],
                    'bands20_std': [ 729.89827633, 1096.01480586, 1273.45393088, 1356.13789355, 1079.19066363,  818.86747235],
                    'bands60_std': [ 554.81258967, 1302.3292881 ],
                    'vv_std' : [5.115911777546365] ,
                    'vh_std' : [5.464428464912864],
                    "vv/vh_std": [ 30.75264076801808]
}



class dataGenBigEarthLMDB:

    def __init__(self,bigEarthPthLMDB ,  state='train',modality =None,
                 train_csv=None, val_csv=None, test_csv=None):

        self.env = lmdb.open(bigEarthPthLMDB, readonly=True, lock=False, readahead=False, meminit=False)


        self.modality = modality
        self.train_bigEarth_csv = train_csv
        self.val_bigEarth_csv = val_csv
        self.test_bigEarth_csv = test_csv
        self.state = state
        self.patch_names = []
        if self.modality == "S2":
            self.readingCSV()
        elif self.modality == "S1":
            self.readingCSV_S1()

    def readingCSV(self):
        if self.state == 'train':
            with open(self.train_bigEarth_csv, 'r') as f:
                csv_reader = csv.reader(f)
                for row in csv_reader:
                    self.patch_names.append(row[0])

        elif self.state == 'val':
            with open(self.val_bigEarth_csv, 'r') as f:
                csv_reader = csv.reader(f)
                for row in csv_reader:
                    self.patch_names.append(row[0])
        elif self.state == "test":
            with open(self.test_bigEarth_csv, 'r') as f:
                csv_reader = csv.reader(f)
                for row in csv_reader:
                    self.patch_names.append(row[0])

    def readingCSV_S1(self):


        if self.state == 'train':
            with open(self.train_bigEarth_csv, 'r') as f:
                csv_reader = csv.reader(f)
                for row in csv_reader:
                    end = row[0].find("_",11)  # find the third underscore, to get the right names for read_scale_raster
                    img = row[0][:end]+"_S1"+ row[0][end:]
                    self.patch_names.append(img)

        elif self.state == 'val':
            with open(self.val_bigEarth_csv, 'r') as f:
                csv_reader = csv.reader(f)
                for row in csv_reader:
                    end = row[0].find("_",11)  # find the third underscore, to get the right names for read_scale_raster
                    img = row[0][:end] + "_S1" + row[0][end:]
                    self.patch_names.append(img)
        elif self.state == "test":
            with open(self.test_bigEarth_csv, 'r') as f:
                csv_reader = csv.reader(f)
                for row in csv_reader:
                    end = row[0].find("_", 11)  # find the third underscore, to get the right names for read_scale_raster
                    img = row[0][:end] + "_S1" + row[0][end:]
                    self.patch_names.append(img)

    def __len__(self):

        return len(self.patch_names)

    def __getitem__(self, idx):

        patch_name = self.patch_names[idx]



        return self._getDataUp(patch_name, idx)


    def _getDataUp(self, patch_name, idx):
        normalize = Normalize(bands_mean,bands_std,self.modality)
        to_tensor = transforms.Compose([ToTensor(self.modality)])


        with self.env.begin(write=False) as txn:
            byteflow = txn.get(patch_name.encode())

        if self.modality == "S2":
            bands10, bands20, _, multiHots = loads_pyarrow(byteflow)

            sample = {'bands10': bands10.astype(np.float32), 'bands20': bands20.astype(np.float32),
                      'label': multiHots.astype(np.float32)}

            sample = normalize(sample)
            sample = to_tensor(sample)
            sample['bands20'] = interpolate(sample["bands20"])

            #sample['bands20'] = interp_band(bands20).astype(np.float32)
            #sample = to_tensor(sample)



        elif self.modality == "S1":
            vv,vh, multiHots = loads_pyarrow(byteflow)
            sample = {'vv': vv.astype(np.float32), 'vh': vh.astype(np.float32),
                      'label': multiHots.astype(np.float32)}

            sample = normalize(sample)
            sample = to_tensor(sample)









        return sample


def interp_band(bands, img10_shape=[120, 120]):

    #torch.nn.functional.interpolate(bands, [120, 120], mode="bicubic")
    print(bands.shape)
    bands_interp = np.zeros([bands.shape[0]] + img10_shape).astype(np.float32)

    for i in range(bands.shape[0]):
        bands_interp[i] = resize(bands[i] / 30000, img10_shape, mode='reflect') * 30000

    return bands_interp

def interpolate(bands, img10_shape=[120, 120]):
    """
    bands: three dim tensor (6,60,60)
    return:

    bands three dim tensor (6,120,120)

    """

    bands = torch.unsqueeze(bands,1)# input for bicubic must be 4 dimensional

    bands_interpolated= torch.nn.functional.interpolate(bands, [120, 120], mode="bicubic")

    return torch.squeeze(bands_interpolated,1)


def loads_pyarrow(buf):
    """
    Args:
        buf: the output of `dumps`.
    """
    return pa.deserialize(buf)


class Normalize(object):
    def __init__(self, bands_mean, bands_std,modality):
        self.modality = modality


        self.bands10_mean = bands_mean['bands10_mean']
        self.bands10_std = bands_std['bands10_std']

        self.bands20_mean = bands_mean['bands20_mean']
        self.bands20_std = bands_std['bands20_std']

        self.vv_mean= bands_mean['vv_mean']
        self.vv_std = bands_std['vv_std']

        self.vh_mean = bands_mean['vh_mean']
        self.vh_std = bands_std['vh_std']

    def __call__(self, sample):

        if self.modality == "S2":
            band10, band20, label = sample['bands10'], sample['bands20'], sample['label']

            for t, m, s in zip(band10, self.bands10_mean, self.bands10_std):
                np.divide(np.subtract(t,m),s)

            for t, m, s in zip(band20, self.bands20_mean, self.bands20_std):
                np.divide(np.subtract(t,m),s)

            return {'bands10': band10, 'bands20': band20, 'label': label}

        elif self.modality == "S1":
            vv, vh, label = sample['vv'], sample['vh'], sample['label']

            for t, m, s in zip(vv, self.vv_mean, self.vv_std):
                np.divide(np.subtract(t,m),s)

            for t, m, s in zip(vh, self.vh_mean, self.vh_std):
                np.divide(np.subtract(t,m),s)
            return {'vv': vv, 'vh': vh, 'label': label}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, modality):
        self.modality = modality

    def __call__(self, sample):
        if self.modality == "S2":
            band10, band20, label = sample['bands10'], sample['bands20'], sample['label']

            sample = {'bands10': torch.tensor(band10), 'bands20': torch.tensor(band20), 'label': label}
            return sample
        elif self.modality == "S1":

            vv,vh, label = sample['vv'], sample['vh'], sample['label']

            sample = {'vv': torch.tensor(vv), 'vh': torch.tensor(vh), 'label': label}
            return sample












