import numpy as np
import torch
import lmdb
import csv
import pyarrow as pa

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


class dataGenBigEarthLMDB:

    def __init__(self, root_folder_s1,root_folder_s2, imgTransform=None, state='train',
                 train_csv=None, val_csv=None, test_csv=None):

        #self.env = lmdb.open(bigEarthPthLMDB, readonly=True, lock=False, readahead=False, meminit=False)
        self.root_folder_s1 = root_folder_s1
        self.root_folder_s2 = root_folder_s2
        self.imgTransform = imgTransform
        self.train_bigEarth_csv = train_csv
        self.val_bigEarth_csv = val_csv
        self.test_bigEarth_csv = test_csv
        self.state = state
        self.patch_names = []
        self.readingCSV()

    def readingCSV(self):
        if self.state == 'train':
            with open(self.train_bigEarth_csv, 'r') as f:
                csv_reader = csv.reader(f)
                for row in csv_reader:
                    self.patch_names.append(row)

        elif self.state == 'val':
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





        sample = {'bands10': bands10.astype(np.float32), 'bands20': bands20.astype(np.float32),
                  'label': multiHots.astype(np.float32)}




        return sample















