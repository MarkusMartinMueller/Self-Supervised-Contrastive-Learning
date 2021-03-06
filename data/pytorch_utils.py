import json
import csv
import os
import numpy as np

from collections import defaultdict

# original labels
LABELS = [
    'Continuous urban fabric',
    'Discontinuous urban fabric',
    'Industrial or commercial units',
    'Road and rail networks and associated land',
    'Port areas',
    'Airports',
    'Mineral extraction sites',
    'Dump sites',
    'Construction sites',
    'Green urban areas',
    'Sport and leisure facilities',
    'Non-irrigated arable land',
    'Permanently irrigated land',
    'Rice fields',
    'Vineyards',
    'Fruit trees and berry plantations',
    'Olive groves',
    'Pastures',
    'Annual crops associated with permanent crops',
    'Complex cultivation patterns',
    'Land principally occupied by agriculture, with significant areas of natural vegetation',
    'Agro-forestry areas',
    'Broad-leaved forest',
    'Coniferous forest',
    'Mixed forest',
    'Natural grassland',
    'Moors and heathland',
    'Sclerophyllous vegetation',
    'Transitional woodland/shrub',
    'Beaches, dunes, sands',
    'Bare rock',
    'Sparsely vegetated areas',
    'Burnt areas',
    'Inland marshes',
    'Peatbogs',
    'Salt marshes',
    'Salines',
    'Intertidal flats',
    'Water courses',
    'Water bodies',
    'Coastal lagoons',
    'Estuaries',
    'Sea and ocean'
]
# the new labels
NEW_LABELS = [
    'Urban fabric',
    'Industrial or commercial units',
    'Arable land',
    'Permanent crops',
    'Pastures',
    'Complex cultivation patterns',
    'Land principally occupied by agriculture, with significant areas of natural vegetation',
    'Agro-forestry areas',
    'Broad-leaved forest',
    'Coniferous forest',
    'Mixed forest',
    'Natural grassland and sparsely vegetated areas',
    'Moors, heathland and sclerophyllous vegetation',
    'Transitional woodland/shrub',
    'Beaches, dunes, sands',
    'Inland wetlands',
    'Coastal wetlands',
    'Inland waters',
    'Marine waters'
]
# removed labels from the original 43 labels
REMOVED_LABELS = [
    'Road and rail networks and associated land',
    'Port areas',
    'Airports',
    'Mineral extraction sites',
    'Dump sites',
    'Construction sites',
    'Green urban areas',
    'Sport and leisure facilities',
    'Bare rock',
    'Burnt areas',
    'Intertidal flats'
]
# merged labels
GROUP_LABELS = {
    'Continuous urban fabric':'Urban fabric',
    'Discontinuous urban fabric':'Urban fabric',
    'Non-irrigated arable land':'Arable land',
    'Permanently irrigated land':'Arable land',
    'Rice fields':'Arable land',
    'Vineyards':'Permanent crops',
    'Fruit trees and berry plantations':'Permanent crops',
    'Olive groves':'Permanent crops',
    'Annual crops associated with permanent crops':'Permanent crops',
    'Natural grassland':'Natural grassland and sparsely vegetated areas',
    'Sparsely vegetated areas':'Natural grassland and sparsely vegetated areas',
    'Moors and heathland':'Moors, heathland and sclerophyllous vegetation',
    'Sclerophyllous vegetation':'Moors, heathland and sclerophyllous vegetation',
    'Inland marshes':'Inland wetlands',
    'Peatbogs':'Inland wetlands',
    'Salt marshes':'Coastal wetlands',
    'Salines':'Coastal wetlands',
    'Water bodies':'Inland waters',
    'Water courses':'Inland waters',
    'Coastal lagoons':'Marine waters',
    'Estuaries':'Marine waters',
    'Sea and ocean':'Marine waters'
}

def parse_json_labels(f_j_path):
    """
    parse meta-data json file for big earth to get image labels
    :param f_j_path: json file path
    :return:
    """
    with open(f_j_path, 'r') as f_j:
        j_f_c = json.load(f_j)
    return j_f_c['labels']


def dumps_pickle(obj):
    """
    Serialize an object.
    Returns:
        Implementation-dependent bytes-like object
    """
    import pickle

    return pickle.dumps(obj)

def cls2multiHot_old(cls_vec):
    """
    create old multi hot label
    """
    tmp = np.zeros((len(LABELS),))
    for cls_nm in cls_vec:
        tmp[LABELS.index(cls_nm)] = 1

    return tmp

def cls2multiHot_new(cls_vec):
    """
    create new multi hot label
    """
    tmp = np.zeros((len(NEW_LABELS),))
    for cls_nm in cls_vec:
        if cls_nm in GROUP_LABELS:
            tmp[NEW_LABELS.index(GROUP_LABELS[cls_nm])] = 1
        elif cls_nm not in set(NEW_LABELS):
            continue
        else:
            tmp[NEW_LABELS.index(cls_nm)] = 1
    return tmp

def multiHot2cls(multiHotCode):
    """
    multi hot labe to list of label_set
    """
    pos = np.where(np.squeeze(multiHotCode))[0].tolist()
    return np.array(LABELS)[pos,]


def read_scale_raster(file_path, GDAL_EXISTED, RASTERIO_EXISTED):
    """
    read raster file with specified scale
    :param file_path:
    :return:
    """
    if GDAL_EXISTED:
        import gdal
    elif RASTERIO_EXISTED:
        import rasterio

    if GDAL_EXISTED:
        band_ds = gdal.Open(file_path, gdal.GA_ReadOnly)
        raster_band = band_ds.GetRasterBand(1)
        band_data = raster_band.ReadAsArray()

    elif RASTERIO_EXISTED:
        band_ds = rasterio.open(file_path)
        band_data = np.array(band_ds.read(1))

    return band_data

class dataGenBigEarthTiff:

    """

    db - later change BigEarth_Serbia_Summer_S2.lmdb in lmdb.open(os.path.join(out_folder, 'BigEarth_Serbia_Summer_S2.lmdb'), map_size=map_size_)
    to the full BigEarthNet_S2.lmdb
    """

    def __init__(self, bigEarthDir=None,
                 labels_folder=None,
                 bands10=None, bands20=None, bands60=None, vv= None, vh = None,
                 patch_names_list=None,
                 RASTERIO_EXISTED=None, GDAL_EXISTED=None
                 ):


        # patches_names_list  is a list which containes lists of s2 and s1 strings, e.g. S2B_MSIL2A_20171107T105229_83_2 and corresponding sentinel 1
        self.Dir = bigEarthDir
        self.labels_folder = labels_folder
        self.bands10 = bands10
        self.bands20 = bands20
        self.bands60 = bands60
        self.vv = vv
        self.vh = vh

        self.GDAL_EXISTED = GDAL_EXISTED
        self.RASTERIO_EXISTED = RASTERIO_EXISTED

        # patches_names_list is a list with train,val and test splits
        self.total_patch = patch_names_list[0] + patch_names_list[1] + patch_names_list[2]



    def __len__(self):

        return len(self.total_patch)

    def __getitem__(self, index):

        return self.__data_generation(index)

    def __data_generation(self, idx):



        imgNm = self.total_patch[idx]




        bands10_array = []
        bands20_array = []
        bands60_array = []
        vv_array = []
        vh_array = []

        if self.bands10 is not None:
            for band in self.bands10:
                bands10_array.append(
                    read_scale_raster(os.path.join(self.Dir, imgNm[0], imgNm[0] + '_B' + band + '.tif'),
                                      self.GDAL_EXISTED, self.RASTERIO_EXISTED))

        if self.bands20 is not None:
            for band in self.bands20:
                bands20_array.append(
                    read_scale_raster(os.path.join(self.Dir, imgNm[0],imgNm[0] + '_B' + band + '.tif'),
                                      self.GDAL_EXISTED, self.RASTERIO_EXISTED))

        if self.bands60 is not None:
            for band in self.bands60:
                bands60_array.append(
                    read_scale_raster(os.path.join(self.Dir, imgNm[0], imgNm[0] + '_B' + band + '.tif'),
                                      self.GDAL_EXISTED, self.RASTERIO_EXISTED))



        if self.vv is not None:
            vv_array.append(
                read_scale_raster(os.path.join(self.Dir, imgNm[1], imgNm[1] + '_VV' + '.tif'),
                                  self.GDAL_EXISTED, self.RASTERIO_EXISTED))
        if self.vh is not None:
            vh_array.append(
                read_scale_raster(os.path.join(self.Dir, imgNm[1], imgNm[1] + '_VH' + '.tif'),
                                  self.GDAL_EXISTED, self.RASTERIO_EXISTED))

        bands10_array = np.asarray(bands10_array).astype(np.float32)
        bands20_array = np.asarray(bands20_array).astype(np.float32)
        bands60_array = np.asarray(bands60_array).astype(np.float32)

        vv_array = np.asarray(vv_array).astype(np.float32)
        vh_array = np.asarray(vh_array).astype(np.float32)

        labels = parse_json_labels(os.path.join(self.labels_folder, imgNm[0], imgNm[0] + '_labels_metadata.json'))
        oldMultiHots = cls2multiHot_old(labels)
        oldMultiHots.astype(int)
        newMultiHots = cls2multiHot_new(labels)
        newMultiHots.astype(int)

        if self.bands10 is not None and self.bands20 is not None and self.bands60 is not None :
            sample = {'bands10': bands10_array, 'bands20': bands20_array, 'bands60': bands60_array,
                      'patch_name': imgNm[0], 'multi_hots_n': newMultiHots, 'multi_hots_o': oldMultiHots}
        elif self.vv is not None and self.vh is not None:

            sample = {'vv': vv_array, 'vh': vh_array,
                      'patch_name': imgNm[1], 'multi_hots_n': newMultiHots, 'multi_hots_o': oldMultiHots}

        return sample


def prep_lmdb_files(root_folder,labels_folder, out_folder,modality, patch_names_list, GDAL_EXISTED, RASTERIO_EXISTED):
    from torch.utils.data import DataLoader
    import lmdb

    if modality == "S2":
        dataGen = dataGenBigEarthTiff(
            bigEarthDir=root_folder,
            labels_folder= labels_folder,
            bands10=['02', '03', '04', '08'],
            bands20=['05', '06', '07', '8A', '11', '12'],
            bands60=['01', '09'],
            vv=None,
            vh=None,
            patch_names_list=patch_names_list,
            GDAL_EXISTED=GDAL_EXISTED,
            RASTERIO_EXISTED=RASTERIO_EXISTED
        )

        nSamples = len(dataGen)
        map_size_ = (dataGen[0]['bands10'].nbytes + dataGen[0]['bands20'].nbytes + dataGen[0]['bands60'].nbytes) * 10 * len(
            dataGen)
        data_loader = DataLoader(dataGen, num_workers=4, collate_fn=lambda x: x)

        db = lmdb.open(os.path.join(out_folder, 'BigEarth_Serbia_Summer_S2.lmdb'), map_size=map_size_)

        txn = db.begin(write=True)
        patch_names = []
        for idx, data in enumerate(data_loader):
            bands10, bands20, bands60, patch_name, multiHots_n, multiHots_o = data[0]['bands10'], data[0]['bands20'], data[0][
                'bands60'], data[0]['patch_name'], data[0]['multi_hots_n'], data[0]['multi_hots_o']
            # txn.put(u'{}'.format(patch_name).encode('ascii'), dumps_pyarrow((bands10, bands20, bands60, multiHots_n, multiHots_o)))
            txn.put(u'{}'.format(patch_name).encode('ascii'), dumps_pickle((bands10, bands20, bands60, multiHots_n)))
            patch_names.append(patch_name)

            if idx % 10000 == 0:
                print("[%d/%d]" % (idx, nSamples))
                txn.commit()
                txn = db.begin(write=True)


    elif modality == "S1":
        dataGen = dataGenBigEarthTiff(
            bigEarthDir=root_folder,
            labels_folder=labels_folder,
            bands10=None,
            bands20=None,
            bands60=None,
            vv=["VV"],
            vh=["VH"],
            patch_names_list=patch_names_list,
            GDAL_EXISTED=GDAL_EXISTED,
            RASTERIO_EXISTED=RASTERIO_EXISTED
        )

        nSamples = len(dataGen)
        map_size_ = (dataGen[0]['vv'].nbytes + dataGen[0]['vh'].nbytes ) * 10 * len(
            dataGen)
        data_loader = DataLoader(dataGen, num_workers=4, collate_fn=lambda x: x)

        db = lmdb.open(os.path.join(out_folder, 'BigEarth_Serbia_Summer_S1.lmdb'), map_size=map_size_)

        txn = db.begin(write=True)
        patch_names = []
        for idx, data in enumerate(data_loader):
            vv,vh, patch_name, multiHots_n, multiHots_o = data[0]['vv'], data[0]['vh'], data[0]['patch_name'], data[0]['multi_hots_n'], data[0]['multi_hots_o']
            # txn.put(u'{}'.format(patch_name).encode('ascii'), dumps_pyarrow((bands10, bands20, bands60, multiHots_n, multiHots_o)))
            txn.put(u'{}'.format(patch_name).encode('ascii'), dumps_pickle((vv,vh, multiHots_n)))
            patch_names.append(patch_name)

            if idx % 10000 == 0:
                print("[%d/%d]" % (idx, nSamples))
                txn.commit()
                txn = db.begin(write=True)



    txn.commit()
    keys = [u'{}'.format(patch_name).encode('ascii') for patch_name in patch_names]

    with db.begin(write=True) as txn:
        txn.put(b'__keys__', dumps_pickle(keys))
        txn.put(b'__len__', dumps_pickle(len(keys)))

    print("Flushing database ...")
    db.sync()
    db.close()