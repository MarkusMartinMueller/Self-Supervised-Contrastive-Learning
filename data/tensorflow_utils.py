#import tensorflow as tf
import numpy as np
import os
import json

# SAR band names to read related GeoTIFF files
band_names_s1 = ["VV", "VH"]

# Spectral band names to read related GeoTIFF files
band_names_s2 = ['B01', 'B02', 'B03', 'B04', 'B05',
                 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12']


def prep_example(bands, BigEarthNet_19_labels, BigEarthNet_19_labels_multi_hot, patch_name_s1, patch_name_s2):
    return tf.train.Example(
        features=tf.train.Features(
            feature={
                'B01': tf.train.Feature(
                    int64_list=tf.train.Int64List(value=np.ravel(bands['B01']))),
                'B02': tf.train.Feature(
                    int64_list=tf.train.Int64List(value=np.ravel(bands['B02']))),
                'B03': tf.train.Feature(
                    int64_list=tf.train.Int64List(value=np.ravel(bands['B03']))),
                'B04': tf.train.Feature(
                    int64_list=tf.train.Int64List(value=np.ravel(bands['B04']))),
                'B05': tf.train.Feature(
                    int64_list=tf.train.Int64List(value=np.ravel(bands['B05']))),
                'B06': tf.train.Feature(
                    int64_list=tf.train.Int64List(value=np.ravel(bands['B06']))),
                'B07': tf.train.Feature(
                    int64_list=tf.train.Int64List(value=np.ravel(bands['B07']))),
                'B08': tf.train.Feature(
                    int64_list=tf.train.Int64List(value=np.ravel(bands['B08']))),
                'B8A': tf.train.Feature(
                    int64_list=tf.train.Int64List(value=np.ravel(bands['B8A']))),
                'B09': tf.train.Feature(
                    int64_list=tf.train.Int64List(value=np.ravel(bands['B09']))),
                'B11': tf.train.Feature(
                    int64_list=tf.train.Int64List(value=np.ravel(bands['B11']))),
                'B12': tf.train.Feature(
                    int64_list=tf.train.Int64List(value=np.ravel(bands['B12']))),
                "VV": tf.train.Feature(
                    float_list=tf.train.FloatList(value=np.ravel(bands['VV']))),
                "VH": tf.train.Feature(
                    float_list=tf.train.FloatList(value=np.ravel(bands['VH']))),
                'BigEarthNet-19_labels': tf.train.Feature(
                    bytes_list=tf.train.BytesList(
                        value=[i.encode('utf-8') for i in BigEarthNet_19_labels])),
                'BigEarthNet-19_labels_multi_hot': tf.train.Feature(
                    int64_list=tf.train.Int64List(value=BigEarthNet_19_labels_multi_hot)),
                'patch_name_s1': tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[patch_name_s1.encode('utf-8')])),
                'patch_name_s2': tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[patch_name_s2.encode('utf-8')]))
            }))


def create_split(root_folder_s1, root_folder_s2, patch_names, TFRecord_writer, label_indices, GDAL_EXISTED,
                 RASTERIO_EXISTED, UPDATE_JSON):
    label_conversion = label_indices['label_conversion']
    BigEarthNet_19_label_idx = {v: k for k, v in label_indices['BigEarthNet-19_labels'].items()}
    if GDAL_EXISTED:
        import gdal
    elif RASTERIO_EXISTED:
        import rasterio
    progress_bar = tf.contrib.keras.utils.Progbar(target=len(patch_names))
    for patch_idx, patch_name in enumerate(patch_names):
        patch_name_s1, patch_name_s2 = patch_name[1], patch_name[0]
        patch_folder_path_s1 = os.path.join(root_folder_s1, patch_name_s1)
        patch_folder_path_s2 = os.path.join(root_folder_s2, patch_name_s2)

        bands = {}
        for band_name in band_names_s1:
            band_path = os.path.join(
                patch_folder_path_s1, patch_name_s1 + '_' + band_name + '.tif')
            if GDAL_EXISTED:
                band_ds = gdal.Open(band_path, gdal.GA_ReadOnly)
                raster_band = band_ds.GetRasterBand(1)
                band_data = raster_band.ReadAsArray()
                bands[band_name] = np.array(band_data)
            elif RASTERIO_EXISTED:
                band_ds = rasterio.open(band_path)
                band_data = np.array(band_ds.read(1))
                bands[band_name] = np.array(band_data)

        for band_name in band_names_s2:
            # First finds related GeoTIFF path and reads values as an array
            band_path = os.path.join(
                patch_folder_path_s2, patch_name_s2 + '_' + band_name + '.tif')
            if GDAL_EXISTED:
                band_ds = gdal.Open(band_path, gdal.GA_ReadOnly)
                raster_band = band_ds.GetRasterBand(1)
                band_data = raster_band.ReadAsArray()
                bands[band_name] = np.array(band_data)
            elif RASTERIO_EXISTED:
                band_ds = rasterio.open(band_path)
                band_data = np.array(band_ds.read(1))
                bands[band_name] = np.array(band_data)

        original_labels_multi_hot = np.zeros(
            len(label_indices['original_labels'].keys()), dtype=int)
        BigEarthNet_19_labels_multi_hot = np.zeros(len(label_conversion), dtype=int)
        patch_json_path = os.path.join(
            patch_folder_path_s1, patch_name + '_labels_metadata.json')

        with open(patch_json_path, 'rb') as f:
            patch_json = json.load(f)

        original_labels = patch_json['labels']
        for label in original_labels:
            original_labels_multi_hot[label_indices['original_labels'][label]] = 1

        for i in range(len(label_conversion)):
            BigEarthNet_19_labels_multi_hot[i] = (
                    np.sum(original_labels_multi_hot[label_conversion[i]]) > 0
            ).astype(int)

        BigEarthNet_19_labels = []
        for i in np.where(BigEarthNet_19_labels_multi_hot == 1)[0]:
            BigEarthNet_19_labels.append(BigEarthNet_19_label_idx[i])

        if UPDATE_JSON:
            patch_json['BigEarthNet_19_labels'] = BigEarthNet_19_labels
            with open(patch_json_path, 'wb') as f:
                json.dump(patch_json, f)

        example = prep_example(
            bands,
            BigEarthNet_19_labels,
            BigEarthNet_19_labels_multi_hot,
            patch_name_s1,
            patch_name_s2
        )
        TFRecord_writer.write(example.SerializeToString())
        progress_bar.update(patch_idx)


def prep_tf_record_files(root_folder_s1, root_folder_s2, out_folder, split_names, patch_names_list, label_indices,
                         GDAL_EXISTED, RASTERIO_EXISTED, UPDATE_JSON):
    try:
        writer_list = []
        for split_name in split_names:
            writer_list.append(
                tf.python_io.TFRecordWriter(os.path.join(
                    out_folder, split_name + '.tfrecord'))
            )
    except:
        print('ERROR: TFRecord writer is not able to write files')
        exit()

    for split_idx in range(len(patch_names_list)):
        print('INFO: creating the split of', split_names[split_idx], 'is started')
        create_split(
            root_folder_s1,
            root_folder_s2,
            patch_names_list[split_idx],
            writer_list[split_idx],
            label_indices,
            GDAL_EXISTED,
            RASTERIO_EXISTED,
            UPDATE_JSON
        )
        writer_list[split_idx].close()

