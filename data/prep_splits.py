from __future__ import print_function
import argparse
import os
import csv
import json
from tensorflow_utils import prep_tf_record_files

GDAL_EXISTED = False
RASTERIO_EXISTED = False
UPDATE_JSON = False

with open('label_indices.json', 'rb') as f:
    label_indices = json.load(f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=
                                     'This script creates TFRecord files for the BigEarthNet train, validation and test splits')
    parser.add_argument('-r1', '--root_folder_s1', dest='root_folder_s1',
                        help='root folder path contains multiple patch folders')
    parser.add_argument('-r2', '--root_folder_s2', dest="root_folder_s2")
    parser.add_argument('-o', '--out_folder_s1', dest='out_folder_s1',
                        help='folder path containing resulting TFRecord or LMDB files')
    parser.add_argument('-o', '--out_folder_s2', dest='out_folder_s2',
                        help='folder path containing resulting TFRecord or LMDB files')
    parser.add_argument('--update_json', default=False, action="store_true", help=
    'flag for adding BigEarthNet-19 labels to the json file of each patch')
    parser.add_argument('-n', '--splits', dest='splits', help=
    'csv files each of which contain list of patch names, patches with snow, clouds, and shadows already excluded',
                        nargs='+')
    parser.add_argument('-l', '--library', type=str, dest='library', help="Limit search to Sentinel mission",
                        choices=['tensorflow'])

    args = parser.parse_args()

    # Checks the existence of patch folders and populate the list of patch folder paths
    folder_path_list = []
    if args.root_folder_s1 and args.root_folder_s2:
        if not os.path.exists(args.root_folder_s1):
            print('ERROR: folder', args.root_folder_s1, 'does not exist')
            exit()
        if not os.path.exists(args.root_folder_s2):
            print('ERROR: folder', args.root_folder_s2, 'does not exist')
            exit()
    else:
        print('ERROR: folder', args.patch_folder, 'does not exist')
        exit()

    # Checks the existence of required python packages
    try:
        import gdal

        GDAL_EXISTED = True
        print('INFO: GDAL package will be used to read GeoTIFF files')
    except ImportError:
        try:
            import rasterio

            RASTERIO_EXISTED = True
            print('INFO: rasterio package will be used to read GeoTIFF files')
        except ImportError:
            print('ERROR: please install either GDAL or rasterio package to read GeoTIFF files')
            exit()

    try:
        import numpy as np
    except ImportError:
        print('ERROR: please install numpy package')
        exit()

    if args.splits:
        try:
            patch_names_list = []
            split_names = []
            for csv_file in args.splits:
                patch_names_list.append([])
                split_names.append(os.path.basename(csv_file).split('.')[0])
                with open(csv_file, 'r') as fp:
                    csv_reader = csv.reader(fp, delimiter=',')
                    for row in csv_reader:
                        patch_names_list[-1].append(row)
        except:
            print('ERROR: some csv files either do not exist or have been corrupted')
            exit()

    if args.update_json:
        UPDATE_JSON = True

    if args.library == 'tensorflow':

        try:
            import tensorflow as tf
        except ImportError:
            print('ERROR: please install tensorflow package to create TFRecord files')
            exit()

        prep_tf_record_files(
            args.root_folder_s1,
            args.root_folder_s2,
            args.out_folder,
            split_names,
            patch_names_list,
            label_indices,
            GDAL_EXISTED,
            RASTERIO_EXISTED,
            UPDATE_JSON
        )
    elif args.library == 'pytorch':

        prep_torch_files(
            args.root_folder_s1,
            args.out_folder_s1,
            "S1",
            split_names,
            patch_names_list,
            label_indices,
            GDAL_EXISTED,
            RASTERIO_EXISTED,
            UPDATE_JSON
        )

        prep_torch_files(
            args.root_folder_s2,
            args.out_folder_s2,
            "S2",
            split_names,
            patch_names_list,
            label_indices,
            GDAL_EXISTED,
            RASTERIO_EXISTED,
            UPDATE_JSON
        )
