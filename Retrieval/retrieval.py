import argparse
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf

from tqdm import tqdm
import h5py
import ray
import numpy as np
import psutil
from sklearn.metrics import pairwise_distances
import json
import torch
from torch.utils.data import DataLoader

from utils import get_fusion, parse_config, prep_logger, get_logger, timer_calc, get_shuffle_buffer_size
from models import get_model
from data import dataGenBigEarthLMDB_joint


class Retrieval():
    def __init__(self, config):
        self.config = config
        self.logger = get_logger()
        self.summary_writer = tf.summary.create_file_writer(
            os.path.join(os.path.join(self.config['logging_params']['save_dir'], self.config['name'],
                                      self.config['logging_params']['name']),
                         self.config['logging_params']['summaries'], self.config['logging_params']['suffix']))

        self.query_feat_path = os.path.join(self.config['logging_params']['save_dir'], self.config['name'],
                                            self.config['logging_params']['name'], 'query.h5')
        self.archive_feat_path = os.path.join(self.config['logging_params']['save_dir'], self.config['name'],
                                              self.config['logging_params']['name'], 'archive.h5')
        self.retrieval_path = os.path.join(self.config['logging_params']['save_dir'], self.config['name'],
                                           self.config['logging_params']['name'], 'retrieval.h5')

        self.state_dict_path = config["state_dict"]
        self.device = torch.device('cpu')

    def prep_feature_extraction(self):

        self.model = get_model(self.config["type"], self.config["n_features"], self.config["projection_dim"],
                               self.config["out_channels"])
        self.model.to(self.device)

        query_dataGen = dataGenBigEarthLMDB_joint(
            bigEarthPthLMDB_S2=config["bigEarthPthLMDB_S2"],
            bigEarthPthLMDB_S1=config["bigEarthPthLMDB_S1"],
            state='train',
            train_csv=config["train_csv"],
            val_csv=config["val_csv"],
            test_csv=config["test_csv"]
        )
        archive_dataGen = dataGenBigEarthLMDB_joint(
            bigEarthPthLMDB_S2=config["bigEarthPthLMDB_S2"],
            bigEarthPthLMDB_S1=config["bigEarthPthLMDB_S1"],
            state='val',
            train_csv=config["train_csv"],
            val_csv=config["val_csv"],
            test_csv=config["test_csv"]
        )
        self.query_dataloader = DataLoader(query_dataGen, config["batch_size"], num_workers=0, shuffle=False,
                                           pin_memory=True)
        self.archive_dataloader = DataLoader(archive_dataGen, config["batch_size"], num_workers=0, shuffle=False,
                                             pin_memory=True)

        # features_path = os.path.join(self.config.dumps.features, self.config.suffix)
        # if not os.path.isdir(features_path):
        # os.makedirs(features_path)

    def finish_retrieval(self):
        self.summary_writer.close()
        self.logger.info('Retrieval is finished')

    def feature_extraction(self):
        self.logger.info('feature extraction is started')
        if (not os.path.isfile(self.query_feat_path)) or (not os.path.isfile(self.archive_feat_path)):
            self.prep_feature_extraction()

            ### restores weights loads model weights
            self.restore_weigths(config["state_dict"])
        with torch.no_grad():

            if not os.path.isfile(self.query_feat_path):

                with timer_calc() as elapsed_time_feat_ext:
                    query_patch_names = []
                    query_labels = []
                    query_features = []
                    with h5py.File(self.query_feat_path, 'w') as hf:
                        for query_batch_id, query_batch in enumerate(self.query_dataloader):
                            with timer_calc() as elapsed_time:

                                imgs_S1 = query_batch["bands_S1"].to(self.device)
                                imgs_S2 = query_batch["bands_S2"].to(self.device)

                                labels = query_batch['labels']

                                h_i, h_j, projection_i, projection_j = self.model(imgs_S1, imgs_S2)
                                # projection_i and _j are the outputs after the mlp heads

                                fused = get_fusion(config["fusion"], projection_i, projection_j)

                                for name_S1, name_S2, label, feature in zip(query_batch['patch_name_S1'],
                                                                            query_batch['patch_name_S2'],
                                                                            labels.numpy(),
                                                                            fused.detach().numpy()):
                                    query_patch_names.append((name_S1, name_S2))
                                    query_labels.append(label)
                                    query_features.append(feature)

                                self.logger.info('a batch of query features is extracted within {:0.2f} seconds'.format(
                                    elapsed_time()))
                        hf.create_dataset('feature', data=query_features)
                        hf.create_dataset('label', data=query_labels)
                        hf.create_dataset('patch_name', data=query_patch_names,
                                          dtype=h5py.string_dtype(encoding='utf-8'))
                    self.logger.info('feature extraction is finished for query set within {:0.2f} seconds'.format(
                        elapsed_time_feat_ext()))

            if not os.path.isfile(self.archive_feat_path):
                with timer_calc() as elapsed_time_feat_ext:
                    archive_patch_names = []
                    archive_labels = []
                    archive_features = []
                    with h5py.File(self.archive_feat_path, 'w') as hf:
                        for archive_batch_id, archive_batch in enumerate(self.archive_dataloader):
                            with timer_calc() as elapsed_time:
                                imgs_S1 = archive_batch["bands_S1"].to(self.device)
                                imgs_S2 = archive_batch["bands_S2"].to(self.device)

                                labels_a = archive_batch['labels'].to(self.device)

                                h_i, h_j, projection_i_a, projection_j_a = self.model(imgs_S1, imgs_S2)
                                # projection_i and _j are the outputs after the mlp heads

                                fused_a = get_fusion(config["fusion"], projection_i_a, projection_j_a)

                                for name_S1, name_S2, label_a, feature_a in zip(archive_batch['patch_name_S1'],
                                                                                archive_batch['patch_name_S2'],
                                                                                labels_a.numpy(),
                                                                                fused_a.detach().numpy()):
                                    archive_patch_names.append((name_S1, name_S2))
                                    archive_labels.append(label_a)
                                    archive_features.append(feature_a)
                                self.logger.info(
                                    'a batch of archive features is extracted within {:0.2f} seconds'.format(
                                        elapsed_time()))
                        hf.create_dataset('feature', data=archive_features)
                        hf.create_dataset('label', data=archive_labels)
                        hf.create_dataset('patch_name', data=archive_patch_names,
                                          dtype=h5py.string_dtype(encoding='utf-8'))
                    self.logger.info('feature extraction is finished for archive set within {:0.2f} seconds'.format(
                        elapsed_time_feat_ext()))

    def retrieval(self):
        self.logger.info('retrieval is started')

        if not os.path.isfile(self.retrieval_path):
            import numpy as np
            import psutil
            import ray
            from sklearn.metrics import pairwise_distances
            from tqdm import tqdm
            num_cpus = 20  # psutil.cpu_count()

            @ray.remote
            def calc_distance(query_feats, archive_feats):
                ## _feats created by np.array(hf['feature'])
                return pairwise_distances(query_feats, archive_feats,
                                          metric=lambda u, v: 0.5 * np.sum(((u - v) ** 2) / (u + v + 1e-10)))

            def batch_with_index(iterable, n=1):
                l = len(iterable)
                for ndx in range(0, l, n):
                    # yield returns a generator, which you can only iterate over once
                    yield [iterable[ndx:min(ndx + n, l)], np.arange(ndx, min(ndx + n, l))]

            BATCH_SIZE = 1000
            with timer_calc() as elapsed_time:

                with h5py.File(self.query_feat_path, 'r') as hf:
                    query_feats = np.array(hf['feature'])

                with h5py.File(self.archive_feat_path, 'r') as hf:
                    archive_feats = np.array(hf['feature'])
                self.logger.info('preparing data within {:0.2f} seconds'.format(elapsed_time()))

            with timer_calc() as elapsed_time:
                with h5py.File(self.retrieval_path, 'w') as hf:

                    # create_dataset creates a data set of given shape and dtype
                    distance_ds = hf.create_dataset('distance', (len(query_feats), len(archive_feats)), dtype='float32')
                    retrieval_res_ds = hf.create_dataset('retrieval_result', (len(query_feats), len(archive_feats)),
                                                         dtype='int32')
                    pbar = tqdm(total=int(np.floor(len(query_feats) / float(BATCH_SIZE))))
                    for query_batch, query_batch_idx in batch_with_index(query_feats, BATCH_SIZE):
                        ray.init(num_cpus=num_cpus, object_store_memory=30 * 1024 * 1024 * 1024)
                        result_ids = []
                        for archive_batch, archive_batch_idx in batch_with_index(archive_feats, BATCH_SIZE):
                            ## calc_distance uses ray.remote
                            result_ids.append(calc_distance.remote(query_batch, archive_batch))
                        distance_batch = ray.get(result_ids)
                        distance_batch = np.concatenate(distance_batch, axis=1)
                        distance_ds[query_batch_idx] = distance_batch
                        retrieval_res_ds[query_batch_idx] = np.argsort(distance_batch, axis=-1)
                        pbar.update(1)
                        del result_ids
                        del distance_batch
                        ray.shutdown()
                    pbar.close()
                    self.logger.info('calculating distance within {:0.2f} seconds'.format(elapsed_time()))

    def prep_metrics(self):
        self.logger.info('metric preparation is started')
        import json
        import numpy as np
        from sklearn.metrics import precision_score
        from sklearn.metrics import recall_score

        with timer_calc() as elapsed_time:
            with h5py.File(self.archive_feat_path, 'r') as archive_feat_hf:
                with h5py.File(self.query_feat_path, 'r') as query_feat_hf:
                    archive_names = list(archive_feat_hf['patch_name'])
                    archive_labels = np.array(archive_feat_hf['label'])
                    query_names = list(query_feat_hf['patch_name'])
                    query_labels = np.array(query_feat_hf['label'])

            self.logger.info('opening files within {:0.2f} seconds'.format(elapsed_time()))

        self.logger.info('calc metrics')

        def divide(a, b):
            return np.divide(a, b, out=np.zeros_like(a), where=(b != 0))

        def nb_shared_labels_fnc(x, y):
            return len(set(np.where(x)[0]).intersection(np.where(y)[0]))

        @ray.remote
        def single_query_metric(max_topk, query_multi_hot, retrieved):
            retrieved_labels = []
            for i in range(len(retrieved)):
                if i < max_topk:
                    retrieved_labels.append(retrieved[i])
                else:
                    break

            retrieved_labels = np.array(retrieved_labels)

            normalized_discounted_cumulative_gains = np.zeros(max_topk)
            average_cumulative_gains = np.zeros(max_topk)
            discounted_cumulative_gains = np.zeros(max_topk)
            max_discounted_cumulative_gains = np.zeros(max_topk)
            average_precision = np.zeros(max_topk)
            weighted_avg_precision = np.zeros(max_topk)
            precision = np.zeros(max_topk)
            recall = np.zeros(max_topk)
            nb_shared_labels = np.array(
                [nb_shared_labels_fnc(query_multi_hot, retrieved_labels[i]) for i in range(max_topk)])
            nb_shared_labels_ideal = -np.sort(-nb_shared_labels)
            is_shared_labels = (nb_shared_labels > 0).astype(np.float)
            nb_max_rel = np.sum(is_shared_labels)
            acc_is_shared_labels = np.array([np.sum(is_shared_labels[:i]) for i in range(1, max_topk + 1)])
            for topk in range(1, max_topk + 1):
                discounted_cumulative_gains[topk - 1] = (2 ** nb_shared_labels[topk - 1] - 1) / np.log2(1 + topk)
                max_discounted_cumulative_gains[topk - 1] = (2 ** nb_shared_labels_ideal[topk - 1] - 1) / np.log2(
                    1 + topk)
                normalized_discounted_cumulative_gains[topk - 1] = np.sum(discounted_cumulative_gains[:topk]) / np.sum(
                    max_discounted_cumulative_gains[:topk])
                average_cumulative_gains[topk - 1] = np.sum(nb_shared_labels[:topk]) / topk
                average_precision[topk - 1] = divide(
                    np.sum((acc_is_shared_labels[:topk] * is_shared_labels[:topk]) / range(1, topk + 1)),
                    acc_is_shared_labels[topk - 1])
                weighted_avg_precision[topk - 1] = divide(
                    np.sum((average_cumulative_gains[:topk] * is_shared_labels[:topk]) / range(1, topk + 1)),
                    acc_is_shared_labels[topk - 1])
                precision[topk - 1] = nb_shared_labels[topk - 1] / np.sum(
                    retrieved_labels[topk - 1])  # acc_is_shared_labels[topk-1] / topk
                recall[topk - 1] = nb_shared_labels[topk - 1] / np.sum(
                    query_multi_hot)  # 1.0 if acc_is_shared_labels[topk-1] > 0 else 0.0
            f1_score = divide(2 * precision * recall, precision + recall)
            return [normalized_discounted_cumulative_gains, average_cumulative_gains, average_precision,
                    weighted_avg_precision, precision, recall, f1_score]

        max_topk = 1000

        normalized_discounted_cumulative_gains = np.zeros(max_topk)
        average_cumulative_gains = np.zeros(max_topk)
        average_precision = np.zeros(max_topk)
        weighted_avg_precision = np.zeros(max_topk)
        precision = np.zeros(max_topk)
        recall = np.zeros(max_topk)
        f1_score = np.zeros(max_topk)
        with h5py.File(self.retrieval_path, 'r+') as hf:
            for key in hf.keys():
                if not key == 'distance':
                    del hf[key]
            distance = hf['distance']
            import psutil
            num_cpus = 20  # psutil.cpu_count()
            ray.init(num_cpus=num_cpus, object_store_memory=30 * 1024 * 1024 * 1024)
            result_ids = []
            process_thres = 1000
            for j in tqdm(range(len(query_names))):
                query = query_names[j]
                query_multi_hot = query_labels[j]
                ins_distance = distance[j]
                ins_sorted_distance = np.argsort(ins_distance)
                retrieved = ins_sorted_distance[range(max_topk)]
                result_ids.append(single_query_metric.remote(max_topk, query_multi_hot, archive_labels[retrieved]))
                if len(result_ids) >= process_thres:
                    with timer_calc() as elapsed_time:
                        scores = np.array(ray.get(result_ids))
                        ray.shutdown()
                        normalized_discounted_cumulative_gains += np.sum(scores[:, 0, :],
                                                                         axis=0)  # normalized_discounted_cumulative_gains
                        average_cumulative_gains += np.sum(scores[:, 1, :], axis=0)  # average_cumulative_gains
                        average_precision += np.sum(scores[:, 2, :], axis=0)  # average_precision
                        weighted_avg_precision += np.sum(scores[:, 3, :], axis=0)  # weighted_avg_precision
                        precision += np.sum(scores[:, 4, :], axis=0)  # precision
                        recall += np.sum(scores[:, 5, :], axis=0)  # recall
                        f1_score += np.sum(scores[:, 6, :], axis=0)  # f1
                        self.logger.info(
                            '{} tasks finished within {:0.2f} seconds'.format(len(result_ids), elapsed_time()))
                        result_ids = []
                        ray.init(num_cpus=num_cpus,
                                 object_store_memory=30 * 1024 * 1024 * 1024)  # , memory = 100 * 1024 * 1024 * 1024)

            if not len(result_ids) == 0:
                with timer_calc() as elapsed_time:
                    scores = np.array(ray.get(result_ids))
                    ray.shutdown()
                    normalized_discounted_cumulative_gains += np.sum(scores[:, 0, :],
                                                                     axis=0)  # normalized_discounted_cumulative_gains
                    average_cumulative_gains += np.sum(scores[:, 1, :], axis=0)  # average_cumulative_gains
                    average_precision += np.sum(scores[:, 2, :], axis=0)  # average_precision
                    weighted_avg_precision += np.sum(scores[:, 3, :], axis=0)  # weighted_avg_precision
                    precision += np.sum(scores[:, 4, :], axis=0)  # precision
                    recall += np.sum(scores[:, 5, :], axis=0)  # recall
                    f1_score += np.sum(scores[:, 6, :], axis=0)  # f1
                    self.logger.info('{} tasks finished within {:0.2f} seconds'.format(len(result_ids), elapsed_time()))
                    result_ids = []

            normalized_discounted_cumulative_gains /= float(len(query_names))
            average_cumulative_gains /= float(len(query_names))
            average_precision /= float(len(query_names))
            weighted_avg_precision /= float(len(query_names))
            precision /= float(len(query_names))
            recall /= float(len(query_names))
            f1_score /= float(len(query_names))

            hf.create_dataset('normalized_discounted_cumulative_gains', data=normalized_discounted_cumulative_gains)
            hf.create_dataset('average_cumulative_gains', data=average_cumulative_gains)
            hf.create_dataset('average_precision', data=average_precision)
            hf.create_dataset('weighted_avg_precision', data=weighted_avg_precision)
            hf.create_dataset('precision', data=precision)
            hf.create_dataset('recall', data=recall)
            hf.create_dataset('f1score', data=f1_score)
            for i in [8, 16, 32, 64, 128, max_topk]:
                print('mAP@{}(%) {}'.format(i, average_precision[i - 1] * 100))

    def restore_weigths(self, state_dict_path):

        if torch.cuda.is_available():
            self.model.load_state_dict(torch.load(state_dict_path)["state_dict"])

        else:
            self.model.load_state_dict(torch.load(state_dict_path, map_location=torch.device('cpu'))["state_dict"])
        # self.model.neural_net = tf.keras.models.load_model(os.path.join(self.config.dumps.model_weights, self.config.suffix))


if __name__ == "__main__":
    prep_logger('retrieval.log')
    logger = get_logger()

    with timer_calc() as elapsed_time:
        config = parse_config('/media/storagecube/markus/project/config/args.yaml')

        retrieval = Retrieval(config)

        retrieval.feature_extraction()
        retrieval.retrieval()
        retrieval.prep_metrics()
        retrieval.finish_retrieval()
        del retrieval
        del config
        logger.info('Args.yaml is finished within {:0.2f} seconds'.format(elapsed_time()))