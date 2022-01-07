import torch
import numpy as np
import pickle
from sklearn.metrics import f1_score, precision_score, recall_score, fbeta_score, \
    classification_report, hamming_loss, accuracy_score, coverage_error, label_ranking_loss,\
    label_ranking_average_precision_score, classification_report
from torch import nn


class Precision_score(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, predict_labels, true_labels):
        sample_prec = precision_score(true_labels, predict_labels, average='samples')
        micro_prec = precision_score(true_labels, predict_labels, average='micro')
        macro_prec = precision_score(true_labels, predict_labels, average='macro')

        return macro_prec, micro_prec, sample_prec


class Recall_score(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, predict_labels, true_labels):
        sample_rec = recall_score(true_labels, predict_labels, average='samples')
        micro_rec = recall_score(true_labels, predict_labels, average='micro')
        macro_rec = recall_score(true_labels, predict_labels, average='macro')

        return macro_rec, micro_rec, sample_rec


class F1_score(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, predict_labels, true_labels):
        macro_f1 = f1_score(true_labels, predict_labels, average="macro")
        micro_f1 = f1_score(true_labels, predict_labels, average="micro")
        sample_f1 = f1_score(true_labels, predict_labels, average="samples")

        return macro_f1, micro_f1, sample_f1


class F2_score(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, predict_labels, true_labels):
        macro_f2 = fbeta_score(true_labels, predict_labels, beta=2, average="macro")
        micro_f2 = fbeta_score(true_labels, predict_labels, beta=2, average="micro")
        sample_f2 = fbeta_score(true_labels, predict_labels, beta=2, average="samples")

        return macro_f2, micro_f2, sample_f2


class Hamming_loss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, predict_labels, true_labels):
        return hamming_loss(true_labels, predict_labels)


class Subset_accuracy(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, predict_labels, true_labels):
        return accuracy_score(true_labels, predict_labels)


class Accuracy_score(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, predict_labels, true_labels):
        # sample accuracy
        TP = (np.logical_and((predict_labels == 1), (true_labels == 1))).astype(int)
        union = (np.logical_or((predict_labels == 1), (true_labels == 1))).astype(int)
        TP_sample = TP.sum(axis=1)
        union_sample = union.sum(axis=1)

        sample_Acc = TP_sample / union_sample

        assert np.isfinite(sample_Acc).all(), 'Nan found in sample accuracy'

        FP = (np.logical_and((predict_labels == 1), (true_labels == 0))).astype(int)
        TN = (np.logical_and((predict_labels == 0), (true_labels == 0))).astype(int)
        FN = (np.logical_and((predict_labels == 0), (true_labels == 1))).astype(int)

        TP_cls = TP.sum(axis=0)
        FP_cls = FP.sum(axis=0)
        TN_cls = TN.sum(axis=0)
        FN_cls = FN.sum(axis=0)

        assert (TP_cls + FP_cls + TN_cls + FN_cls == predict_labels.shape[0]).all(), 'wrong'

        macro_Acc = np.mean((TP_cls + TN_cls) / (TP_cls + FP_cls + TN_cls + FN_cls))

        micro_Acc = (TP_cls.mean() + TN_cls.mean()) / (TP_cls.mean() + FP_cls.mean() + TN_cls.mean() + FN_cls.mean())

        return macro_Acc, micro_Acc, sample_Acc.mean()


class One_error(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, predict_probs, true_labels):
        row_inds = np.arange(predict_probs.shape[0])
        col_inds = np.argmax(predict_probs, axis=1)
        return np.mean((true_labels[tuple(row_inds), tuple(col_inds)] == 0).astype(int))


class Coverage_error(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, predict_probs, true_labels):
        return coverage_error(true_labels, predict_probs)


class Ranking_loss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, predict_probs, true_labels):
        return label_ranking_loss(true_labels, predict_probs)


class LabelAvgPrec_score(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, predict_probs, true_labels):
        return label_ranking_average_precision_score(true_labels, predict_probs)


class calssification_report(nn.Module):

    def __init__(self, target_names):
        super().__init__()
        self.target_names = target_names

    def forward(self, predict_labels, true_labels):
        report = classification_report(true_labels, predict_labels, target_names=self.target_names, output_dict=True)

        return report


def get_metrics(query,retrieved_labels,N=100):
    Precision = "Total_Precison@{}"

    Recall = "Total_Recall@{}"

    metrics_dict = {}

    for n in range(N):
        metrics_dict[Precision.format(n+1)] = total_precision(query,retrieved_labels[:n+1])
        metrics_dict[Recall.format(n+1)] = total_recall(query, retrieved_labels[:n+1])

    return metrics_dict



def total_recall(query,retrieved_labels):
    """
    :query: torch.tensor containing labels with shape[19]
    :feature_dict: dict - contains for each key a tuple of (torch.tensor: projection_head_s1,torch.tensor: projection_head_s2, fusion, label of archive_image)

    """



    recall_mean = 0

    label_q = query.tolist()


    for idx in range(len(retrieved_labels)):

        label_a = torch.squeeze(retrieved_labels[idx]).tolist()
        recall_mean += recall(label_q,label_a)

    recall_mean = recall_mean / len(retrieved_labels)


    return recall_mean

def total_precision(query,retrieved_labels):
    """
       :query: torch.tensor containing labels with shape[19]
       :retrieved_labels: tuple - labels sorted from max to min sim_scores are returned as tuple of torch.tensors[1,19]

    """

    precision_mean = 0

    label_q = query.tolist()

    for idx in range(len(retrieved_labels)):
        label_a = torch.squeeze(retrieved_labels[idx]).tolist()
        precision_mean += precision(label_q, label_a)

    precision_mean = precision_mean / len(retrieved_labels)

    return precision_mean

def recall(label_q,label_archive)-> float:
    """
    :label_q: either torch.tensor or list containing query labels as multi hot
    :label_archive: either torch.tensor or list containing archive labels multi hot

    :return:  f - float value recall for one comparison of query and the i-th archive label
    """

    intersection = [x for idx, x in enumerate(label_q) if x == label_archive[idx]]

    if label_q.count(1) != 0:
        r = intersection.count(1) / label_q.count(1)
    else:
        r = 0

    return r




    pass
def precision(label_q,label_archive)-> float:
    """
        :label_q: either torch.tensor or list containing query labels as multi hot
        :label_archive: either torch.tensor or list containing archive labels multi hot

        :return:  f - float value recall for one comparison of query and the i-th archive label
        """
    intersection = [x for idx,x in enumerate(label_q) if x == label_archive[idx]]

    if label_archive.count(1) != 0:
        r = intersection.count(1)/ label_archive.count(1)
    else:
        r = 0

    return r
if __name__ == "__main__":


    retrieved_labels = []

    for x in range(100):
        t = torch.rand(1,19)
        thres = torch.Tensor([0.5])  # threshold
        out = (t > thres).float() * 1
        retrieved_labels.append(out)



    label = torch.tensor([0., 1., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 1., 0., 0.,
         0.])

    print(total_recall(label,retrieved_labels))
    print(total_precision(label, retrieved_labels))

    print(get_metrics(label,retrieved_labels))
