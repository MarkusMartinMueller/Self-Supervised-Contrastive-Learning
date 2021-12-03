import torch
import numpy as np
import pickle



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
