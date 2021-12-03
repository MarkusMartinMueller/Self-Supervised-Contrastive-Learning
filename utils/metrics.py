import torch
import numpy as np
import pickle

def total_recall(query,feature_dict):
    """
    :query: torch.tensor containing labels with shape[19]
    :feature_dict: dict - contains for each key a tuple of (torch.tensor: projection_head_s1,torch.tensor: projection_head_s2, fusion, label of archive_image)

    """



    recall_mean = 0

    label_q = query.tolist()


    for idx in range(len(feature_dict)):

        label_a = torch.squeeze(feature_dict[idx][3]).tolist() # dim = 3 contains labels with shape [1,19]
        recall_mean += recall(label_q,label_a)

    recall_mean = recall_mean / len(feature_dict)


    return recall_mean

def total_precision(query,feature_dict):
    """
       :query: torch.tensor containing labels with shape[19]
       :feature_dict: dict - contains for each key a tuple of (torch.tensor: projection_head_s1,torch.tensor: projection_head_s2, fusion, label of archive_image)

    """

    precision_mean = 0

    label_q = query.tolist()

    for idx in range(len(feature_dict)):
        label_a = torch.squeeze(feature_dict[idx][3]).tolist()  # dim = 3 contains labels with shape [1,19]
        precision_mean += precision(label_q, label_a)

    precision_mean = precision_mean / len(feature_dict)

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


    feature_dict = pickle.load(open("C:/Users/Markus/Desktop/project/Retrieval/archive_separate_avg.p", "rb"))

    print(feature_dict[0][3].shape)

    label = torch.tensor([0., 1., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 1., 0., 0.,
         0.])

    print(total_recall(label,feature_dict))
    print(total_precision(label, feature_dict))
