import torch
import numpy as np
import pickle

def total_recall(query,feature_dict):
    """


    """

    recall_mean = 0

    label_q = query["label"]


    for idx, image in enumerate(feature_dict):
        recall_mean += recall(label_q,feature_dict[idx][3])






    return recall_mean

def total_precision(query,feature_dict):



    pass
def recall(label_q,label_archive)-> float:
    """
    :label_q: either torch.tensor or list containing query labels as multi hot
    :label_archive: either torch.tensor or list containing archive labels multi hot

    :return:  f - float value recall for one comparison of query and the i-th archive label
    """

    intersection = [x for idx, x in enumerate(label_q) if x == label_archive[idx]]

    r = len(intersection)/ len(label_q)

    return r




    pass
def precision(label_q,label_archive)-> float:
    """
        :label_q: either torch.tensor or list containing query labels as multi hot
        :label_archive: either torch.tensor or list containing archive labels multi hot

        :return:  f - float value recall for one comparison of query and the i-th archive label
        """
    intersection = [x for idx,x in enumerate(label_q) if x == label_archive[idx]]

    r = len(intersection )/ len(label_archive)

    return r
if __name__ == "__main__":

    print("hello")
    feature_dict = pickle.load(open("C:/Users/Markus/Desktop/project/Retrieval/archive_db.p", "rb"))

    print(feature_dict[0][3])

    label = [0., 1., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 1., 0., 0.,
         0.]

    print(recall(label,feature_dict[0][3].tolist()[0]))
