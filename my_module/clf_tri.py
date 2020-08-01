import pickle, math, random, copy
import numpy as np

from imblearn.over_sampling import SMOTE
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

import clf_pair, my_path, trigram_order, clf_main

tri_label = copy.deepcopy(trigram_order.label_tri)

with open(my_path.project_path()+
'classification/source/source_hinsi_trigram.txt', mode="rb") as f:
    source_data_trigram = pickle.load(f)

def get_mean_tri(result_):
    scores = [i['s'] for i in result_]
    imp = [list(i['i']) for i in result_]

    mean_score = sum(scores) / len(scores)
    mean_imp = {}

    for j in imp:
        for i,v in enumerate(tri_label):
            if not v in mean_imp:
                mean_imp[v] = 0
            else:
                mean_imp[v] += float(j[i])

    for k,v in mean_imp.items():
        mean_imp[k] = v/len(scores)

    mean_res = {'s': mean_score, 'i': mean_imp}
    return mean_res

def classify_trigram(authors_, k_, pre_tar_option_=False, model_option_="tree", sample_option_="plain"):
    source_ = copy.deepcopy(source_data_trigram)

    data_content, data_label = clf_main.source_split(source_, authors_)
    if sample_option_ == 'up':
        data_content, data_label = clf_main.up_sampling(data_content, data_label)

    connected_source = clf_main.connect_data_label(data_content, data_label)

    result, pre_tar = clf_main.k_cross_val(connected_source, k_, model_option_)

    #要修正
    mean_res = get_mean_tri(result)

    if pre_tar_option_:
        return mean_res, pre_tar
    return mean_res
