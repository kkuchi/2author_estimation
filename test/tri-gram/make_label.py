import term_public_func

import edit_feature
from copy import deepcopy

label = deepcopy(edit_feature.hinsi_label)

label_tri = []

for i in label:
    for j in label:
        for k in label:
            label_tri.append('{}-{}-{}'.format(i,j,k))


with open('./tri_label.txt', mode="w") as f:
    f.write('label_tri = \n{}'.format(label_tri))
