import pickle

"""
with open('../src/booklist/new_booklist.txt', mode="rb") as f:
    list = pickle.load(f)

print(list['akuta'])
"""
import term_public_func
'''
import clf_pair

print(clf_pair.name_pair())
'''

import clf_pair, clf_main, clf_output

authors = ['akuta', 'natsume']

res, pre_tar = clf_main.classify('hinsi', authors, 5, True, 'forest', 'up')
clf_output.print_mean(res, 4)
clf_output.print_pre_tar(pre_tar)
