import pickle

"""
with open('../src/booklist/new_booklist.txt', mode="rb") as f:
    list = pickle.load(f)

print(list['akuta'])
"""
import my_module
'''
import clf_pair

print(clf_pair.name_pair())
'''

import clf_pair, clf_main, clf_output, clf_tri

authors = ['akuta', 'natsume']

res, pre_tar = clf_tri.classify_trigram(authors, 5, True, 'forest')
clf_output.print_mean(res, 4)
clf_output.print_pre_tar(pre_tar)
