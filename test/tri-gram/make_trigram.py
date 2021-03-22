import os
from collections import Counter
import my_module

import my_path, edit_feature

book_dir = my_path.project_path()+'src/wakati/akuta/'

books = os.listdir(book_dir)

with open(book_dir + books[2], mode="r") as f:
    hinsis = f.read().split('\n')[1].split()

tri_gram = []

for i in range(len(hinsis) - 2):
    tri_gram.append('{}-{}-{}'.format(hinsis[i],hinsis[i+1],hinsis[i+2]))

counter = Counter(tri_gram)

morphs = len(tri_gram)
per_counter = {}
for i,v in counter.items():
    per_counter[i] = v / morphs
#print(counter)

#print(per_counter)

print(len(list(counter.keys())))
