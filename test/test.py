import pickle

with open('../src/booklist/new_booklist.txt', mode="rb") as f:
    list = pickle.load(f)

print(list['akuta'])
