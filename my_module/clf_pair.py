import full_name

#名前keyのリスト
names = list(full_name.fullname_dict.keys())

#名前keyのペアリストを作成
def name_pair():
    #return用のリスト
    pairs = []

    #繰り返し
    for key1, val1 in enumerate(names):
        #key1より後のindexをkey2で指定する
        for key2 in range(1, len(names) - key1):
            #ここでnamesのindexを指定するため、key2は1から始める
            pairs.append([val1, names[key1 + key2]])
    return pairs
