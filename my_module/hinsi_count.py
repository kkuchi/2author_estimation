import pickle, sys, os
from collections import Counter
import full_name, my_path

with open(my_path.project_path()+'src/booklist/new_booklist.txt', mode="rb") as f:
    new_booklist = pickle.load(f)

#品詞数のcounterを作成
def make_hinsi_num(wakati_path_, booklist_):
    r_list = []
    for book in booklist_:
        with open(wakati_path_+'wakati_'+book+'.txt', mode="r") as f:
            hinsi = f.read().split('\n')[1].split()
        h_counter = Counter(hinsi)
        r_list.append(h_counter)
    return r_list

#品詞n-gramのcounterを作成
def make_hinsi_bigram(wakati_path_, booklist_):
    #return用
    r_list = []
    #新字新仮名作品リスト booklist_で繰り返し
    for book in booklist_:
        #この作品のpath
        bookpath = 'wakati_{}.txt'.format(book)
        #openして品詞リスト取り出し
        with open(wakati_path_+bookpath, mode="r") as f:
            hinsi = f.read().split('\n')[1].split()
        #bigramのリスト
        hinsi_bigram = []
        #index,valueを取り出して繰り返し
        for i,v in enumerate(hinsi):
            #vとその次の要素でbigram作成
            hinsi_bigram.append('{0}-{1}'.format(v, hinsi[i+1]))
            #iが後ろから2番目のインデックスになったら終わり
            if i == len(hinsi) - 2:
                break
        bigram_counter = Counter(hinsi_bigram)
        r_list.append(bigram_counter)
    return r_list

#数のCounterから割合のdictに変換
def num_to_per(hinsi_num_):
    r_list = []
    for ele in hinsi_num_:
        t_list = {}
        h_sum = sum(list(ele.values()))
        for k,v in ele.items():
            t_list[k] = v / h_sum
        r_list.append(t_list)
    return r_list

def init_feature_location():
    feature_path = my_path.project_path()+'src/feature/'
    os.mkdir(feature_path)
    with open(feature_path+'hinsi_per.txt', mode="wb") as f:
        pickle.dump({},f)
    with open(feature_path+'bigram_per.txt', mode="wb") as f:
        pickle.dump({},f)

#品詞率の特徴量を作成、追加
def add_auth_feature_hinsi(auth_):
    #分かち書きの場所
    wakati_path = my_path.project_path()+'src/wakati/{}/'.format(auth_)
    #著者の新字新仮名作品リスト
    booklist = new_booklist[auth_]

    #品詞毎の数のCounter
    fnum_auth = make_hinsi_num(wakati_path, booklist)
    #fnum_authを品詞率に変換
    fper_auth = num_to_per(fnum_auth)

    #保存先ファイルの元データを取り出し
    save_path = my_path.project_path()+'src/feature/hinsi_per.txt'
    with open(save_path, mode="rb") as f:
        res = pickle.load(f)

    #追加するデータを辞書型に追記
    res[auth_] = fper_auth
    #保存
    with open(save_path, mode="wb") as f:
        pickle.dump(res, f)

def add_auth_feature_bigram(auth_):
    wakati_path = my_path.project_path()+'src/wakati/{}/'.format(auth_)
    booklist = new_booklist[auth_]

    bi_num = make_hinsi_bigram(wakati_path, booklist)
    bi_per = num_to_per(bi_num)

    save_path = my_path.project_path()+'src/feature/bigram_per.txt'
    with open(save_path, mode="rb") as f:
        res = pickle.load(f)

    res[auth_] = bi_per

    with open(save_path, mode="wb") as f:
        pickle.dump(res, f)
