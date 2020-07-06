import pickle
from collections import Counter
import my_path

src_dir = my_path.project_path()+'src/wakati/'

trigram_save_path = my_path.project_path()+'src/feature/trigram_per.txt'

with open(my_path.project_path()+'src/booklist/new_booklist.txt', mode="rb") as f:
    new_booklist = pickle.load(f)

def init_feature_location():
    with open(trigram_save_path, mode="wb") as f:
        pickle.dump({}, f)

#品詞n-gramのcounterを作成
def make_hinsi_trigram(wakati_path_, booklist_):
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
        hinsi_trigram = []
        """
        #index,valueを取り出して繰り返し
        for i,v in enumerate(hinsi):
            #vとその次の要素でbigram作成
            hinsi_bigram.append('{0}-{1}'.format(v, hinsi[i+1]))
            #iが後ろから2番目のインデックスになったら終わり
            if i == len(hinsi) - 2:
                break
        """
        for i in range(len(hinsi)-2):
            hinsi_trigram.append('{}-{}-{}'.format(hinsi[i], hinsi[i+1], hinsi[i+2]))

        trigram_counter = Counter(hinsi_trigram)
        r_list.append(trigram_counter)
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

def add_auth_feature_trigram(auth_):
    wakati_path = my_path.project_path()+'src/wakati/{}/'.format(auth_)
    booklist = new_booklist[auth_]

    tri_num = make_hinsi_trigram(wakati_path, booklist)
    tri_per = num_to_per(tri_num)

    save_path = trigram_save_path
    with open(save_path, mode="rb") as f:
        res = pickle.load(f)

    res[auth_] = tri_per

    with open(save_path, mode="wb") as f:
        pickle.dump(res, f)
