import pickle, my_path

import trigram_order

#特徴量のデータ
source_path = my_path.project_path()+'src/feature/'
#分類用のデータ保存先
dump_path = my_path.project_path()+'classification/source/'

with open(source_path+'hinsi_per.txt', mode="rb") as f:
    source_hinsi_dict = pickle.load(f)

with open(source_path+'bigram_per.txt', mode="rb") as f:
    source_bigram_dict = pickle.load(f)

with open(source_path+'trigram_per.txt', mode="rb") as f:
    source_trigram_dict = pickle.load(f)

hinsi_label = [
    '名詞', '接尾辞', '助詞', '動詞', '特殊',
    '副詞', '形容詞', '判定詞', '未定義語', '助動詞',
    '接続詞', '指示詞', '接頭辞', '連体詞', '感動詞'
]

bigram_label = [
        '名詞-名詞', '名詞-接尾辞', '名詞-助詞', '名詞-動詞', '名詞-特殊',
        '名詞-副詞', '名詞-形容詞', '名詞-判定詞', '名詞-未定義語', '名詞-助動詞',
        '名詞-接続詞', '名詞-指示詞', '名詞-接頭辞', '名詞-連体詞', '名詞-感動詞',
        '接尾辞-名詞', '接尾辞-接尾辞', '接尾辞-助詞', '接尾辞-動詞', '接尾辞-特殊',
        '接尾辞-副詞', '接尾辞-形容詞', '接尾辞-判定詞', '接尾辞-未定義語', '接尾辞-助動詞',
        '接尾辞-接続詞', '接尾辞-指示詞', '接尾辞-接頭辞', '接尾辞-連体詞', '接尾辞-感動詞',
        '助詞-名詞', '助詞-接尾辞', '助詞-助詞', '助詞-動詞', '助詞-特殊',
        '助詞-副詞', '助詞-形容詞', '助詞-判定詞', '助詞-未定義語', '助詞-助動詞',
        '助詞-接続詞', '助詞-指示詞', '助詞-接頭辞', '助詞-連体詞', '助詞-感動詞',
        '動詞-名詞', '動詞-接尾辞', '動詞-助詞', '動詞-動詞', '動詞-特殊',
        '動詞-副詞', '動詞-形容詞', '動詞-判定詞', '動詞-未定義語', '動詞-助動詞',
        '動詞-接続詞', '動詞-指示詞', '動詞-接頭辞', '動詞-連体詞', '動詞-感動詞',
        '特殊-名詞', '特殊-接尾辞', '特殊-助詞', '特殊-動詞', '特殊-特殊',
        '特殊-副詞', '特殊-形容詞', '特殊-判定詞', '特殊-未定義語', '特殊-助動詞',
        '特殊-接続詞', '特殊-指示詞', '特殊-接頭辞', '特殊-連体詞', '特殊-感動詞',
        '副詞-名詞', '副詞-接尾辞', '副詞-助詞', '副詞-動詞', '副詞-特殊',
        '副詞-副詞', '副詞-形容詞', '副詞-判定詞', '副詞-未定義語', '副詞-助動詞',
        '副詞-接続詞', '副詞-指示詞', '副詞-接頭辞', '副詞-連体詞', '副詞-感動詞',
        '形容詞-名詞', '形容詞-接尾辞', '形容詞-助詞', '形容詞-動詞', '形容詞-特殊',
        '形容詞-副詞', '形容詞-形容詞', '形容詞-判定詞', '形容詞-未定義語', '形容詞-助動詞',
        '形容詞-接続詞', '形容詞-指示詞', '形容詞-接頭辞', '形容詞-連体詞', '形容詞-感動詞',
        '判定詞-名詞', '判定詞-接尾辞', '判定詞-助詞', '判定詞-動詞', '判定詞-特殊',
        '判定詞-副詞', '判定詞-形容詞', '判定詞-判定詞', '判定詞-未定義語', '判定詞-助動詞',
        '判定詞-接続詞', '判定詞-指示詞', '判定詞-接頭辞', '判定詞-連体詞', '判定詞-感動詞',
        '未定義語-名詞', '未定義語-接尾辞', '未定義語-助詞', '未定義語-動詞', '未定義語-特殊',
        '未定義語-副詞', '未定義語-形容詞', '未定義語-判定詞', '未定義語-未定義語', '未定義語-助動詞',
        '未定義語-接続詞', '未定義語-指示詞', '未定義語-接頭辞', '未定義語-連体詞', '未定義語-感動詞',
       '助動詞-名詞', '助動詞-接尾辞', '助動詞-助詞', '助動詞-動詞', '助動詞-特殊',
        '助動詞-副詞', '助動詞-形容詞', '助動詞-判定詞', '助動詞-未定義語', '助動詞-助動詞',
        '助動詞-接続詞', '助動詞-指示詞', '助動詞-接頭辞', '助動詞-連体詞', '助動詞-感動詞',
        '接続詞-名詞', '接続詞-接尾辞', '接続詞-助詞', '接続詞-動詞', '接続詞-特殊',
        '接続詞-副詞', '接続詞-形容詞', '接続詞-判定詞', '接続詞-未定義語', '接続詞-助動詞',
        '接続詞-接続詞', '接続詞-指示詞', '接続詞-接頭辞', '接続詞-連体詞', '接続詞-感動詞',
        '指示詞-名詞', '指示詞-接尾辞', '指示詞-助詞', '指示詞-動詞', '指示詞-特殊',
        '指示詞-副詞', '指示詞-形容詞', '指示詞-判定詞', '指示詞-未定義語', '指示詞-助動詞',
        '指示詞-接続詞', '指示詞-指示詞', '指示詞-接頭辞', '指示詞-連体詞', '指示詞-感動詞',
        '接頭辞-名詞', '接頭辞-接尾辞', '接頭辞-助詞', '接頭辞-動詞', '接頭辞-特殊',
        '接頭辞-副詞', '接頭辞-形容詞', '接頭辞-判定詞', '接頭辞-未定義語', '接頭辞-助動詞',
        '接頭辞-接続詞', '接頭辞-指示詞', '接頭辞-接頭辞', '接頭辞-連体詞', '接頭辞-感動詞',
        '連体詞-名詞', '連体詞-接尾辞', '連体詞-助詞', '連体詞-動詞', '連体詞-特殊',
        '連体詞-副詞', '連体詞-形容詞', '連体詞-判定詞', '連体詞-未定義語', '連体詞-助動詞',
        '連体詞-接続詞', '連体詞-指示詞', '連体詞-接頭辞', '連体詞-連体詞', '連体詞-感動詞',
        '感動詞-名詞', '感動詞-接尾辞', '感動詞-助詞', '感動詞-動詞', '感動詞-特殊',
        '感動詞-副詞', '感動詞-形容詞', '感動詞-判定詞', '感動詞-未定義語', '感動詞-助動詞',
        '感動詞-接続詞', '感動詞-指示詞', '感動詞-接頭辞', '感動詞-連体詞', '感動詞-感動詞',
]

#ラベルの順番に並べ替える関数
def make_feature_list(source_, label_):
    #return用
    r_list = []
    #source（作品）数だけ繰り返し
    for i in source_:
        #この作品
        t_book = []
        #ラベルの順番に繰り返し
        for label in label_:
            #ラベルがなければ0をappend
            if not label in i:
                t_book.append(0)
                continue
            #ある場合はvalueをappend
            t_book.append(i[label])
        #全ラベル終了したらr_listへ
        r_list.append(t_book)
    return r_list

def sort_feature(auth_, opt_):
    #opt_の内容で分岐
    if opt_ == 'hinsi':
        auth_booklist = make_feature_list(source_hinsi_dict[auth_], hinsi_label)
    elif opt_ == 'bigram':
        auth_booklist = make_feature_list(source_bigram_dict[auth_], bigram_label)
    elif opt_ == 'trigram':
        auth_booklist = make_feature_list(source_trigram_dict[auth_], trigram_order.label_tri)

    return auth_booklist

def add_auth_source(auth_, opt_):
    #opt_でload_pathを分岐
    if opt_ == 'hinsi':
        load_path = dump_path+'source_hinsi_per.txt'
    elif opt_ == 'bigram':
        load_path = dump_path+'source_hinsi_bigram.txt'
    elif opt_ == 'trigram':
        load_path = dump_path+'source_hinsi_trigram.txt'

    #既存のデータ読み込み
    with open(load_path, mode="rb") as f:
        res_dict = pickle.load(f)

    #著者追加
    res_dict[auth_] = sort_feature(auth_, opt_)

    #新しいres_dictを保存
    with open(load_path, mode="wb") as f:
        pickle.dump(res_dict, f)

#保存先の初期化
def init_source():
    with open(dump_path+'source_hinsi_per.txt', mode="wb") as f:
        pickle.dump({}, f)
    with open(dump_path+'source_hinsi_bigram.txt', mode="wb") as f:
        pickle.dump({}, f)

def init_source_trigram():
    with open(dump_path+'source_hinsi_trigram.txt', mode="wb") as f:
        pickle.dump({}, f)
