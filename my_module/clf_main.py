import pickle, random, math, copy
import numpy as np
#import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

import my_path

#-----------------------------------
#sourceのデータの順番
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

#-----------------------------------

#sourceのロード
'''
使用時はcopy.deepcopyで読み込み
そのまま使うと上書きされてデータ量が増える
'''

with open(my_path.project_path()+
'classification/source/source_hinsi_per.txt', mode="rb") as f:
    source_data_hinsi = pickle.load(f)

with open(my_path.project_path()+
'classification/source/source_hinsi_bigram.txt', mode="rb") as f:
    source_data_bigram = pickle.load(f)

#-----------------------------------

#指定した2著者のデータ取り出し
#入力：source_data_hinsi/bigram, [auth, auth]
#出力：data（2著者の特徴量のデータ）, label（dataの著者ラベル 前者が0, 後者が1）
def source_split(source_, authors_):
    #source_から指定した2著者のデータ取り出し
    auth_datas = []
    for auth in authors_:
        t_auth = source_[auth]
        auth_datas.append(t_auth)

    #ラベル配列の作成 2著者の作品数分の0と1
    auth_labels = [
        [0 for i in range(len(auth_datas[0]))],
        [1 for i in range(len(auth_datas[1]))]
    ]

    #extendで配列を1つに結合
    for i in ['data', 'label']:
        exec('auth_{0}s[0].extend(auth_{0}s[1])'.format(i))

    #結合後の配列をdata,labelと命名してreturn
    data = auth_datas[0]
    label = auth_labels[0]
    return data, label

#アップサンプリングの関数
def up_sampling(data_, label_):
    #SMOTEでアップサンプリング
    sm = SMOTE()
    data, label = sm.fit_sample(data_, label_)

    #変更後のデータ数を確認
    for i in [0,1]:
        print('label-{}: {}'.format(i, np.count_nonzero(label == i)))

    #fit_sampleの返り値はnp.ndarrayなので、listにしてreturn
    return data.tolist(), label.tolist()

#dataとlabelの結合
#data各要素の末尾にlabelを入れる
def connect_data_label(datas_, labels_):
    #return用
    r_list = []
    #zipで繰り返し
    for data, label in zip(datas_, labels_):
        #dataの末尾にlabelを追加、r_listへ
        data.append(label)
        r_list.append(data)
    return r_list

#k交差検証を行う関数
#入力：2著者のデータ配列, kの回数, option（分類方法。決定木/ランダムフォレスト）
def k_cross_val(source_, k_, option_='tree'):
    #source_をランダムに並べ替え
    random.shuffle(source_)
    #配列をk_個に分ける
    splitted_source = np.array_split(source_, k_)
    #option_で分類モデル変更
    if option_ == 'tree':
        clf = DecisionTreeClassifier()
    if option_ == 'forest':
        clf = RandomForestClassifier()

    #return用配列
    #スコアと重要度
    result = []
    #予測(predict)と正解(target)
    pre_tar = []

    for i,v in enumerate(splitted_source):
        #このループでの結果
        this_loop_res = {}
        this_pre_tar = {}

        #vからテストデータ作成
        test_data = [element[:-1] for element in v]
        test_label = [element[-1] for element in v]

        #訓練データ(splitted_sourceの4つ)を1つの配列にまとめる
        train_list = []
        #v以外をextendでつなげる
        for index, ele in enumerate(splitted_source):
            if index == i:
                continue
            #train_listに結合
            train_list.extend(ele)

        #train_listから訓練データ作成
        train_data = [element[:-1] for element in train_list]
        train_label = [element[-1] for element in train_list]

        #fit
        clf.fit(train_data, train_label)

        #スコアと特徴量の重要度を記録
        this_loop_res['s'] = clf.score(test_data, test_label)
        this_loop_res['i'] = clf.feature_importances_

        #predictとtargetを記録（混同行列の出力用）
        this_pre_tar['pre'] = clf.predict(test_data)
        this_pre_tar['tar'] = test_label

        #このループの結果をappend
        result.append(this_loop_res)
        pre_tar.append(this_pre_tar)
    return result, pre_tar

#score,importancesの平均値算出
#入力：result_=[{s: score, i: importances}*k(交差検証の回数)]の配列、opt_=品詞率かbigram出現率かの判定
def get_mean(result_, opt_):
    #opt_に対応したlabelを指定
    if opt_ == 'hinsi':
        label = hinsi_label
    elif opt_ == 'bigram':
        label = bigram_label

    #result_からscoreとimportancesを分けてまとめる
    scores = [i['s'] for i in result_]
    imp = [list(i['i']) for i in result_]

    #scoreの平均値
    mean_score = sum(scores) / len(scores)
    #importance平均値を入れる
    mean_imp = {}

    #impで繰り返し
    for j in imp:
        #label毎の数値を確認
        for i,v in enumerate(label):
            #mean_impにまだlabelのkeyがない場合
            if not v in mean_imp:
                #新たに作成 初期値0
                mean_imp[v] = 0
            else:
                #既にある場合は加算
                mean_imp[v] += float(j[i])

    #合計値になっているmean_impを平均値にする
    for k,v in mean_imp.items():
        #scoresの要素数(交差検証の回数)で割る
        mean_imp[k] = v/len(scores)

    #return用 sは数値, iは辞書型
    mean_res = {'s': mean_score, 'i': mean_imp}
    return mean_res

'''
入力
data_option_ = hinsi or bigram
authors_ = 著者ペア
k_ = 交差検証の回数
pre_tar_option_ = 混同行列用のデータ出力をするかどうか(True/False)
mode_option_ = 決定木(tree) or ランダムフォレスト(forest)
sample_option_ = そのまま(plain) or アップサンプリング(up)
'''
'''
出力
pre_tar_無し：{s: mean_score, i: mean_imp}
pre_tar_あり：{s: mean_score, i: mean_imp}, [{pre, tar}*k_]
'''
def classify(data_option_, authors_, k_, pre_tar_option_=False, model_option_='tree', sample_option_='plain'):
    #data_option_に応じてsource_を設定
    if data_option_ == 'hinsi':
        source_ = copy.deepcopy(source_data_hinsi)
    elif data_option_ == 'bigram':
        source_ = copy.deepcopy(source_data_bigram)
    #hinsiでもbigramでもない場合はエラー
    else:
        print('Error:clf_main.classify')
        print('data_option_に想定外のオプション"{}"が指定されています'.format(data_option_))

    #authors_で指定された2著者の取り出し
    data_content, data_label = source_split(source_, authors_)

    #sample_option_に応じてアップサンプリングを行う
    if sample_option_ == 'up':
        data_ontent, data_label = up_sampling(data_content, data_label)

    #data, labelを結合
    connected_source = connect_data_label(data_content, data_label)

    #k_に応じて交差検証
    result, pre_tar = k_cross_val(connected_source, k_, model_option_)

    #結果の平均値化
    mean_res = get_mean(result, data_option_)

    #pre_tar_option_に応じてreturnの変更
    if pre_tar_option_:
        return mean_res, pre_tar
    return mean_res
