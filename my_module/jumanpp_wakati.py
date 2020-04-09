import re, os, sys, math
from pyknp.juman.juman import Juman
from collections import Counter

import word_slice, my_path


#全てのフィルに実行するための関数
def recursive_file_check(path, author):
    # ディレクトリかどうかの確認
    if os.path.isdir(path):
        # パス内のファイルを要素とするリスト
        files = os.listdir(path)
        # リストを全実行
        for file in files:
            # 隠しファイルで無ければ
            if file[0] != '.':
                # ここにもauthorが必要なので注意
                recursive_file_check(path + '/' + file, author)
    # ディレクトリで無ければ処理実行
    else:
        lis = wakati_(path)
        save_file_(lis, author)

def wakati_(path):
    text = open(path, mode='r', encoding="shift-jisx0213").read()
    text = word_slice.word_slice(text)

    file_name = re.split('/', path)[-1].replace('.txt', '')

    #形態素解析の準備
    juman = Juman()

    #テキストは1回に250文字 f_numはそれに対応する実行回数
    text_len = 250
    f_num = math.ceil(len(text)/text_len)

    #250文字毎に分けた文のリスト
    t_list = []

    #textを分けてt_listにappend 最後のループはtextの範囲を最後までにする
    for i in range(f_num):
        if i == f_num-1:
            t_list.append(text[i*text_len:])
            break
        t_list.append(text[i*text_len:(i+1)*text_len])

    #midasiとhinsiのリスト
    res_midasi_list = []
    res_hinsi_list = []

    #midasi、hinsiのappend
    for ele in t_list:
        mlist = juman.analysis(ele)
        for i in mlist.mrph_list():
            res_midasi_list.append(i.midasi)
            res_hinsi_list.append(i.hinsi)

    #element count
    c_midasi = Counter(res_midasi_list)
    c_hinsi = Counter(res_hinsi_list)

    return [file_name, res_midasi_list, res_hinsi_list]

#ファイル保存関数
def save_file_(list, author):
    name = list[0]
    list[1] = ' '.join(list[1])
    list[2] = ' '.join(list[2])
    list = '\n'.join(list[1:])
    with open(my_path.project_path()+'src/wakati/{0}/wakati_{1}.txt'.format(author, name), mode='w') as f:
        f.write(list)

#日本語でのフルネーム表記の辞書をimport
import full_name

def auth_wakati_mkdir(auth_):
    #保存先ディレクトリを作成しておく
    save_path = my_path.project_path() + 'src/wakati/{0}'.format(auth_)
    os.mkdir(save_path)
