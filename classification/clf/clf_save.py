import my_module
import clf_pair, clf_main, clf_output, my_path, full_name, clf_tri
import pickle

#結果を要約して保存する関数
def write_res_view(source_, data_opt_, num=None):
    #ファイルにwriteするための配列
    write_list = []

    #全著者ペアで繰り返し
    for k,v in source_.items():
        #keyをsplitして著者keyを取り出し、フルネームに変換
        authors = k.split('-')
        authors = [full_name.fullname_dict[i] for i in authors]
        #著者、正答率をwrite用に用意
        authors_print = '著者：'+','.join(authors)
        score_print = 'score: {:.5f}'.format(v['res']['s'])
        #重要度write用の変数imp_print
        imp_print = ['特徴量の重要度']
        #引数numで止めるためにcountを設定
        count = 0
        #重要度降順で繰り返し
        for key, val in sorted(v['res']['i'].items(), key=lambda x: x[1], reverse=True):
            #imp_printにappend
            imp_print.append(key+': {:.5f}'.format(val))
            #countがnumと一致したら終了
            count += 1
            if count == num:
                break


        conf_print = []
        confusion_header = ['', 'p_0', 'p_1']
        #混同行列用の数値を取得
        out_confusion = clf_output.gen_confusion(v['con'])
        #全混同行列で繰り返し
        for conf in out_confusion:
            this_print = []
            #予測の誤答率を計算
            this_print.append('t_1 / p_0 : {:.5f}'.format(conf[1][1] / (conf[0][1] + conf[1][1])))
            this_print.append('t_0 / p_1 : {:.5f}'.format(conf[0][2] / (conf[0][2] + conf[1][2])))
            this_print.append('--------------------------')
            #この回の数値をconf_printにappend
            conf_print.append('\n'.join(this_print))

        #ここまで用意したデータをwrite_listにappend
        write_list.append(authors_print)
        write_list.append(score_print)
        write_list.append('\n'.join(imp_print))
        write_list.append('\n'.join(conf_print))

    #ファイル名指定して保存
    with open(my_path.project_path()+'classification/result/res_{}_view.txt'.format(data_opt_), mode="w") as f:
        f.write('\n'.join(write_list))

def main(data_opt_):
    #最初にprint
    if data_opt_ == 'hinsi':
        opt = '品詞出現率'
    elif data_opt_ == 'bigram':
        opt = '品詞2-gram出現率'
    print('使用特徴量：'+opt)

    #全著者実行するため、name_pairで著者ペアリストを作成
    key_pairs = clf_pair.name_pair()
    #結果保存の辞書型
    all_result = {}
    #key_pairs全実行
    for pair in key_pairs:
        #classify
        result, confusion = clf_main.classify(data_opt_, pair, 5, True, 'forest', 'up')
        #all_resultに[著者key1-著者key2]のkeyで結果を保存
        all_result['-'.join(pair)] = {'res': result, 'con': confusion}

    #all_resultをpickleでdump
    with open(my_path.project_path()+'classification/result/res_{}_raw.txt'.format(data_opt_), mode="wb") as f:
        pickle.dump(all_result, f)

    #結果の要旨をテキストで保存する関数
    write_res_view(all_result, data_opt_, 5)

def main_tri():
    print('使用特徴量： 品詞3-gram出現率')

    key_pairs = clf_pair.name_pair()
    all_result = {}
    for pair in key_pairs:
        result, confusion = clf_tri.classify_trigram(pair, 5, True, 'forest', 'up')
        all_result['-'.join(pair)] = {'res': result, 'con': confusion}

    with open(my_path.project_path() + 'classification/result/res_trigram_raw.txt', mode="wb") as f:
        pickle.dump(all_result, f)

    write_res_view(all_result, 'trigram', 5)

#main('bigram')

main_tri()
