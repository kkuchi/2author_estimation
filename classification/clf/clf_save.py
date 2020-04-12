import term_public_func
import clf_pair, clf_main, clf_output, my_path, full_name
import pickle

def write_res_view(source_, data_opt_, num=None):
    write_list = []

    for k,v in source_.items():
        authors = k.split('-')
        authors = [full_name.fullname_dict[i] for i in authors]
        authors_print = '著者：'+','.join(authors)
        score_print = 'score: {:.5f}'.format(v['res']['s'])
        imp_print = ['特徴量の重要度']
        count = 0
        for key, val in sorted(v['res']['i'].items(), key=lambda x: x[1], reverse=True):
            imp_print.append(key+': {:.5f}'.format(val))
            count += 1
            if count == num:
                break

        conf_print = []
        confusion_header = ['', 'p_0', 'p_1']
        out_confusion = clf_output.gen_confusion(v['con'])
        for conf in out_confusion:
            this_print = []

            this_print.append('t_1 / p_0 : {:.5f}'.format(conf[1][1] / (conf[0][1] + conf[1][1])))
            this_print.append('t_0 / p_1 : {:.5f}'.format(conf[0][2] / (conf[0][2] + conf[1][2])))
            this_print.append('--------------------------')
            conf_print.append('\n'.join(this_print))

        write_list.append(authors_print)
        write_list.append(score_print)
        write_list.append('\n'.join(imp_print))
        write_list.append('\n'.join(conf_print))

    with open(my_path.project_path()+'classification/result/res_{}_view.txt'.format(data_opt_), mode="w") as f:
        f.write('\n'.join(write_list))

def main(data_opt_):
    if data_opt_ == 'hinsi':
        opt = '品詞出現率'
    elif data_opt_ == 'bigram':
        opt = '品詞2-gram出現率'
    print('使用特徴量：'+opt)

    key_pairs = clf_pair.name_pair()
    all_result = {}
    for pair in key_pairs:
        result, confusion = clf_main.classify(data_opt_, pair, 5, True, 'forest', 'up')
        all_result['-'.join(pair)] = {'res': result, 'con': confusion}

    with open(my_path.project_path()+'classification/result/res_{}_raw.txt'.format(data_opt_), mode="wb") as f:
        pickle.dump(all_result, f)

    write_res_view(all_result, data_opt_, 5)

main('bigram')
