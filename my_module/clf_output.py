from tabulate import tabulate

#printする関数

#正答率、重要度をprint
def print_mean(mean_result_, opt_=None):
    #mean_result_[s]から正答率をprint
    #小数点以下3桁
    print('正答率: {:.3f}'.format(mean_result_['s']))

    #opt_がある場合のための変数
    count = 0
    print('各要素の重要度')
    #重要度の降順で繰り返し
    for k,v in sorted(mean_result_['i'].items(), key=lambda x: x[1], reverse=True):
        #重要度を小数点以下3桁でprint
        print(k,'{:.3f}'.format(v))
        #count加算
        count += 1
        #countがopt_と一致したら終了
        if opt_ == count:
            break

#混同行列を出力する関数
#入力：[{pre: 予測値, tar: テストデータのラベル}*k_]の配列
#出力：[[t_0, zz, oz], [t_1, zo, oo]]*k_の配列
def gen_confusion(pre_tar_):
    #return用配列
    out_res = []
    #pre_tar_の要素数で繰り返し
    for t_clf in pre_tar_:
        #このループの結果をまとめる変数
        p_list = []
        #pre,tarで変数を用意し集計. 00, 01, 10, 11
        zz, zo, oz, oo = 0,0,0,0
        #pre,tarをzipで繰り返し
        for p,t in zip(t_clf['pre'], t_clf['tar']):
            #ifでpreとtarの組み合わせを確認、集計していく
            if p == 0:
                if t == 0:
                    zz += 1
                elif t == 1:
                    zo += 1
            elif p == 1:
                if t == 0:
                    oz += 1
                elif t == 1:
                    oo += 1
        #p_listに結果を入れる
        p_list = [
            ['t_o', zz, oz],
            ['t_1', zo, oo]
        ]
        #全体の結果out_resにappend
        out_res.append(p_list)
    return out_res

#混同行列をprintする関数
def print_pre_tar(pre_tar_):
    #print用のヘッダー
    out_header = ['', 'p_o', 'p_1']
    #gen_confusionで行列作成
    out_res = gen_confusion(pre_tar_)

    #混同行列を1つずつprint
    for i in out_res:
        print('--------------------')
        out = tabulate(i, out_header, tablefmt='grid')
        print(out)
