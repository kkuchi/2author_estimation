import my_module
import hinsi_count, full_name

def add_auth_f(auth_):
    hinsi_count.add_auth_feature_hinsi(auth_)
    hinsi_count.add_auth_feature_bigram(auth_)

def main():
    #保存先の作成
    hinsi_count.init_feature_location()
    #著者リスト
    authors = list(full_name.fullname_dict.keys())
    #全著者で実行
    for auth in authors:
        add_auth_f(auth)

main()
