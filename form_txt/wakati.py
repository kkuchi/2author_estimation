import os
import term_public_func
import full_name, my_path, word_slice, jumanpp_wakati

def wakati_dir(path, author):
    if os.path.isdir(path):
        files = os.listdir(path)

        for file in files:
            if file[0] != '.':
                wakati_dir(path + '/' + file, author)
    else:
        lis = jumanpp_wakati.wakati_(path)
        jumanpp_wakati.save_file_(lis, author)

def main():
    books = os.listdir(my_path.project_path() + 'src/original_txt/')
    authors = list(full_name.fullname_dict.keys())
    for auth in authors:
        #保存先dir作成
        jumanpp_wakati.auth_wakati_mkdir(auth)
        #分かち書き実行
        original_path = my_path.project_path() + 'src/original_txt/{}_txt/'.format(auth)
        jumanpp_wakati.wakati_dir(original_path, auth)

def test():
    authors = list(full_name.fullname_dict.keys())
    print(authors)
    jumanpp_wakati.auth_wakati_mkdir('tanaka')
    wakati_dir(my_path.project_path()+'src/original_txt/akuta_txt/「仮面」の人々.txt', 'tanaka')

main()
