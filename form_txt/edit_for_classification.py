import term_public_func
import edit_feature, full_name

def main():
    edit_feature.init_source()
    authors = list(full_name.fullname_dict.keys())
    opts = ['hinsi', 'bigram']
    for auth in authors:
        for opt in opts:
            edit_feature.add_auth_source(auth, opt)

#tri-gramのsrcをラベルの順に並べ替え
#labelはtrigram_orderから
def add_tri():
    edit_feature.init_source_trigram()
    authors = list(full_name.fullname_dict.keys())
    opt = 'trigram'
    for auth in authors:
        edit_feature.add_auth_source(auth, opt)

add_tri()
