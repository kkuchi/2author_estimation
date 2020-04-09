import term_public_func
import edit_feature, full_name

def main():
    edit_feature.init_source()
    authors = list(full_name.fullname_dict.keys())
    opts = ['hinsi', 'bigram']
    for auth in authors:
        for opt in opts:
            edit_feature.add_auth_source(auth, opt)

main()
