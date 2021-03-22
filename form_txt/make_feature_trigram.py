import my_module
import trigram_feature, full_name

def main():
    trigram_feature.init_feature_location()
    names = list(full_name.fullname_dict.keys())
    for auth in names:
        trigram_feature.add_auth_feature_trigram(auth)

main()
