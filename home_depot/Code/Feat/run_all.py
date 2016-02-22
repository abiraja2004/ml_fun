import os

# preprocess data
# os.system("python preprocess.py")

# generate kfold
#os.system("python gen_kfold.py")

# query id
#os.system("python genFeat_id_feat.py")

# counting feat
#os.system("python genFeat_counting_feat.py")

# distance feat
os.system("python genFeat_distance_feat.py")

# # basic tfidf
os.system("python genFeat_basic_tfidf_feat.py")

# # cooccurrence tfidf
os.system("python genFeat_cooccurrence_tfidf_feat.py")

# combine feat
# os.system("python combine_feat_[LSA_and_stats_feat_Jun09]_[Low].py")

# combine feat
# os.system("python combine_feat_[LSA_svd150_and_Jaccard_coef_Jun14]_[Low].py")

# combine feat
# os.system("python combine_feat_[svd100_and_bow_Jun23]_[Low].py")

# combine feat
# os.system("python combine_feat_[svd100_and_bow_Jun27]_[High].py")
