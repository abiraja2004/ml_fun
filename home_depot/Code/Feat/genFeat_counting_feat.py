
"""
__file__

    genFeat_counting_feat.py

__description__

    This file generates the following features for each run and fold, and for the entire training and testing set.

        1. Basic Counting Features
            
            1. Count of n-gram in query/title/description

            2. Count & Ratio of Digit in query/title/description

            3. Count & Ratio of Unique n-gram in query/title/description

        2. Intersect Counting Features

            1. Count & Ratio of a's n-gram in b's n-gram

        3. Intersect Position Features

            1. Statistics of Positions of a's n-gram in b's n-gram

            2. Statistics of Normalized Positions of a's n-gram in b's n-gram

__author__

    Chenglong Chen < c.chenglong@gmail.com >

"""

import re
import sys
import ngram
import cPickle
import numpy as np
from nlp_utils import stopwords, english_stemmer, stem_tokens
from feat_utils import try_divide, dump_feat_name
sys.path.append("../")
from param_config import config



def get_position_list(target, obs):
    """
        Get the list of positions of obs in target
    """
    pos_of_obs_in_target = [0]
    if len(obs) != 0:
        pos_of_obs_in_target = [j for j,w in enumerate(obs, start=1) if w in target]
        if len(pos_of_obs_in_target) == 0:
            pos_of_obs_in_target = [0]
    return pos_of_obs_in_target


######################
## Pre-process data ##
######################
token_pattern = r"(?u)\b\w\w+\b"
def preprocess_data(line,
                    token_pattern=token_pattern,
                    exclude_stopword=config.cooccurrence_word_exclude_stopword,
                    encode_digit=False):
    token_pattern = re.compile(token_pattern, flags = re.UNICODE | re.LOCALE)
    tokens = [x.lower() for x in token_pattern.findall(line)]
    tokens_stemmed = stem_tokens(tokens, english_stemmer)
    if exclude_stopword:
        tokens_stemmed = [x for x in tokens_stemmed if x not in stopwords]
    return tokens_stemmed


def extract_feat(df):

    names = ["query", "product_title", "product_description"]

    ## unigram
    for n in names:
        print("Generate unigram: %s" % n)
        df["%s_unigram"%(n)] = list(df.apply(lambda x: preprocess_data(x[n]), axis=1))

    ## bigram
    for n in names:
        print("Generate bigram: %s" % n)
        join_str = "_"
        df["%s_bigram"%(n)] = list(df.apply(lambda x: ngram.getBigram(x["%s_unigram"%(n)], join_str), axis=1))

    ## trigram
    for n in names:
        print("Generate trigram: %s" % n)
        join_str = "_"
        df["%s_trigram"%(n)] = list(df.apply(lambda x: ngram.getTrigram(x["%s_unigram"%(n)], join_str), axis=1))


    ################################
    ## word count and digit count ##
    ################################
    print "generate word counting features"
    feat_names = ["query", "product_title", "product_description"]
    grams = ["unigram", "bigram", "trigram"]
    count_digit = lambda x: sum([1. for w in x if w.isdigit()])
    for feat_name in feat_names:
        for gram in grams:
            ## word count
            df["count_of_%s_%s"%(feat_name,gram)] = list(df.apply(lambda x: len(x[feat_name+"_"+gram]), axis=1))
            df["count_of_unique_%s_%s"%(feat_name,gram)] = list(df.apply(lambda x: len(set(x[feat_name+"_"+gram])), axis=1))
            df["ratio_of_unique_%s_%s"%(feat_name,gram)] = map(try_divide, df["count_of_unique_%s_%s"%(feat_name,gram)], df["count_of_%s_%s"%(feat_name,gram)])

        ## digit count
        df["count_of_digit_in_%s"%feat_name] = list(df.apply(lambda x: count_digit(x[feat_name+"_unigram"]), axis=1))
        df["ratio_of_digit_in_%s"%feat_name] = map(try_divide, df["count_of_digit_in_%s"%feat_name], df["count_of_%s_unigram"%(feat_name)])

    ## description missing indicator
    df["description_missing"] = list(df.apply(lambda x: int(x["product_description_unigram"] == ""), axis=1))


    ##############################
    ## intersect word count ##
    ##############################
    print "generate intersect word counting features"
    #### unigram
    for gram in grams:
        for obs_name in feat_names:
            for target_name in feat_names:
                if target_name != obs_name:
                    ## query
                    df["count_of_%s_%s_in_%s"%(obs_name,gram,target_name)] = list(df.apply(lambda x: sum([1. for w in x[obs_name+"_"+gram] if w in set(x[target_name+"_"+gram])]), axis=1))
                    df["ratio_of_%s_%s_in_%s"%(obs_name,gram,target_name)] = map(try_divide, df["count_of_%s_%s_in_%s"%(obs_name,gram,target_name)], df["count_of_%s_%s"%(obs_name,gram)])

        ## some other feat
        df["title_%s_in_query_div_query_%s"%(gram,gram)] = map(try_divide, df["count_of_product_title_%s_in_query"%gram], df["count_of_query_%s"%gram])
        df["title_%s_in_query_div_query_%s_in_product_title"%(gram,gram)] = map(try_divide, df["count_of_product_title_%s_in_query"%gram], df["count_of_query_%s_in_product_title"%gram])
        df["description_%s_in_query_div_query_%s"%(gram,gram)] = map(try_divide, df["count_of_product_description_%s_in_query"%gram], df["count_of_query_%s"%gram])
        df["description_%s_in_query_div_query_%s_in_product_description"%(gram,gram)] = map(try_divide, df["count_of_product_description_%s_in_query"%gram], df["count_of_query_%s_in_product_description"%gram])


    ######################################
    ## intersect word position feat ##
    ######################################
    print "generate intersect word position features"
    for gram in grams:
        for target_name in feat_names:
            for obs_name in feat_names:
                if target_name != obs_name:
                    pos = list(df.apply(lambda x: get_position_list(x[target_name+"_"+gram], obs=x[obs_name+"_"+gram]), axis=1))
                    ## stats feat on pos
                    df["pos_of_%s_%s_in_%s_min" % (obs_name, gram, target_name)] = map(np.min, pos)
                    df["pos_of_%s_%s_in_%s_mean" % (obs_name, gram, target_name)] = map(np.mean, pos)
                    df["pos_of_%s_%s_in_%s_median" % (obs_name, gram, target_name)] = map(np.median, pos)
                    df["pos_of_%s_%s_in_%s_max" % (obs_name, gram, target_name)] = map(np.max, pos)
                    df["pos_of_%s_%s_in_%s_std" % (obs_name, gram, target_name)] = map(np.std, pos)
                    ## stats feat on normalized_pos
                    df["normalized_pos_of_%s_%s_in_%s_min" % (obs_name, gram, target_name)] = map(try_divide, df["pos_of_%s_%s_in_%s_min" % (obs_name, gram, target_name)], df["count_of_%s_%s" % (obs_name, gram)])
                    df["normalized_pos_of_%s_%s_in_%s_mean" % (obs_name, gram, target_name)] = map(try_divide, df["pos_of_%s_%s_in_%s_mean" % (obs_name, gram, target_name)], df["count_of_%s_%s" % (obs_name, gram)])
                    df["normalized_pos_of_%s_%s_in_%s_median" % (obs_name, gram, target_name)] = map(try_divide, df["pos_of_%s_%s_in_%s_median" % (obs_name, gram, target_name)], df["count_of_%s_%s" % (obs_name, gram)])
                    df["normalized_pos_of_%s_%s_in_%s_max" % (obs_name, gram, target_name)] = map(try_divide, df["pos_of_%s_%s_in_%s_max" % (obs_name, gram, target_name)], df["count_of_%s_%s" % (obs_name, gram)])
                    df["normalized_pos_of_%s_%s_in_%s_std" % (obs_name, gram, target_name)] = map(try_divide, df["pos_of_%s_%s_in_%s_std" % (obs_name, gram, target_name)] , df["count_of_%s_%s" % (obs_name, gram)])


if __name__ == "__main__":

    ###############
    ## Load Data ##
    ###############
    ## load data
    with open(config.processed_train_data_path, "rb") as f:
        dfTrain = cPickle.load(f)
    with open(config.processed_test_data_path, "rb") as f:
        dfTest = cPickle.load(f)
    ## load pre-defined stratified k-fold index
    with open("%s/stratifiedKFold.%s.pkl" % (config.data_folder, config.stratified_label), "rb") as f:
            skf = cPickle.load(f)

    ## file to save feat names
    feat_name_file = "%s/counting.feat_name" % config.feat_folder


    #######################
    ## Generate Features ##
    #######################
    print("==================================================")
    print("Generate counting features...")

    print("==================================================")
    print("Train")
    extract_feat(dfTrain)
    feat_names = [
        name for name in dfTrain.columns \
            if "count" in name \
            or "ratio" in name \
            or "div" in name \
            or "pos_of" in name
    ]
    feat_names.append("description_missing")
    print("Done.")

    print("==================================================")
    print("K-folds in cross-validation")
    for run in range(config.n_runs):
        ## use 33% for training and 67 % for validation
        ## so we switch trainInd and validInd
        for fold, (validInd, trainInd) in enumerate(skf[run]):
            print("Run: %d, Fold: %d" % (run+1, fold+1))
            path = "%s/Run%d/Fold%d" % (config.feat_folder, run+1, fold+1)
              
            #########################
            ## get word count feat ##
            #########################
            for feat_name in feat_names:
                X_train = dfTrain[feat_name].values[trainInd]
                X_valid = dfTrain[feat_name].values[validInd]
                with open("%s/train.%s.feat.pkl" % (path, feat_name), "wb") as f:
                    cPickle.dump(X_train, f, -1)
                with open("%s/valid.%s.feat.pkl" % (path, feat_name), "wb") as f:
                    cPickle.dump(X_valid, f, -1)
    print("Done.")

    print("==================================================")
    print("Test")
    path = "%s/All" % config.feat_folder
    extract_feat(dfTest)
    for feat_name in feat_names:
        X_train = dfTrain[feat_name].values
        X_test = dfTest[feat_name].values
        with open("%s/train.%s.feat.pkl" % (path, feat_name), "wb") as f:
            cPickle.dump(X_train, f, -1)
        with open("%s/test.%s.feat.pkl" % (path, feat_name), "wb") as f:
            cPickle.dump(X_test, f, -1)
    print("Done.")
            
    ## save feat names
    print("Feature names are stored in %s" % feat_name_file)
    dump_feat_name(feat_names, feat_name_file)
    print("All Done.")
