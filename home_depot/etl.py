import numpy as np
import pandas as pd
from datetime import datetime
from nltk.stem.snowball import SnowballStemmer


def str_stemmer(s):
    slist =  [stemmer.stem(w) for w in s]
    s = " ".join(slist)
    return s

def str_common_word(str1, str2):
    return sum(int(str2.find(word) >= 0) for word in str1.split())


# read data
print 'Loading data...'
datadir = '/Users/cavagnolo/ml_fun/home_depot/data/'
df_train = pd.read_csv(datadir + 'train.csv', encoding="ISO-8859-1")
df_test = pd.read_csv(datadir + 'test.csv', encoding="ISO-8859-1")
df_pro_desc = pd.read_csv(datadir + 'product_descriptions.csv')
# df_attr = pd.read_csv(datadir + 'attributes.csv')

# combine data
df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)
df_all = pd.merge(df_all, df_pro_desc, how='left', on='product_uid')

# transform search words
stemmer = SnowballStemmer('english')
print 'Transforming words...'
cols = ['search_term', 'product_title', 'product_description']
for c in cols:
    startTime = datetime.now()
    df_all[c] = df_all[c].map(lambda x: str_stemmer(x.lower().split()))
    print 'Done with ' + c + ' in ' + str(datetime.now() - startTime)
df_all['product_info'] = df_all['search_term'] + "\t" + df_all['product_title'] + "\t" + df_all['product_description']
df_all['len_of_query'] = df_all['search_term'].map(lambda x: len(x.split())).astype(np.int64)

# tokenize
print 'Tokenizing data...'
df_all['word_in_title'] = df_all['product_info'].map(lambda x: str_common_word(x.split('\t')[0], x.split('\t')[1]))
df_all['word_in_description'] = df_all['product_info'].map(lambda x: str_common_word(x.split('\t')[0], x.split('\t')[2]))

# drop unwanted data
df_all = df_all.drop(['search_term', 'product_title', 'product_description', 'product_info'], axis=1)

# save df as hdf5
df_all.to_hdf(datadir + 'all_prod_data.h5', 'df')
