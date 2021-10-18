import numpy as np
import os
import nltk
import re
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from itertools import chain
root=r'D:\Python_data\IMDB'
negative_files_path=os.path.join(root,'train','neg')
positive_files_path=os.path.join(root,'train','pos')
unsupervised_files_path=os.path.join(root,'train','unsup')
CV=CountVectorizer()
def load_data(dir_path):
    lst=[]
    files=os.listdir(dir_path)
    for file in files:
        with open(os.path.join(dir_path,file),encoding='utf-8') as f:
            lst.append(f.read())
    return lst
def corpus_create(stc):
    stc=re.sub('[^a-zA-Z0-9_ ]','',stc)
    stc=nltk.word_tokenize(stc.lower())
    return stc
negative_data=load_data(negative_files_path)#['hello world','this is my room']
positive_data=load_data(positive_files_path)
unsupervised_data=load_data(unsupervised_files_path)
negative_corpus=[corpus_create(stc) for stc in negative_data]#[['hello','world'],['this'].['is']]
positive_corpus=[corpus_create(stc) for stc in positive_data]

w2v=CV.fit(negative_data[:5]+positive_data[:5]) 
print(w2v.vocabulary_)


#corpus[['i','was','born'],['he','is','so','good'],]
#gensim to traing word2vec

"""
import gensim
w2v_model=gensim.models.word2vec.Word2Vec(negative_seg_data,window=10,epochs=100)
similar_word={search_term:[item[0] for item in w2v_model.wv.most_similar([search_term],topn=5)] for search_term in ['good','boring','violent']}
u=pd.DataFrame(similar_word).transpose()
print(u)
"""
