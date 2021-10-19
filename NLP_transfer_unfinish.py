import numpy as np
import os
import nltk
import re
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import joblib 
from tensorflow.keras.preprocessing.sequence import pad_sequences
root=r'D:\Python_data\IMDB'
negative_files_path=os.path.join(root,'train','neg')
positive_files_path=os.path.join(root,'train','pos')
unsupervised_files_path=os.path.join(root,'train','unsup')
CV=CountVectorizer(lowercase=True)
def load_data(dir_path):
    lst=[]
    files=os.listdir(dir_path)
    for file in files:
        with open(os.path.join(dir_path,file),encoding='utf-8') as f:
            lst.append(f.read())
    return lst
negative_data=load_data(negative_files_path)#['hello world','this is my room']
positive_data=load_data(positive_files_path)
unsupervised_data=load_data(unsupervised_files_path)
if not os.path.exists('cv_model.pkl'):
    print('start training')
    cv=CV.fit(negative_data+positive_data)
    joblib.dump(cv,'cv_model.pkl')
else:
    cv=joblib.load('cv_model.pkl')
    
def data_preprocess(raw_data,cv=cv,SENTENCE_LENGTH=20):
    """
    raw_data:list #['hello world','this is my room']
    """
    lst=[]
    for stc in raw_data:
        stc=re.sub(r'[^a-zA-Z0-9_ ]','',stc.lower())
        word_list=nltk.word_tokenize(stc) #['this', 'is', 'my', 'room']
        word_id_seq=[cv.vocabulary_[item]  for item in word_list if cv.vocabulary_.get(item,0)]#[1,3,4,0,0,0]
        lst.append(word_id_seq)
    lst=pad_sequences(lst,maxlen=SENTENCE_LENGTH,padding='post',truncating='post')#keep same length
    return lst
#gensim to traing word2vec

"""
import gensim
w2v_model=gensim.models.word2vec.Word2Vec(negative_seg_data,window=10,epochs=100)
similar_word={search_term:[item[0] for item in w2v_model.wv.most_similar([search_term],topn=5)] for search_term in ['good','boring','violent']}
u=pd.DataFrame(similar_word).transpose()
print(u)
"""
negative_seq=data_preprocess(negative_data)
