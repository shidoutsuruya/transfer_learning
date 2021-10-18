
"""
import gensim
w2v_model=gensim.models.word2vec.Word2Vec(negative_seg_data,window=10,epochs=100)
similar_word={search_term:[item[0] for item in w2v_model.wv.most_similar([search_term],topn=5)] for search_term in ['good','boring','violent']}
u=pd.DataFrame(similar_word).transpose()
print(u)
"""
