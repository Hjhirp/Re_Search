#import required stuff
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
nltk.download('wordnet')

import re #for working with regular expression
import nltk #for natural language processing (nlp)
import string #This is a module, Python also has built-in class str, these are different

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from gensim.models.word2vec import Word2Vec
from gensim.models import KeyedVectors


df_cleaned = pd.read_csv('df_final.csv')

df_cleaned.drop(['Issue', 'Art. No.', 'Page start', 'Page end', 
               'Page count', 'Cited by', 'Document Type', 
               'Source'],axis=1, inplace=True)

#Let us see what do we have. 
print(df_cleaned.head(3))
#Note that he have a text column, which we will use n this demo

#Info on the training set
df_cleaned.info()

df_cleaned['Abstract']=df_cleaned['Abstract'].str.lower()
df_cleaned['Source title']=df_cleaned['Source title'].str.lower()
df_cleaned['Author Keywords']=df_cleaned['Author Keywords'].str.lower()
df_cleaned['Index Keywords']=df_cleaned['Index Keywords'].str.lower()

punctuation=string.punctuation
#print(type(punctuation), punctuation)

mapping=str.maketrans("","",punctuation)
#print(type(mapping), mapping)

df_cleaned['Abstract']=df_cleaned['Abstract'].str.translate(mapping)
df_cleaned['Source title']=df_cleaned['Source title'].str.translate(mapping)
df_cleaned['Author Keywords']=df_cleaned['Author Keywords'].str.translate(mapping)
df_cleaned['Index Keywords']=df_cleaned['Index Keywords'].str.translate(mapping)

df_cleaned['Author Keywords']=df_cleaned['Author Keywords'].fillna('')
df_cleaned['Index Keywords']=df_cleaned['Index Keywords'].fillna('')

nltk.download('stopwords')


nltk_stopwords = stopwords.words('english')

nltk_stopwords += ['mentalhealthmatters','positive','mentalhealthawareness','like','life','amp','feeling','background','recent','however','also','use','using','used'] 

print(type(stopwords.words()), len(stopwords.words()))

print(type(stopwords.words('english')), len(stopwords.words('english')))

def remove_stopwords(in_str):
    new_str=''
    words = in_str.split() #string is splitted through white space in a list of words
    for tx in words:
        if tx not in nltk_stopwords:
            new_str=new_str + tx + " "
    return new_str

df_cleaned['Abstract']=df_cleaned['Abstract'].apply(lambda x: remove_stopwords(x))
df_cleaned['Source title']=df_cleaned['Source title'].apply(lambda x: remove_stopwords(x))

# df_cleaned['Author_Keywords_Abstract'] = df_cleaned['Abstract'] + df_cleaned['Author Keywords']
# df_cleaned['Index_Keywords_Abstract'] = df_cleaned['Abstract'] + df_cleaned['Index Keywords']

# df_cleaned['Author_Keywords_Source_title'] = df_cleaned['Source title'] + df_cleaned['Author Keywords']
# df_cleaned['Index_Keywords_Source_title'] = df_cleaned['Source title'] + df_cleaned['Index Keywords']

# df_cleaned['Author_Keywords_Title'] = df_cleaned['Title'] + df_cleaned['Author Keywords']
# df_cleaned['Index_Keywords_Title'] = df_cleaned['Title'] + df_cleaned['Index Keywords']

df_cleaned['All'] = df_cleaned['Abstract'] + df_cleaned['Source title'] + df_cleaned['Title'] +  df_cleaned['Index Keywords'] + df_cleaned['Author Keywords']

# df_cleaned['Author_Keywords_Title'] = df_cleaned['Author_Keywords_Title'].apply(lambda x: remove_stopwords(x))
# df_cleaned['Index_Keywords_Title'] = df_cleaned['Index_Keywords_Title'].apply(lambda x: remove_stopwords(x))
# df_cleaned['Author_Keywords_Source_title'] = df_cleaned['Author_Keywords_Source_title'].apply(lambda x: remove_stopwords(x))
# df_cleaned['Index_Keywords_Source_title'] = df_cleaned['Index_Keywords_Source_title'].apply(lambda x: remove_stopwords(x))
# df_cleaned['Author_Keywords_Abstract'] = df_cleaned['Author_Keywords_Abstract'].apply(lambda x: remove_stopwords(x))
# df_cleaned['Index_Keywords_Abstract'] = df_cleaned['Index_Keywords_Abstract'].apply(lambda x: remove_stopwords(x))
df_cleaned['All'] = df_cleaned['All'].apply(lambda x: remove_stopwords(x))

# print(df_cleaned['Abstract'].head(10))

#------------------------------------------

from nltk.stem.porter import PorterStemmer

#Create instance of a PorterStemmer
stemmer=PorterStemmer()


def do_stemming(in_str):
    new_str=""
    for word in in_str.split():
        new_str=new_str + stemmer.stem(word) + " "
    return new_str

# trdf["Stemmed"]=trdf["lowered_stop_freq_rare_removed"].apply(lambda x: do_stemming(x))

df_cleaned['Abstract']=df_cleaned['Abstract'].apply(lambda x: do_stemming(x))
df_cleaned['Source title']= df_cleaned['Source title'].apply(lambda x: do_stemming(x))

# df_cleaned['Author_Keywords_Title'] = df_cleaned['Author_Keywords_Title'].apply(lambda x: do_stemming(x))
# df_cleaned['Index_Keywords_Title'] = df_cleaned['Index_Keywords_Title'].apply(lambda x: do_stemming(x))
# df_cleaned['Author_Keywords_Source_title'] = df_cleaned['Author_Keywords_Source_title'].apply(lambda x: do_stemming(x))
# df_cleaned['Index_Keywords_Source_title'] = df_cleaned['Index_Keywords_Source_title'].apply(lambda x: do_stemming(x))
# df_cleaned['Author_Keywords_Abstract'] = df_cleaned['Author_Keywords_Abstract'].apply(lambda x: do_stemming(x))
# df_cleaned['Index_Keywords_Abstract'] = df_cleaned['Index_Keywords_Abstract'].apply(lambda x: do_stemming(x))
df_cleaned['All'] = df_cleaned['All'].apply(lambda x: do_stemming(x))
#Confirm after stemming
# print(trdf["Stemmed"].head(5))
#Note changes in the output, you may not be happy, another option is SnowballStemmer

#---------------
# def remove_frequentwords(in_str):
#     new_str=''
#     words = in_str.split() #string is splitted through white space in a list of words
#     for tx in words:
#         if tx not in nltk_stopwords:
#             new_str=new_str + tx + " "
#     return new_str

# #-------------------------
# from collections import Counter

# def CountFrequent(in_str):
#   counter=Counter()

#   for text in in_str:
#     for word in text.split():
#         counter[word]+=1
        
#   print(type(counter))
#   #list with 10 most frequent word. List is a list of (10) tuples
#   most_cmn_list=counter.most_common(10)
    
#   print(type(most_cmn_list), most_cmn_list) #type is list (list of tuples/word,frequency pair)
    
#   most_cmn_words_list=[]

#   for word, freq in most_cmn_list:
#       most_cmn_words_list.append(word)

#   return most_cmn_words_list 

# #------------------------------------
# #Remove top 10 frequent words 

# most_cmn_words_abstract = CountFrequent(df_cleaned['Abstract'])
# most_cmn_words_src = CountFrequent(df_cleaned['Source title'])

#function to remove words
# def remove_frequent(in_str, cmn_words):
#     new_str=''
#     for word in in_str.split():
#         if word not in cmn_words:
#             new_str=new_str + word + " "
#     return new_str

# df_cleaned['Abstract']=df_cleaned['Abstract'].apply(lambda x: remove_frequent(x,most_cmn_words_abstract))
# df_cleaned['Source title']=df_cleaned['Source title'].apply(lambda x: remove_frequent(x,most_cmn_words_src))

#print(train["lowered_text_stop_removed_freq_removed"].head(10))

# df_cleaned.to_csv('cleaned_first_draft.csv')
# df = pd.read_csv('cleaned_first_draft.csv')

#------------------------
# import matplotlib.pyplot as plt
# import seaborn as sns 
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk_stopwords = nltk.corpus.stopwords.words('english')

#-----------------------------WORDCLOUD--------------
# from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
# def create_wordcloud(text_series, title):
    
#     wc = WordCloud(background_color = 'white', stopwords=nltk_stopwords, height = 600, width = 600)
#     words = ' '.join(w for w in text_series)
#     wc.generate(words)

#     plt.figure(figsize=(10,10))
#     plt.imshow(wc, interpolation='bilinear')
#     plt.title(title, fontsize= 20)
#     plt.axis('off')
#     plt.show()

# create_wordcloud(df['Abstract'], 'Cloud 1')

# #------------------------
# create_wordcloud(df['Author_Keywords_Title'], 'Cloud 1')

# #---------------NLTK---------------
nltk.download('punkt')

from nltk.tokenize import word_tokenize
# df_cleaned['Abstract'] = df_cleaned['Abstract'].apply(word_tokenize)
# df_cleaned['Source title'] = df_cleaned['Source title'].apply(word_tokenize)

# df_cleaned['Title'] = df_cleaned['Title'].apply(word_tokenize)
# df_cleaned['Author_Keywords_Abstract'] = df_cleaned['Author_Keywords_Abstract'].apply(word_tokenize)
# df_cleaned['Index_Keywords_Abstract'] = df_cleaned['Index_Keywords_Abstract'].apply(word_tokenize)
# df_cleaned['Author_Keywords_Title'] = df_cleaned['Author_Keywords_Title'].apply(word_tokenize)
# df_cleaned['Index_Keywords_Title'] = df_cleaned['Index_Keywords_Title'].apply(word_tokenize)
# df_cleaned['Author_Keywords_Source_title'] = df_cleaned['Author_Keywords_Source_title'].apply(word_tokenize)
# df_cleaned['Index_Keywords_Source_title'] = df_cleaned['Index_Keywords_Source_title'].apply(word_tokenize)
df_cleaned['All'] = df_cleaned['All'].apply(word_tokenize)

#------------------TFIDF APPROACH--------------

# from nltk.stem import WordNetLemmatizer


# def tokenizer(sentence, stopwords=nltk_stopwords, lemmatize=True):
#     """
#     Lemmatize, tokenize, crop and remove stop words.
#     """
#     if lemmatize:
#         stemmer = WordNetLemmatizer()
#         tokens = [stemmer.lemmatize(w) for w in word_tokenize(sentence)]
#     else:
#         tokens = [w for w in word_tokenize(sentence)]
#     token = [w for w in tokens if (len(w) > 2 and len(w) < 400
#                                                         and w not in stopwords)]
#     return tokens 

#-------------------

# # Adapt stop words
# token_stop = tokenizer(' '.join(nltk_stopwords), lemmatize=False)

# # Fit TFIDF
# vectorizer = TfidfVectorizer(stop_words=token_stop, tokenizer=tokenizer) 
# tfidf_mat = vectorizer.fit_transform(df_cleaned['All'].apply(lambda x: ' '.join(x)).values) # -> (num_sentences, num_vocabulary)
# tfidf_mat.shape

#-----------------------
# from sklearn.metrics.pairwise import cosine_similarity


# def extract_best_indices(m, topk, mask=None):

#     # return the sum on all tokens of cosinus for each sentence
#     if len(m.shape) > 1:
#         cos_sim = np.mean(m, axis=0) 
#     else: 
#         cos_sim = m
#     index = np.argsort(cos_sim)[::-1] # from highest idx to smallest score 
#     if mask is not None:
#         assert mask.shape == m.shape
#         mask = mask[index]
#     else:
#         mask = np.ones(len(cos_sim))
#     mask = np.logical_or(cos_sim[index] != 0, mask) #eliminate 0 cosine distance
#     best_index = index[mask][:topk]  
#     return best_index


# def get_recommendations_tfidf(sentence, tfidf_mat):

#     # Embed the query sentence
#     tokens = [str(tok) for tok in tokenizer(sentence)]
#     vec = vectorizer.transform(tokens)
#     # Create list with similarity between query and dataset
#     mat = cosine_similarity(vec, tfidf_mat)
#     # Best cosine distance for each token independantly
#     # print(mat.shape)
#     best_index = extract_best_indices(mat, topk=5)
#     return best_index

# query_sentence= 'classification'


# Def to send few random recommendation initially
def sampling(): 
    df = df_cleaned[['Title','Authors', 'Year', 'Link']].sample(5).reset_index(drop=True)
    df.index = df.index + 1
    return df

# Run this while training

#------------------WORD2VEC APPROACH--------------

# Create model
w2v_model = Word2Vec(min_count=0, workers = 8, size=400) 
# Prepare vocab
w2v_model.build_vocab(df_cleaned['All'].values)
# Train
w2v_model.train(df_cleaned['All'].values, total_examples=w2v_model.corpus_count, epochs=30)

#Saving the model for backup
w2v_model.save('vectors.kv')


def prediction_w2v(query_sentence, dataset, model, topk=5):
    query_sentence = query_sentence.split()
    in_vocab_list = []
    best_index = [0]*topk

    for w in query_sentence:
        in_vocab_list.append(w)
    # Retrieve the similarity between two words as a distance

    if len(in_vocab_list) > 0:
        similarity_matrix = np.zeros(len(dataset))  # TO DO
        for i, data_sentence in enumerate(dataset):
            if data_sentence:
                similar_sentence = model.wv.n_similarity(in_vocab_list, data_sentence)
            else:
                similar_sentence = 0
            similarity_matrix[i] = np.array(similar_sentence)
        # Take the five highest norm
        best_index = np.argsort(similarity_matrix)[::-1][:topk]
    return best_index

def recommend(query_sentence):
  query_sentence = query_sentence.lower()
  best_index =prediction_w2v(query_sentence, df_cleaned['All'].values, w2v_model)    
  df = df_cleaned[['Title','Authors', 'Year', 'Link']].iloc[best_index].reset_index(drop=True)
  df.index = df.index + 1
  return df

def more(query_sentence, n):
    query_sentence = query_sentence.lower()
    best_index =prediction_w2v(query_sentence, df_cleaned['All'].values, w2v_model , topk = 5*n)    
    df = df_cleaned[['Title','Authors', 'Year', 'Link']].iloc[best_index].reset_index(drop=True)
    # df = df.iloc[5*n:5*n+1]
    df.index = df.index + 1
    return df

# display(df[['Title']].iloc[best_index])

#----------------------AUTOENCODER APPROACH--------------------------------

# # defining constants / hyperparametrs
# num_words = 2000
# maxlen = 30
# embed_dim = 150
# batch_size = 16


# # preprocessing the input

# from keras.preprocessing.sequence import pad_sequences
# from keras.preprocessing.text import Tokenizer

# tokenizer = Tokenizer(num_words = num_words, split=' ')
# tokenizer.fit_on_texts(df_cleaned['All'].apply(lambda x: ' '.join(x)).values)
# seq = tokenizer.texts_to_sequences(df_cleaned['All'].apply(lambda x: ' '.join(x)).values)
# pad_seq = pad_sequences(seq, maxlen)

# import tensorflow as tf
# import keras
# from keras import Input
# from keras.layers import Embedding,Bidirectional,LSTM,Dense,RepeatVector,Dense
# from keras import Model

# print(pad_seq.shape)


# # creating the encoder model


# encoder_inputs = Input(shape=(maxlen,), name='Encoder-Input')
# emb_layer = Embedding(num_words, embed_dim,input_length = maxlen, name='Body-Word-Embedding', mask_zero=False)
# x = emb_layer(encoder_inputs)
# #encoder LSTM 

# state_h = Bidirectional(LSTM(128, activation='relu', name='Encoder-Last-LSTM'))(x)
# encoder_model = Model(inputs=encoder_inputs, outputs=state_h, name='Encoder-Model')
# seq2seq_encoder_out = encoder_model(encoder_inputs)


# decoded = RepeatVector(maxlen)(seq2seq_encoder_out)
# decoder_lstm = Bidirectional(LSTM(128, return_sequences=True, name='Decoder-LSTM-before'))
# decoder_lstm_output = decoder_lstm(decoded)
# decoder_dense = Dense(num_words, activation='softmax', name='Final-Output-Dense-before')
# decoder_outputs = decoder_dense(decoder_lstm_output)



# # fitting the model
# seq2seq_Model = Model(encoder_inputs, decoder_outputs)
# seq2seq_Model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
# history = seq2seq_Model.fit(pad_seq, np.expand_dims(pad_seq, -1),
#           batch_size=batch_size,
#           epochs=10)



# vecs = encoder_model.predict(pad_seq)




# from sklearn.metrics.pairwise import cosine_similarity
# from sklearn.feature_extraction.text import TfidfVectorizer



# def extract_best_indices(m, topk, mask=None):

#     # return the sum on all tokens of cosinus for each sentence
#     if len(m.shape) > 1:
#         cos_sim = np.mean(m, axis=0) 
#     else: 
#         cos_sim = m
#     index = np.argsort(cos_sim)[::-1] # from highest idx to smallest score 
#     if mask is not None:
#         assert mask.shape == m.shape
#         mask = mask[index]
#     else:
#         mask = np.ones(len(cos_sim))
#     mask = np.logical_or(cos_sim[index] != 0, mask) #eliminate 0 cosine distance
#     best_index = index[mask][:topk]  
#     return best_index


# def get_recommendations_tfidf(vec, vecs):

#     # Embed the query sentence
   
#     # vec = vecs
#     # Create list with similarity between query and dataset
#     mat = cosine_similarity(vec, vecs)
#     # Best cosine distance for each token independantly
#     # print(mat.shape)
#     best_index = extract_best_indices(mat, topk=5)
#     return best_index

# # prediction

# query_sentence= 'deep learning using stock'
# seq = tokenizer.texts_to_sequences(query_sentence)
# pad_seq = pad_sequences(seq, maxlen)
# sentence_vec = encoder_model.predict(pad_seq)

# best_index = get_recommendations_tfidf(sentence_vec, vecs)

# display(df[['Title']].iloc[best_index])



