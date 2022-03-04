#import required stuff
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
nltk.download('wordnet')

import re #for working with regular expression
import nltk #for natural language processing (nlp)
# import spacy #also for nlp
import string #This is a module, Python also has built-in class str, these are different

df_cleaned = pd.read_csv('df_final.csv')

df_cleaned.drop(['Authors', 'Year', 'Issue', 'Art. No.', 'Page start', 'Page end', 
               'Page count', 'Cited by', 'Link', 'Document Type', 
               'Source'],axis=1, inplace=True)

#Let us see what do we have. 
print(df_cleaned.head(3))
#Note that he have a text column, which we will use n this demo

#Info on the training set
df_cleaned.info()

df_cleaned['Abstract']=df_cleaned['Abstract'].str.lower()
# df_cleaned['Title']=df_cleaned['Title'].str.lower()
df_cleaned['Source title']=df_cleaned['Source title'].str.lower()
df_cleaned['Author Keywords']=df_cleaned['Author Keywords'].str.lower()
df_cleaned['Index Keywords']=df_cleaned['Index Keywords'].str.lower()

punctuation=string.punctuation
#print(type(punctuation), punctuation)

mapping=str.maketrans("","",punctuation)
#print(type(mapping), mapping)

df_cleaned['Abstract']=df_cleaned['Abstract'].str.translate(mapping)
# df_cleaned['Title']=df_cleaned['Title'].str.translate(mapping)
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
# df_cleaned['Title']=df_cleaned['Title'].apply(lambda x: remove_stopwords(x))
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

#df_cleaned['Author Keywords']=df_cleaned['Author Keywords'].apply(lambda x: remove_stopwords(x))
#df_cleaned['Index Keywords']=df_cleaned['Index Keywords'].apply(lambda x: remove_stopwords(x))

print(df_cleaned['Abstract'].head(10))
#notice the removal of stopwords in line 0, 2, 4, 5, 7, 8 and 9.

#------------------------------------------

from nltk.stem.porter import PorterStemmer

#Create instance of a PorterStemmer
stemmer=PorterStemmer()

#Before Stemming
# print(trdf["lowered_stop_freq_rare_removed"].head(5))

def do_stemming(in_str):
    new_str=""
    for word in in_str.split():
        new_str=new_str + stemmer.stem(word) + " "
    return new_str

# trdf["Stemmed"]=trdf["lowered_stop_freq_rare_removed"].apply(lambda x: do_stemming(x))

df_cleaned['Abstract']=df_cleaned['Abstract'].apply(lambda x: do_stemming(x))
# df_cleaned['Title']=df_cleaned['Title'].apply(lambda x: do_stemming(x))
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
def remove_frequentwords(in_str):
    new_str=''
    words = in_str.split() #string is splitted through white space in a list of words
    for tx in words:
        if tx not in nltk_stopwords:
            new_str=new_str + tx + " "
    return new_str

#-------------------------
from collections import Counter

def CountFrequent(in_str):
  counter=Counter()

  for text in in_str:
    for word in text.split():
        counter[word]+=1
        
  print(type(counter))
  #list with 10 most frequent word. List is a list of (10) tuples
  most_cmn_list=counter.most_common(10)
    
  print(type(most_cmn_list), most_cmn_list) #type is list (list of tuples/word,frequency pair)
    
  most_cmn_words_list=[]

  for word, freq in most_cmn_list:
      most_cmn_words_list.append(word)

  return most_cmn_words_list 

#------------------------------------
#Remove top 10 frequent words 

most_cmn_words_abstract = CountFrequent(df_cleaned['Abstract'])
# most_cmn_words_title = CountFrequent(df_cleaned['Title'])
most_cmn_words_src = CountFrequent(df_cleaned['Source title'])

#function to remove words
def remove_frequent(in_str, cmn_words):
    new_str=''
    for word in in_str.split():
        if word not in cmn_words:
            new_str=new_str + word + " "
    return new_str

df_cleaned['Abstract']=df_cleaned['Abstract'].apply(lambda x: remove_frequent(x,most_cmn_words_abstract))
# df_cleaned['Title']=df_cleaned['Title'].apply(lambda x: remove_frequent(x,most_cmn_words_title))
df_cleaned['Source title']=df_cleaned['Source title'].apply(lambda x: remove_frequent(x,most_cmn_words_src))

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
df_cleaned['Abstract'] = df_cleaned['Abstract'].apply(word_tokenize)
# df_cleaned['Title'] = df_cleaned['Title'].apply(word_tokenize)
df_cleaned['Source title'] = df_cleaned['Source title'].apply(word_tokenize)

# df_cleaned['Author_Keywords_Abstract'] = df_cleaned['Author_Keywords_Abstract'].apply(word_tokenize)
# df_cleaned['Index_Keywords_Abstract'] = df_cleaned['Index_Keywords_Abstract'].apply(word_tokenize)
# df_cleaned['Author_Keywords_Title'] = df_cleaned['Author_Keywords_Title'].apply(word_tokenize)
# df_cleaned['Index_Keywords_Title'] = df_cleaned['Index_Keywords_Title'].apply(word_tokenize)
# df_cleaned['Author_Keywords_Source_title'] = df_cleaned['Author_Keywords_Source_title'].apply(word_tokenize)
# df_cleaned['Index_Keywords_Source_title'] = df_cleaned['Index_Keywords_Source_title'].apply(word_tokenize)
df_cleaned['All'] = df_cleaned['All'].apply(word_tokenize)

#------------------TFIDF--------------

from nltk.stem import WordNetLemmatizer


def tokenizer(sentence, stopwords=nltk_stopwords, lemmatize=True):
    """
    Lemmatize, tokenize, crop and remove stop words.
    """
    if lemmatize:
        stemmer = WordNetLemmatizer()
        tokens = [stemmer.lemmatize(w) for w in word_tokenize(sentence)]
    else:
        tokens = [w for w in word_tokenize(sentence)]
    token = [w for w in tokens if (len(w) > 2 and len(w) < 400
                                                        and w not in stopwords)]
    return tokens 

#-------------------
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Adapt stop words
token_stop = tokenizer(' '.join(nltk_stopwords), lemmatize=False)

# Fit TFIDF
vectorizer = TfidfVectorizer(stop_words=token_stop, tokenizer=tokenizer) 
tfidf_mat = vectorizer.fit_transform(df_cleaned['All'].apply(lambda x: ' '.join(x)).values) # -> (num_sentences, num_vocabulary)
tfidf_mat.shape

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


def sampling():
    return df_cleaned[['Title']].sample(5)

from gensim.models.word2vec import Word2Vec

# Create model
word2vec_model = Word2Vec(min_count=0, workers = 8, vector_size=400) 
# Prepare vocab
word2vec_model.build_vocab(df_cleaned['All'].values)
# Train
word2vec_model.train(df_cleaned['All'].values, total_examples=word2vec_model.corpus_count, epochs=30)

word2vec_model.save('vectors.kv')

from gensim.models import KeyedVectors

def is_word_in_model(word, model):
    """
    Check on individual words ``word`` that it exists in ``model``.
    """
    reloaded_word_vectors = KeyedVectors.load('vectors.kv')
    is_in_vocab = word in reloaded_word_vectors.key_to_index.keys()
    return is_in_vocab

def predict_w2v(query_sentence, dataset, model, topk=5):
    query_sentence = query_sentence.split()
    in_vocab_list, best_index = [], [0]*topk
    for w in query_sentence:
        # remove unseen words from query sentence
        # if is_word_in_model(w, model.wv):
        in_vocab_list.append(w)
    # Retrieve the similarity between two words as a distance
    if len(in_vocab_list) > 0:
        sim_mat = np.zeros(len(dataset))  # TO DO
        for i, data_sentence in enumerate(dataset):
            if data_sentence:
                sim_sentence = model.wv.n_similarity(in_vocab_list, data_sentence)
            else:
                sim_sentence = 0
            sim_mat[i] = np.array(sim_sentence)
        # Take the five highest norm
        best_index = np.argsort(sim_mat)[::-1][:topk]
    return best_index

def recommend(query_sentence):
  best_index =predict_w2v(query_sentence, df_cleaned['All'].values, word2vec_model)    
  return df_cleaned[['Title']].iloc[best_index]
# display(df[['Title']].iloc[best_index])
