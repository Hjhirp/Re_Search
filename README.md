<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/HJHirpara/Re_Search">
    <h2>RE-SEARCH for Researchers</h2>
  </a>
</div>


<!-- ABOUT THE PROJECT -->
## About The Project

Re-Search for Researchers is a recommender system which is built for the use of researchers who face the hassle of finding the research papers they require depending upon their topic of interest.

The recommender system takes the phrase/topic of interest as query input and displays relevant research papers according to the topic entered. The database of the research papers is collected from the various papers available on different subjects on the Scopus web database. 

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- GETTING STARTED -->
## Getting Started

### Prerequisites

* streamlit
* numpy
* pandas
* regex
* scikit-learn
* scipy
* seaborn
* sklearn
* sklearn-pandas
* spacy
* statsmodels
* nltk
* gensim

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/HJHirpara/Re_Search.git
   ```
<p align="right">(<a href="#top">back to top</a>)</p>


# Research-Paper-Summerisation

## Dataset
For this project we prepared a dataset which consists of the research paper abstract data from 5000 recent research papers from 2022, belonging to 10 different subjects of interest. These subjects of interest include- Covid, Data Mining, Cyber Security, Deep Learning,  Internet of Things, Nanotechnology, Climate Change, Natural Language Processing, Big Data and Semi Supervised Learning. Only research papers in English language were considered. 

This data was taken from the database of the Scopus website. The information taken from the data of each research paper was – Author, Author ID, Document Title, Year, EID, Source Title, Volume, Issue, Pages, Citation count, Source & Document Type, Abstract, Author keywords, Index keywords. Research papers belonging to the 10 classes mentioned were downloaded and merged into a single csv file to make it ready for implementation. 

## Data Pre-Processing 

To enhance the performance of our models following pre-processing tasks were carried out: 
Dropping Redundant columns
After deliberation and testing, the features listed below were dropped to avoid null values and irrelevant data
Issue, Article No., Page start, Page end, Page count, Cited by, Document Type, Source

Information rich columns: Abstract, Title, Source Title, Author Keywords & Index Keywords

Converting to Lowercase
The Information rich columns were all converted to lowercase, a common NLP step.
Removing Punctuation
Punctuation marks !"#$%&'()*+,-./:;<=>?@[\]^_{|}~  were removed from the Information rich columns.
Removing Stopwords
Stop words are the most common words in any language and do not add much information to the text. English stop words, which are irrelevant to the data were removed.

Identifying and Removing High Frequency Words and Rare Words
Frequent words are almost always devoid of meaning. They provide very little semantic content and don’t change the meaning of the text. Rare words are often denominated by noise. Hence they are removed. 

Applying WordNetLemmetizer
The Information rich columns were Lemmatized to convert the words into their root form. 

Tokenize
These columns were tokenized.

After the data-preprocessing, all Information rich columns were merged into a single ‘ALL’ column. 

## Recommendation System Approaches

We tried the following approaches for our recommendation system:

TF-IDF – Cosine Similarity:
Term Frequency — Inverse Document Frequency is a technique to quantify words in a set of documents. We generally compute a score for each word to signify its importance in the document. By computing the Cosine similarity between the TF-IDF score of our query and each Research Paper in our dataset we are able to rank them according to their relevance. 

Word2Vec:
Word2vec algorithm uses a neural network model to learn word associations from a large corpus of text. Once trained, such a model can detect synonymous words or suggest additional words for a partial sentence. Its input is a text corpus and its output is a set of vectors: feature vectors that represent words in that corpus. Word2Vec helps us find the transitional probabilities between words which are discreet values: the likelihood that they will co-occur. We create a similarity matrix using Word2Vec and use it to find and sort the relevant results in order of their relevance. 

Autoencoder:
Auto encoder is an unsupervised learning technique which is used  mainly for compression of sparse data. It is also used for feature extraction in image processing. In our project we tried to use the auto encoder to do word embeddings as we have a large bag of words. Auto encoder will help us to compress the bag of words and give some relationship between each words. It will result in a numerical embedding matrix which will help us to find the cosine similarity between the research documents.

Best results were achieved using the Word2Vec approach. 

<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- CONTACT -->
## Contact

<p>Aditya Sud - (https://www.linkedin.com/in/adityavsud)</p>
<p>Harshal Hirpara - (https://www.linkedin.com/in/harshaljhirpara)</p>
<p>Tanmay Joshi - (https://www.linkedin.com/in/tanmay-joshi-59bb5b214/)</p>
<p>Manan Patel - (https://www.linkedin.com/in/manan-patel-0299a9202/)</p>
<p>Manushi Munshi - (https://www.linkedin.com/in/ManushiMunshi/)</p>

Project Link: [https://github.com/HJHirpara/Re_Search](https://github.com/HJHirpara/Re_Search)

<p align="right">(<a href="#top">back to top</a>)</p>

