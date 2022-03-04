from random import sample
from turtle import width
import streamlit as st
# st.write("hersulllll and manen bot sweetie")
from model import recommend
from model import sampling
import pandas as pd
from PIL import Image


st.set_page_config(
  page_title= "Main Page",
  layout="wide",
  initial_sidebar_state="expanded",
)

st.set_option('deprecation.showPyplotGlobalUse', False)
st.sidebar.markdown('<h1 style="margin-left:8%; color:	#ffffff ">Re - Search for Researchers </h1>',
                    unsafe_allow_html=True)

add_selectbox = st.sidebar.radio(
    "",
    ("Home", "Project Summary", "Visualization", "Conclusion", "Team info")
)

if add_selectbox == 'Home':
  def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

  def remote_css(url):
      st.markdown(f'<link href="{url}" rel="stylesheet">', unsafe_allow_html=True)    

  def icon(icon_name):
      st.markdown(f'<i class="material-icons">{icon_name}</i>', unsafe_allow_html=True)

  local_css("style.css")
  remote_css('https://fonts.googleapis.com/icon?family=Material+Icons')

  icon("search")
  selected = st.text_input("", "Search...")
  button_clicked = st.button("OK")

  if button_clicked:
    output = recommend(selected)

    data = [['tommmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm', 10], ['nick', 15], ['juli', 14]]
    df = pd.DataFrame(data, columns = ['Name', 'Age'])

    st.dataframe(output, width= 1500 , height = 900)
  else:
    output = sampling()
    data = [['tommmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm', 10], ['nick', 15], ['juli', 14]]
    df = pd.DataFrame(data, columns = ['Name', 'Age'])
    st.dataframe(output, width= 1500 , height = 900)

elif add_selectbox == 'Project Summary':
  with st.container():
    st.title("Data Collection")
    st.write("Data for this project would be from the research papers available across various platforms. We have taken the data available from research papers on the Scopus website database. Our dataset consists of data from 5000 research paper belonging to 10 different topics. The required features for the recommendation system have been extracted. Data cleaning and data pre-processing has been performed  through lemmatization, porter stemmer and tokenization. Data is made ready for modelling.")
  
  with st.container():
    st.title("Data Modelling")
    st.write("Model Building Review based approach has been used to train data and build the model. The recommendation system takes a phrase as query input and recommends top 5 papers relevant to the query. Algorithms used for model building are: \n 1.	TF-IDF Similarity\n 2.	Cosine Similarity\n3. Word2Vec")

  with st.container():
    st.title("UI Design")
    st.write("Streamlit has been used for UI application.  The UI representation has a sidebar with 5 filters displayed. The 5 filters are: \n1.	Home\n2.	Project Summary\n3.	Visualizations\n4.	Conclusion\n5.	Team Information")


elif add_selectbox == 'Team info':
    st.title("Our Team")
    st.subheader('MineD Hackathon 2022')
    st.write('Team Name- Team_Deadpool')
    st.write('College- Institute of Technology, Nirma University')
    st.write('Project Definition - Making Scientific Research Accesible using AI and Big Data')
    st.write('Project Title - Recommender System for Researchers')
    st.write('Application Name - Re-Search for Researchers')
    st.subheader('COLLABORATORS')
    st.markdown('• <a href="https://www.linkedin.com/in/adityavsud/">Aditya Sud</a>',
                unsafe_allow_html=True)

    st.markdown('• <a href="https://www.linkedin.com/in/harshaljhirpara/">Harshal Hirpara</a>',
                unsafe_allow_html=True)

    st.markdown('• <a href="https://www.linkedin.com/in/tanmay-joshi-59bb5b214/">Tanmay Joshi</a>',
                unsafe_allow_html=True)

    st.markdown('• <a href="https://www.linkedin.com/in/manan-patel-0299a9202/">Manan Patel</a>',
                unsafe_allow_html=True)
    st.markdown('• <a href="https://www.linkedin.com/in/ManushiMunshi/">Manushi Munshi</a>',
                unsafe_allow_html=True)

elif add_selectbox == 'Visualization':
  st.title("Visualize")
  image1 = Image.open('image1.png')
  image2 = Image.open('image2.png')
  image3 = Image.open('image3.png')
  st.image(image1, caption='Abstract')
  st.image(image2, caption='Title')
  st.image(image3, caption='All')