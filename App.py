import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import requests

movies_dict = pickle.load(open("movies.pkl", "rb"))
movies=pd.DataFrame(movies_dict)
movie_summary_matrix=pickle.load(open("Matrix.pkl", "rb"))
st.title("Movie Recommendation System")
movie_selected=st.selectbox(
    'Select a Movie',
    movies.index)
a=[]
def Recommended(moviename):
    query_index = int(np.where(movies.index == moviename)[0])
    NNR = NearestNeighbors(metric="cosine", algorithm="brute")
    NNR.fit(movie_summary_matrix)
    dist, indices = NNR.kneighbors(movies.iloc[query_index, :].values.reshape(1, -1),
                                   n_neighbors=6)  # neartest 6 distance and index number is obtained

    for i in range(1, len(dist.flatten())):
           a.append(movies.index[indices.flatten()[i]])
    return a

if st.button('Recommend'):
      Recommended(movie_selected)
      for i in a:
         st.write(i)


