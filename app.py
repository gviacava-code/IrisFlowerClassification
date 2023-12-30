import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# page configuration
st.set_page_config(
    page_title='Iris Dataset - ML Classification',
    layout='wide',
    initial_sidebar_state='expanded'
)

# Import the dataset & create the dataframe
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target_names[iris.target]
df.columns = [c.replace(' (cm)', '') for c in df.columns]
df = df[['sepal length', 'sepal width', 'petal length', 'petal width', 'target']]

# title of the app
st.title('Iris Dataset - ML Classification')

# input widgets
st.sidebar.subheader('Input features')
sepal_length = st.sidebar.slider('Sepal length', 4.3, 7.9, 5.84)
sepal_width = st.sidebar.slider('Sepal width', 2.0, 4.4, 3.6)
petal_length =  st.sidebar.slider('Petal length', 1.0, 6.9, 3.76)
petal_width = st.sidebar.slider('Petal width', 0.1, 2.5, 1.2)

# import the model
model=pickle.load(open('model.pkl', 'rb'))

# Print EDA
st.subheader('Brief EDA')
st.write('The data that is grouped by the class and the viarible mean is computed for each class.')
groupby_target_mean=df.groupby('target').mean()
st.write(groupby_target_mean)
st.line_chart(groupby_target_mean.T)

# Print input features
st.subheader('Input Features')
input_features =  pd.DataFrame(data=[[sepal_length, sepal_width, petal_length, petal_width]], columns=['sepal_length', 'sepal_width', 'petal_length','petal_width'])
st.write(input_features)

# Make predictions
st.subheader('Predicted Class')
ouput=model.predict([[sepal_length,sepal_width,petal_length,petal_width]])
st.write(iris.target_names[ouput])
