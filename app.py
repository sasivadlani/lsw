import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import datetime
import pandas as pd
from flask import make_response
import pandas as pd
import numpy as np
import os # Operating System
import numpy as np
import pandas as pd
import datetime as dt # Datetime
import json # library to handle JSON files
from waitress import serve
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import requests # library to handle requests
from pandas.io.json import json_normalize # tranform JSON file into a pandas dataframe
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer
from sklearn.preprocessing import normalize

# Matplotlib and associated plotting modules
import matplotlib.cm as cm
import matplotlib.colors as colors
from bs4 import BeautifulSoup
import nltk
nltk.download('stopwords')  # Download stopwords from NLTK
nltk.download('wordnet')  # Download WordNet from NLTK
nltk.download('omw-1.4')  # Download Open Multilingual WordNet from NLTK
from nltk.corpus import stopwords  # Import stopwords from NLTK
from nltk.stem import PorterStemmer  # Import PorterStemmer from NLTK
from nltk.stem import WordNetLemmatizer
import unicodedata

# Regular expressions
import re

# Tokenization
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize

# Part of speech tagging and named entity recognition
from nltk import pos_tag
from nltk import ne_chunk
#!conda install -c conda-forge folium=0.5.0 --yes
import folium #import folium # map rendering library
app = Flask(__name__)
#model = pickle.load(open('models/blocknaive36', 'rb'))
#modeltime=pickle.load(open('models/finalized_modelmain', 'rb'))

# @app.route('/predict2',methods=['POST'])
# def predict2():
#     int_features = [int(x) for x in request.form.values()]
#     final_features = np.array([np.array(int_features)])
#     # if(final_features.shape[0]!=0&final_features.shape[1]!=0):
#     prediction = modeltime.predict(final_features)
#     output = prediction

#     return render_template('time.html', prediction_text='Crime Predicted {}'.format(output))
#     # else:
#     #     return render_template('time.html')
def remove_html_tags_func(text):
    return BeautifulSoup(text, 'html.parser').get_text()
#The function returns string s with no html tags
def remove_url_func(text):
    return re.sub(r'https?://\S+|www\.\S+', '', text)

def remove_accented_chars_func(text):
    '''
    Removes all accented characters from a string, if present
    
    Args:
        text (str): String to which the function is to be applied, string
    
    Returns:
        Clean string without accented characters
    '''
    return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')

def remove_punctuation_func(text):
    '''
    Removes all punctuation from a string, if present
    
    Args:
        text (str): String to which the function is to be applied, string
    
    Returns:
        Clean string without punctuations
    '''
    return re.sub(r'[^a-zA-Z0-9]', ' ', text)

def remove_irr_char_func(text):
    '''
    Removes all irrelevant characters (numbers and punctuation) from a string, if present
    
    Args:
        text (str): String to which the function is to be applied, string
    
    Returns:
        Clean string without irrelevant characters
    '''
    return re.sub(r'[^a-zA-Z]', ' ', text)

def remove_extra_whitespaces_func(text):
    '''
    Removes extra whitespaces from a string, if present
    
    Args:
        text (str): String to which the function is to be applied, string
    
    Returns:
        Clean string without extra whitespaces
    ''' 
    return re.sub(r'^\s*|\s\s*', ' ', text).strip()

def preprocess_text(text):
    # Remove punctuation, numbers, and stopwords
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    #stop_words = set(stopwords.words('english'))Z
    words = text.split()
    #words = [w for w in words if not w.lower() in stop_words]
    # Lemmatize the words
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(lemmatized_words)
stemmer = PorterStemmer()
tf_idf_vectorizer = TfidfVectorizer(ngram_range=(1, 3))
def get_top_features_cluster(tf_idf_array, prediction, n_feats):
    labels = np.unique(prediction)
    dfs = []
    for label in labels:
        id_temp = np.where(prediction==label) # indices for each cluster
        x_means = np.mean(tf_idf_array[id_temp], axis = 0) # returns average score across cluster
        sorted_means = np.argsort(x_means)[::-1][:n_feats] # indices with top 20 scores
        features = tf_idf_vectorizer.get_feature_names_out()
        best_features = [(features[i], x_means[i]) for i in sorted_means]
        df = pd.DataFrame(best_features, columns = ['features', 'score'])
        dfs.append(df)
    return dfs
def kmeans_zoom(features, k, num_iterations):
    # Initialize cluster centroids randomly
    centroids = features[np.random.choice(len(features), k, replace=False)]
    for iteration in range(num_iterations):
        # Assign each feature to the closest centroid
        distances = np.linalg.norm(features[:, np.newaxis] - centroids, axis=2)
        cluster_assignments = np.argmin(distances, axis=1)
        # Update each centroid as the mean of the features assigned to it
        for i in range(k):
            centroid_features = features[cluster_assignments == i]
            if len(centroid_features) > 0:
                centroids[i] = np.mean(centroid_features, axis=0)
    # Assign each feature to the closest centroid and return the cluster assignments
    distances = np.linalg.norm(features[:, np.newaxis] - centroids, axis=2)
    cluster_assignments = np.argmin(distances, axis=1)
    return cluster_assignments
def preprocess_text(text):
  text = re.sub(r'[^\w\s]', '', text)
  text = re.sub(r'\d+', '', text)
  words = text.split()
  lemmatizer = WordNetLemmatizer()
  lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
  return ' '.join(lemmatized_words)
def stem_text(text):
    return stemmer.stem(text)

@app.route('/')
def home():
    return render_template('index.html')
@app.route('/indexzoom')
def indexzoom():
    return render_template('indexzoom.html')


@app.route('/predict3',methods=['POST'])
def predict3():
    int_features = [int(x) for x in request.form.values()]
    final_features = np.array([np.array(int_features)])
    n_clusters=final_features[0][0]
    df = pd.read_csv("webex_data.csv")
    df_c = df.copy()
    df=df['feature_discription']
    df = df.astype(str)
    df = df.str.lower()
    df = df.apply(remove_html_tags_func)
    df = df.apply(remove_url_func)
    df = df.apply(remove_accented_chars_func)
    df = df.apply(remove_punctuation_func)
    df = df.apply(remove_irr_char_func)
    df = df.apply(remove_extra_whitespaces_func)
    stop_words = stopwords.words('english')
    df = df.apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))
    preprocessed_data = [preprocess_text(str(text)) for text in df]
    df = df.apply(stem_text)
    
    vectorizer = CountVectorizer()
    bag_of_words = vectorizer.fit_transform(df)
    bag_of_words = pd.DataFrame(bag_of_words.toarray(), columns = vectorizer.get_feature_names_out())
    bag_of_words=vectorizer.transform(df)
    kmeans = KMeans(n_clusters, random_state=0, n_init=10).fit(bag_of_words)
    df_c['cluster'] = kmeans.labels_
    old_feature = np.array([])
    new_feature = np.array([])
    feature_dictionary=[]
    for x in range(n_clusters):
        cluster_data = df_c[df_c['cluster'] == x]
        old_count = (cluster_data['release_month'].astype(int) < 6).sum()
        old_feature = np.append(old_feature, old_count)

        new_count = (cluster_data['release_month'].astype(int) >= 6).sum()
        new_feature = np.append(new_feature, new_count)
        feature_dictionary.append(dict(Total_Features=cluster_data.shape[0],Older_Features=old_count,Newer_Features=new_count))
    #output = new_feature
    X = tf_idf_vectorizer.fit_transform(preprocessed_data)
    X_norm = normalize(X)
    X_arr = X_norm.toarray()
    kmeans_map = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
    prediction = kmeans_map.labels_
    dfs = get_top_features_cluster(X_arr, prediction, 15)
    fig = plt.figure(figsize=(15,12))
    x = np.arange(len(dfs[0]))
    for i, df in enumerate(dfs):
        ax = fig.add_subplot(1, len(dfs), i+1)
        ax.set_title("Cluster: "+ str(i), fontsize = 14)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_frame_on(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-2,2))
        ax.barh(x, df.score.tolist(), align='center', color='skyblue')
        ax.set_yticks(x)
        yticks = ax.set_yticklabels(df.features.tolist())
    plt.savefig('static/Images2/plot.png')
    
    return render_template('index.html', prediction_text='Clusters{}'.format(feature_dictionary))
    # else:
    #     return render_template('time.html')
@app.route('/datasetweb')
def datasetweb():
    dfp = pd.read_csv('webex_data.csv')
    return render_template('datasetweb.html', tables=[dfp.to_html()], titles=[''])


@app.route('/datasetzoom')
def datasetzoom():
    dfp = pd.read_csv('Zoom-features-2022.csv')
    return render_template('datasetzoom.html', tables=[dfp.to_html()], titles=[''])
@app.route('/predict4',methods=['POST'])
def predict4():
    dfzoom = pd.concat(pd.read_excel('Zoom-features-2022.xlsx', sheet_name=None), ignore_index=True)
    preprocessed_data = [preprocess_text(str(text)) for text in dfzoom['Feature Description'].tolist()]
    vectorizer = TfidfVectorizer()
    vectorized_data = vectorizer.fit_transform(preprocessed_data)
    int_features = [int(x) for x in request.form.values()]
    final_features = np.array([np.array(int_features)])
    k=final_features[0][0]
    num_iterations = 1000
    cluster_assignments = kmeans_zoom(vectorized_data.toarray(), k, num_iterations)
    dfzoom['cluster'] = cluster_assignments
    dfzoom.to_excel('Zoom-features-clustered.xlsx', index=False)
    feature_dictionary=[]
    for x in range(k):
        cluster_data = dfzoom[dfzoom['cluster'] == x]
        old_count = (pd.to_datetime(cluster_data['Release Date'], format='%B %d, %Y') < pd.to_datetime('2022-06-01')).sum()
        new_count = (pd.to_datetime(cluster_data['Release Date'], format='%B %d, %Y') >= pd.to_datetime('2022-06-01')).sum()

        feature_dictionary.append(dict(Total_Features=cluster_data.shape[0],Older_Features=old_count,Newer_Features=new_count))
    return render_template('indexzoom.html', prediction_text='Clusters{}'.format(feature_dictionary))
@app.route('/contact')
def contact():
    return render_template('contact.html')
@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run()