import numpy as np 
import pandas as pd 
import re
import nltk 
import pickle
# import matplotlib.pyplot as plt

data_source = "train.csv"
tweets = pd.read_csv(data_source)
features = tweets.iloc[:, 3].values
labels = tweets.iloc[:, 4].values
processed_features = []
hashtags =[]
for sentence in range(0, len(features)):
    # Remove all the special characters
    hashtag=re.findall(r"#(\w+)", str(features[sentence]))
    processed_feature = re.sub(r"#(\w+)", ' ', str(features[sentence]))
    processed_feature = re.sub(r'\W', ' ', processed_feature)
    # remove all single characters
    processed_feature= re.sub(r'\s+[a-zA-Z]\s+', ' ', processed_feature)
    # Remove single characters from the start
    processed_feature = re.sub(r'\^[a-zA-Z]\s+', ' ', processed_feature) 
    # Substituting multiple spaces with single space
    processed_feature = re.sub(r'\s+', ' ', processed_feature, flags=re.I)
    # Removing prefixed 'b'
    processed_feature = re.sub(r'^b\s+', '', processed_feature)
    # Converting to Lowercase
    processed_feature = processed_feature.lower()
    # print(processed_feature)
    hash=''
    for word in range(0,len(hashtag)): 
        hash=hash+hashtag[word]+' '
    processed_features.append(processed_feature)
    hashtags.append(hash)
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer (max_features=2500,min_df=5,stop_words=stopwords.words('english'))
vectorizer2 = TfidfVectorizer (max_features=1500,min_df=2,stop_words=stopwords.words('english'))
processed_features = vectorizer.fit_transform(processed_features).toarray()
hashtags = vectorizer2.fit_transform(hashtags).toarray()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(processed_features, labels, test_size=0.18, random_state=4)
X2_train, X2_test, y2_train, y2_test = train_test_split(hashtags, labels, test_size=0.18, random_state=4)
from sklearn.ensemble import RandomForestClassifier
model =RandomForestClassifier(random_state=0)
model2 =RandomForestClassifier(random_state=0)
model.fit(X_train, y_train)
model2.fit(X2_train,y2_train)
prediction = model.predict(X_test)
prediction2 = model2.predict(X2_test)
# print(type(X_test))
# tuple_objects = (model, vectorizer)
tuple_objects = (model, vectorizer, model2, vectorizer2) 
pkl_filename = "pickle_model.pkl"
pickle.dump(tuple_objects, open(pkl_filename, 'wb'))    
from sklearn.metrics import classification_report
print(classification_report(y_test, prediction))
print(classification_report(y2_test, prediction2))
