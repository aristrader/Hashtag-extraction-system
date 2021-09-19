import numpy as np
import pickle
import re
import nltk
from sklearn.ensemble import RandomForestClassifier
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd


# def predict():
pkl_filename = "pickle_model.pkl"
model, vectorize, model2, vectorize2 = pickle.load(open(pkl_filename, 'rb'))    
tweets_file = pd.read_csv("tweets.csv")
# flags,comments_id,comments = list(comment_file["flag"]),list(comment_file["comments_id"]),list(comment_file["comments"])   
tweets=tweets_file.iloc[:, 0].values
processed_tweets = []
hashtags = []
# print(tweets)
for sentence in range(0, len(tweets)):
    hashtag=re.findall(r"#(\w+)", str(tweets[sentence]))
    processed_tweet = re.sub(r"#(\w+)", ' ', str(tweets[sentence]))
    processed_tweet = re.sub(r'\W', ' ', processed_tweet)
    # remove all single characters
    processed_tweet= re.sub(r'\s+[a-zA-Z]\s+', ' ', processed_tweet)
    # Remove single characters from the start
    processed_tweet = re.sub(r'\^[a-zA-Z]\s+', ' ', processed_tweet) 
    # Substituting multiple spaces with single space
    processed_tweet = re.sub(r'\s+', ' ', processed_tweet, flags=re.I)
    # Removing prefixed 'b'
    processed_tweet = re.sub(r'^b\s+', '', processed_tweet)
    
    # Converting to Lowercase
    processed_tweet = processed_tweet.lower()
    # print(processed_feature)
    hash=''
    for word in range(0,len(hashtag)): 
        hash=hash+hashtag[word]+' '
    processed_tweets.append(processed_tweet)
    hashtags.append(hash)
# print(processed_tweets)
# print(hashtags)
X_predict = vectorize.transform(processed_tweets).toarray()
Y_predict = model.predict_proba(X_predict)
X2_predict = vectorize2.transform(hashtags).toarray()
Y2_predict = model2.predict_proba(X2_predict)
predictions =[]
# print(Y_predict)        
# print(Y2_predict)
for pr in range(0,len(Y2_predict)):
    # print(Y2_predict[pr][1])
    if Y2_predict[pr][1]==0.4161057735809483:
        if Y_predict[pr][1]>=0.5:
            predictions.append(1)
        else:
            predictions.append(0)
    else:
        if (Y2_predict[pr][1]+Y_predict[pr][1]>=1 or Y2_predict[pr][1]>=0.75 or Y_predict[pr][1]>=0.65):
            predictions.append(1)
        else :
            predictions.append(0)         
# print(Y_predict)        
# print(Y2_predict)
print(predictions)
