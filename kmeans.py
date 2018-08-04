import pandas as pd
import numpy as np
import json
import operator
import sys
from sklearn.cluster import KMeans
from heapq import nlargest




#import dataset
X = pd.read_csv('tmdb_5000_movies.csv')
Y = pd.read_csv('tmdb_5000_credits.csv')

"""

['budget', 'genres', 'homepage', 'id', 'keywords', 
'original_language', 'original_title', 'overview', 
'popularity','production_companies', 
'production_countries', 'release_date', 
'revenue', 'runtime', 'spoken_languages', 
'status', 'tagline', 'title', 'vote_average', 
'vote_count', 'cast', 'crew']

"""

#merge two dataset X and Y
data = pd.concat([X,Y.iloc[:,2:]],axis=1)

#delete irrelevant columns
data.drop('homepage', axis=1, inplace=True)
data.drop('popularity', axis=1, inplace=True)
data.drop('spoken_languages', axis=1, inplace=True)
data.drop('status', axis=1, inplace=True)
data.drop('title', axis=1, inplace=True)
data.drop('vote_count', axis=1, inplace=True)

#collect important words
def popular(feature):
    dic = {}
    
    for example in data[feature]:
        diclist = json.loads(example)
        counter = 0
        for i in diclist:
            if feature == 'crew':
                if i["job"] == "Director":
                    if i["name"] not in dic:
                        dic[i["name"]] = 1
                    else:
                        dic[i["name"]] +=1
                        
            elif feature == 'cast':
                while counter <= 10:
                    if i["name"] not in dic:
                        dic[i["name"]] = 1
                        counter += 1
                    else:
                        dic[i["name"]] += 1
                        counter += 1
            else:
                if i["name"] not in dic:
                    dic[i["name"]] = 1
                else:
                    dic[i["name"]] += 1
                
    n_largest = nlargest(10, dic, key = dic.get)
            
    return n_largest

themes = popular('genres')
words = popular('keywords')
companies = popular('production_companies')
countries = popular('production_countries')
actors = popular('cast')
directors = popular('crew')

X = User(themes,words,companies,countries,actors,directors)

class User:
    def __init__(self,themes,words,companies,countries,actors,directors):
        self.themes = themes
        self.words = words
        self.companies = companies
        self.countries = countries
        self.actors = actors
        self.directors = directors
        
    def user_input(self):
        x = input("Your Answer: ")
        print()
        return x

    def display(self,alist,item):
        print("Please select a {} by typing corresponding number.".format(item))
        for i in range(len(alist)):
            print("{}\t{}".format(i,alist[i]))
            
        if item != 'genre':
            print("If you want to skip, please type (s)")

    def check(self,ui):
        if self.ui == 'q':
            sys.exit(2)
    
    def start(self):
        print("Welcome to Sujoung's movie recommender.")
        print("If you want to quit, press (q) button.")
        print()
        
        print("What is your name?")
        self.name = self.user_input()
        print("Hello, {}!".format(self.name))

        self.display(self.themes,'genre')
        u_theme = self.user_input()
        self.check(u_theme)
        
        
    

#preprocess the data
# -- depending on important(frequent) words,
#    select 20 different examples as test set.
# -- tokenization of overview, title, tagline

#collecting user input
# -- make a function that takes user input
# -- genre(must),actor(optional),director(optional)
#    data should be searched based on the frequency of id in clusters.
# -- select word embeddings method, clustering function
# -- select features for model

#form a vectorspace with different word embeddings
# -- set a data shape
# -- create function with sklean.tfidfVectorizer
# -- create function with countvector
# -- create function with continuous bag of words
# -- make a function that makes dummy
# -- set a function for making vector space

#run the kmeans clustering
# -- create two functions(1. sklearn kmeans, 2.sklearn minibatch kmeans, 3.tensorflow)
# -- input vectors
# -- run the kmeans clustering with cluster = 10(not fixed)
# -- map the data by group
# -- present the group with title
# -- Input test examples and present the title by group

#Evaluation ( asking people, analysis etc.)
# -- Hand made evaluation
# -- ask people
