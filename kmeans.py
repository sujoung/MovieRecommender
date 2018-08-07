import pandas as pd
from gensim.models import Word2Vec
import logging
from nltk import word_tokenize
import numpy as np
import random
import json
import operator
import sys
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from heapq import nlargest

# import dataset
X = pd.read_csv('tmdb_5000_movies.csv')
Y = pd.read_csv('tmdb_5000_credits.csv')

# merge two dataset X and Y
data = pd.concat([X, Y.iloc[:, 2:]], axis=1)

# delete irrelevant columns
data.drop('homepage', axis=1, inplace=True)
data.drop('spoken_languages', axis=1, inplace=True)
data.drop('status', axis=1, inplace=True)
data.drop('title', axis=1, inplace=True)
data.drop('vote_count', axis=1, inplace=True)

# split train and test set
train, test = train_test_split(data, test_size=0.2)


# collect important words
def popular(feature):
    dic = {}
    li = []

    for example in train[feature]:
        diclist = json.loads(example)
        counter = 0
        minilist = []
        for i in diclist:
            new_name = i["name"].replace(" ", "_")
            if feature == 'crew':
                if i["job"] == "Director":
                    minilist.append(new_name)
                    if new_name not in dic:
                        dic[new_name] = 1
                    else:
                        dic[new_name] += 1

            elif feature == 'cast':
                while counter <= 10:
                    minilist.append(new_name)
                    if new_name not in dic:
                        dic[new_name] = 1
                        counter += 1
                    else:
                        dic[new_name] += 1
                        counter += 1
            elif feature == 'production_companies':
                while counter <= 2:
                    minilist.append(new_name)
                    if new_name not in dic:
                        dic[new_name] = 1
                        counter += 1
                    else:
                        dic[new_name] += 1
                        counter += 1
            else:
                minilist.append(new_name)
                if new_name not in dic:
                    dic[new_name] = 1
                else:
                    dic[new_name] += 1

        li.append(minilist)

    return dic, li


def top(dic):
    n_largest = nlargest(15, dic, key=dic.get)
    assert isinstance(n_largest, list)
    return n_largest


genr = popular('genres')
keyw = popular('keywords')
comp = popular('production_companies')
acto = popular('cast')
dito = popular('crew')

gen = top(genr[0])
key = top(keyw[0])
com = top(comp[0])
act = top(acto[0])
dit = top(dito[0])


class User:
    def __init__(self, themes, words, companies, actors, directors):
        self.themes = themes
        self.words = words
        self.companies = companies
        self.actors = actors
        self.directors = directors

    @staticmethod
    def user_input():
        x = input("Your Answer: ")
        print()
        return x

    @staticmethod
    def display(alist, item):
        print("Please select a {} by typing corresponding number.".format(item))
        for i in range(len(alist)):
            print("{}\t{}".format(i, alist[i]))

        if item != 'genre':
            print("If you want to skip, please type (s)")

    def integer_check(self, command, li, skipable=False):
        if not skipable:
            while not isinstance(command, int) and command not in range(len(li)):
                try:
                    command = int(command)

                except ValueError:
                    print("Invalid command. Try again!")
                    command = self.user_input()

            return command

        if skipable:
            while not isinstance(command, int) and command not in range(len(li)):
                if command == 's':
                    return None
                try:
                    command = int(command)

                except ValueError:
                    print("Invalid command. Try again!")
                    command = self.user_input()

        return command

    def extract(self, ui, alist):
        ind = None
        if ui == 'q':
            sys.exit(2)

        elif alist == self.themes:
            ind = self.integer_check(ui, self.themes)

        elif alist != self.themes:
            ind = self.integer_check(ui, alist, skipable=True)

        if ind:
            return alist[ind]
        else:
            return None

    def start(self):
        print("Welcome to Sujoung's movie recommender.")
        print("If you want to quit, press (q) button.")
        print()

        print("What is your name?")
        self.name = self.user_input()
        if self.name != 'q':
            print("Hello, {}!".format(self.name))
        elif self.name == 'q':
            sys.exit(2)

        self.display(self.themes, 'genre')
        u_theme = self.user_input()
        theme_ = self.extract(u_theme, self.themes)

        self.display(self.words, 'keyword')
        u_word = self.user_input()
        word_ = self.extract(u_word, self.words)

        self.display(self.companies, 'production company')
        u_comp = self.user_input()
        comp_ = self.extract(u_comp, self.companies)

        self.display(self.actors, 'actor/actress')
        u_act = self.user_input()
        act_ = self.extract(u_act, self.actors)

        self.display(self.directors, 'director')
        u_dire = self.user_input()
        dire_ = self.extract(u_dire, self.directors)

        self.word_vector = [theme_, word_, comp_, act_, dire_]
        print(self.word_vector)

class Model:
    def __init__(self,user_vector,training_set):
        self.uv = user_vector
        self.train = training_set
        """
        ['budget', 'genres',  'id', 'keywords', 
        'original_language', 'original_title', 'overview', 
        'popularity','production_companies', 
        'production_countries', 'release_date', 
        'revenue', 'runtime', 'tagline', 
        'vote_average', 'cast', 'crew']
        """

class WordEmbeddings:
    def __init__(self,training_set):
        self.train = training_set

    def f_genres(self):
        #Take first 3 genres (param = 3)
        genre_data = genr[1]
        genre_model = Word2Vec(genre_data, size=10, workers=4)

    def f_keywords(self):
        #Take all
        keywords_data = keyw[1]
        keywords_model = Word2Vec(keywords_data, size=10, workers=4)

    def overview(self):
        #Text normalization and select top 30 words with tf-idf
        #Non available data -> mean of other values
        pass

    def pro_comp(self):
        pass

    def pro_cont(self):
        pass

    def tagline(self):
        pass

    def cast(self):
        pass

    def crew(self):
        pass

    def load(self):
        try:
            self.genres = Word2Vec.load('genres.bin')
            self.keywords = Word2Vec.load('keywords.bin')
            self.pro_comp = Word2Vec.load('pro_comp.bin')
            self.pro_cont = Word2Vec.load('pro_cont.bin')
            self.tagline = Word2Vec.load('tagline.bin')
            self.cast = Word2Vec.load('cast.bin')
            self.crew = Word2Vec.load('crew.bin')
        except:
            print("models are not ready.")
        # 1)
        # first, make a list of whole words
        # and then do word_embeddings



if __name__ == "__main__":
    X = User(gen, key, com, act, dit)
    X.start()
    Y = Model(X.word_vector,data)

# preprocess the data
# -- depending on important(frequent) words,
#    select 20 different examples as test set.
# -- tokenization of overview, title, tagline

# -- select word embeddings method, clustering function

# form a vectorspace with different word embeddings
# -- set a data shape
# -- create function with sklean.tfidfVectorizer
# -- create function with countvector
# -- create function with continuous bag of words
# -- make a function that makes dummy
# -- set a function for making vector space

# run the kmeans clustering
# -- create two functions(1. sklearn kmeans, 2.sklearn minibatch kmeans, 3.tensorflow)
# -- input vectors
# -- run the kmeans clustering with cluster = 10(not fixed)
# -- map the data by group
# -- present the group with title
# -- Input test examples and present the title by group

# Evaluation ( asking people, analysis etc.)
# -- Hand made evaluation
# -- ask people
