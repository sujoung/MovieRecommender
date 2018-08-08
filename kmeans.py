import numpy as np
import pandas as pd
import json
import sys
import string
import re
from gensim.models import Word2Vec
from nltk import word_tokenize
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
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
data.drop('original_title', axis=1, inplace=True)
data.drop('vote_count', axis=1, inplace=True)


# collect important words
def popular(feature):
    dic = {}
    li = []

    for example in data[feature]:
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
cont = popular('production_countries')
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
    def __init__(self, user_vector, dataset):
        self.uv = user_vector
        self.data = dataset
        """
        ['budget', 'genres',  'id', 'keywords', 
        'original_language', 'title', 'overview', 
        'popularity','production_companies', 
        'production_countries', 'release_date', 
        'revenue', 'runtime', 'tagline', 
        'vote_average', 'cast', 'crew']
        """


class WordEmbeddings:
    def __init__(self, dataset):
        self.data = dataset
        self.stw = stopwords.words("english")
        self.stop = self.stw + list(string.punctuation)
        self.genre = genr[1]
        self.keyword = keyw[1]
        self.company = comp[1]
        self.country = cont[1]
        self.actor = acto[1]
        self.director = dito[1]
        self.stemmer = SnowballStemmer("english")

    @staticmethod
    def get_vectors(self, feature_data, n, name):
        # get w2v model of genre,keyword,company,country,actor,director
        model = Word2Vec(feature_data, size=n, workers=4)
        model_name = name + ".bin"
        model.save(model_name)
        return model

    def stem_tokens(self,tokens, stemmer):
        alist = []
        for token in tokens:
            alist.append(stemmer.stem(token))
        return alist

    def tokenize(self,text):
        tokens = word_tokenize(text)
        stems = self.stem_tokens(tokens,self.stemmer)
        return stems

    def overview(self):
        # return 4803 of vectors
        overview_data = self.data['overview']
        filtered_data = []
        
        for example in overview_data:
            f = self.tokenize(example)
            filtered = ' '.join(f)
            filtered_data.append(filtered)
        print(filtered_data[0:10])
            
        v = TfidfVectorizer(stop_words=self.stw,
                            ngram_range=(1, 1), analyzer='word')
        
        overview_vector = v.fit_transform(filtered_data.values.astype(str))
        overview_vec_array = overview_vector.toarray()
        return overview_vec_array

    def tag_title(self, feature):
        # 'tagline','title'
        alist = []
        data_s = self.data[feature]
        for sent in data_s:
            if isinstance(sent, str):
                alist.append([o for o in word_tokenize(sent.lower()) if o not in self.stop])
        return self.get_vectors(alist, 3, feature)

    def release_date(self):
        alist = []
        self.data['release_date'] = pd.to_datetime(self.data['release_date'])
        recent_date = self.data['release_date'].max()
        for date in self.data['release_date']:
            if data:
                alist.append(pd.Timedelta(recent_date - date))
            else:
                alist.append('Nan')

    def load(self):
        try:
            self.genres = Word2Vec.load('genres.bin')
            self.keywords = Word2Vec.load('keywords.bin')
            self.pro_comp = Word2Vec.load('pro_comp.bin')
            self.pro_cont = Word2Vec.load('pro_cont.bin')
            self.cast = Word2Vec.load('cast.bin')
            self.crew = Word2Vec.load('crew.bin')
            self.tagline = Word2Vec.load('tagline.bin')
            self.title = Word2Vec.load('title.bin')

        except:
            self.get_vectors(self.genre, 10, 'genres')
            self.get_vectors(self.keyword, 10, 'keywords')
            self.get_vectors(self.company, 10, 'pro_comp')
            self.get_vectors(self.country, 10, 'pro_cont')
            self.get_vectors(self.actor, 10, 'cast')
            self.get_vectors(self.director, 10, 'crew')
            self.tag_title('tagline')
            self.tag_title('title')


class BuildStructure:
    def __init__(self, original_data, m_genre, m_keywords, m_pro_comp,
                 m_pro_cont, m_cast, m_crew, m_tagline, m_title):
        self.original_data = original_data
        self.m_genre = m_genre
        self.m_keywords = m_keywords
        self.m_pro_comp = m_pro_comp
        self.m_pro_cont = m_pro_cont
        self.m_cast = m_cast
        self.m_crew = m_crew
        self.m_tagline = m_tagline
        self.m_title = m_title
        self.datadict = {}

    def overview(self):
        pass

    def rel_date(self):
        pass

    def fill_dict(self):
        result_dict = {}
        for feature in list(self.original_data):
            result_dict['id'] = self.make_list('id')
        

    def make_list(self,feature):    
        res = []
        for i in range(len(self.original_data)):
            result_list.append(self.original_data[feature][i])
        return res
            
            
        # dict = {'id':333. d}
        # replace the data with numbers
        # fill the Nan data with mean of the rest
        # make a list including dummy value (np.nan)
        # put the list in dictionary mapping with id



if __name__ == "__main__":
    X = User(gen, key, com, act, dit)
    X.start()
    Z = WordEmbeddings(data)
    Z.load()
    print()
    B = BuildStructure(data, Z.genres, Z.keywords, Z.pro_comp, Z.pro_cont,
                       Z.cast, Z.crew, Z.tagline, Z.title)
