import numpy as np
import pandas as pd
import math
import operator
import json
import sys
import string
import random
import time
import pickle
from gensim.models.doc2vec import Doc2Vec
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.stem.snowball import SnowballStemmer
from heapq import nlargest
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer

"""
=======================================================
Dependency files should be acquired from other files.
* Order of execution *
1) doc2vec_features.py
2) train_kmeans.py
3) make_dummy_eval.py
4) fill_dummpy.py
=======================================================
"""

# import dataset
X = pd.read_csv('tmdb_5000_movies.csv')
Y = pd.read_csv('tmdb_5000_credits.csv')

# merge two dataset X and Y
data = pd.concat([X, Y.iloc[:, 2:]], axis=1)[:2000]

# delete irrelevant columns
data.drop('homepage', axis=1, inplace=True)
data.drop('spoken_languages', axis=1, inplace=True)
data.drop('status', axis=1, inplace=True)
data.drop('original_title', axis=1, inplace=True)
data.drop('vote_count', axis=1, inplace=True)

# make dictionaries used for mapping dataframe index and actual tmdb movie id
index_id_dict = {x: y for x, y in enumerate(data['id'])}  # (x = index : y = movie id(tmbd))
id_index_dict = {y: x for x, y in enumerate(data['id'])}  # (y = movie id : x = index)


def extract_attribute(feature):
    """
    This function is to extract attributes and frequency & row data in list type from each feature.
    Dictionary contains the useful attribute as key and frequency as value.
    List contains sub_list each row. for example [[Adventure, Fantasy],[Family, Drama],...]
    :param feature: original data's column (in string type)
    :return: dictionary, list
    """
    dic = {}
    li = []
    for example in data[feature]:
        diclist = json.loads(example)
        counter = 0
        minilist = []
        for i in diclist:
            new_name = i["name"].replace(" ", "_")
            # replace to underscore to consider it as one word in tf-idf matrix
            if feature == 'crew':
                if i["job"] == "Director":  # only take director's information
                    minilist.append(new_name)
                    if new_name not in dic:
                        dic[new_name] = 1
                    else:
                        dic[new_name] += 1
            else:
                minilist.append(new_name)
                if new_name not in dic:
                    dic[new_name] = 1
                else:
                    dic[new_name] += 1
        if feature == 'cast' or feature == 'production_companies':
            for i in diclist:
                new_name = i["name"].replace(" ", "_")
                if feature == 'cast':
                    while counter < 3:  # top 3 actors
                        if new_name not in dic:
                            minilist.append(new_name)
                            dic[new_name] = 1
                            counter += 1
                        else:
                            dic[new_name] += 1
                            counter += 1
                else:
                    while counter < 2:  # top 2 production companies
                        if new_name not in dic:
                            minilist.append(new_name)
                            dic[new_name] = 1
                            counter += 1
                        else:
                            dic[new_name] += 1
                            counter += 1
        li.append(minilist)
    return dic, li


def top(dic, n):
    """
    This function is to extract the most frequent attribute from each examples.
    :param dic:  dictionary extracted from 'extract_attribute' function
    :param n: for N largest
    :return: list of the attribute names which appeared most frequently (top N)
    """
    n_largest = nlargest(n, dic, key=dic.get)
    assert isinstance(n_largest, list)
    return n_largest


genr = extract_attribute('genres')
keyw = extract_attribute('keywords')
comp = extract_attribute('production_companies')
acto = extract_attribute('cast')
dito = extract_attribute('crew')


def get_tfidf(list_of_stems):
    """
    To create the matrix of tf-idf
    :param list_of_stems: stem here means the list of words which are processed with stemmer.(here: SnowballStemmer)
    :return: tf-idf matrix
    """
    v = TfidfVectorizer(min_df=1)
    idf = v.fit_transform(list_of_stems)
    tfidf = idf.toarray()
    return tfidf


class PreProcess:
    """
    This class is used to cluster the movies 'roughly' with selected information.
    (selected information here means for example, 'director' is taken among all the crews)
    It clusters the movies and also shows analysis how well it clustered using tf-idf with features.
    To see how the analysis works, see 'cluster_analysis' method.
    """
    def __init__(self, dataset):
        self.data = dataset[:100]
        self.stemmer = SnowballStemmer("english")

    def token_stem(self, attribute):
        """
        Filter out text with stop words and punctuation.
        SnowballStemmer stems all the words and return to simple stemmed words.
        This is first to make clear to consider all the same rooted inflective words as one,
        Second, to reduce the dimensionality in tf-idf matrix since it is very sparse.
        :param attribute: original data frame's column name
        :return: list of data filtered with all the stop words, punctuation, stmming tools.
        """
        stopw = stopwords.words("english")
        stopp = list(string.punctuation)
        stop = stopw + stopp
        # return 4803 of vectors
        overview_data = self.data[attribute]
        filtered_data = []
        for example in overview_data:
            temp = []
            if isinstance(example, str):
                tokens = word_tokenize(example)
            else:
                tokens = word_tokenize(str(example))
            for token in tokens:
                token = token.lower()
                if token not in stop:
                    s = self.stemmer.stem(token)
                    temp.append(s)
            sent = ' '.join(temp)
            filtered_data.append(sent)
        return filtered_data

    @staticmethod
    def elbow(matrix):
        """
        This function is not explicitly used since it helps deciding 'k' for clustering
        :param matrix: tf-idf matrix
        :return: show graph with the degree of distortion
        """
        elbow = KElbowVisualizer(KMeans(), k=10)
        elbow.fit(matrix)
        elbow.poof()

    @staticmethod
    def silhouette(matrix, k):
        """
        This function is also not explicitly used since it shows the decided 'k' is good or not.
        :param matrix: tf-idf matrix
        :param k: decided k (from elbow matrix)
        :return: show graph with all cluster's internal similarities and uniqueness with other clusters.
        """
        model_kmeans = KMeans(n_clusters=k, max_iter=200)
        silhouette = SilhouetteVisualizer(model_kmeans)
        silhouette.fit(matrix)
        silhouette.poof()

    @staticmethod
    def clustering(matrix, k):
        """
        K-means clustering only returns to list like [0, 1, 1, 0, 1, 0].
        If it is 5 examples and clustered to 2 groups.
        This returns a dictionary like {0: [0, 3, 5], 1: [1, 2, 4]}
        :param matrix: tf-idf matrix
        :param k: decided k (maybe got helped from elbow and silhouoette method)
        :return: dictionary {cluster_number : [index_numbers]}
        """
        dic = {}
        model_kmeans = KMeans(n_clusters=k).fit(matrix)
        kmeans = list(model_kmeans.labels_)
        for i in range(len(kmeans)):
            if kmeans[i] not in dic:
                dic[kmeans[i]] = [i]
            else:
                dic[kmeans[i]] += [i]
        return dic

    def cluster_analysis(self, cluster_dic, k_cluster):
        """
        It only prints the 3 frequent words in each cluster and their tf-idf(magnitude).
        There are 3 nested functions.
        1) count_doc:
        2) proportion : calculate tf-idf
        3) display : it takes top 3 to be ready for printing
        :param cluster_dic: returned from 'clustering' method
        :param k_cluster: used k in 'clustering' method
        :return: it does not return anything, it just prints the words and tf-idf
        """
        tl = {}  # title
        kd = {}  # keywords
        gd = {}  # genres
        dd = {}  # directors
        cd = {}  # companies
        ad = {}  # actors
        for key, value in cluster_dic.items():
            temp_t = []  # title
            temp_k = {}  # keywords
            temp_g = {}  # genres
            temp_d = {}  # directors
            temp_c = {}  # companies
            temp_a = {}  # actors
            for ind in value:
                temp_t.append(self.data.iloc[ind]['title'])
                for keyword in keyw[1][ind]:
                    if keyword not in temp_k:
                        temp_k[keyword] = 1
                    else:
                        temp_k[keyword] += 1
                for genre in genr[1][ind]:
                    if genre not in temp_g:
                        temp_g[genre] = 1
                    else:
                        temp_g[genre] += 1
                for director in dito[1][ind]:
                    if director not in temp_d:
                        temp_d[director] = 1
                    else:
                        temp_d[director] = 1
                for company in comp[1][ind]:
                    if company not in temp_c:
                        temp_c[company] = 1
                    else:
                        temp_c[company] += 1
                for actor in acto[1][ind]:
                    if actor not in temp_a:
                        temp_a[actor] = 1
                    else:
                        temp_a[actor] += 1
            tl[key] = temp_t
            kd[key] = temp_k
            gd[key] = temp_g
            dd[key] = temp_d
            cd[key] = temp_c
            ad[key] = temp_a

        def count_doc(dic):
            """
            vhcandido's Answer in stackoverflow.
            link: https://stackoverflow.com/a/51992198/8626681
            :param dic: {cluster_num: {attribute_name : frequency}}
            :return: dic: {attribute_name: number of occurrence from all clusters }
            """
            # Flatten the dictionary
            genres = [g for v in dic.values() for g in v.keys()]
            # Count each unique element and build a dictionary
            occurs = {g: genres.count(g) for g in set(genres)}
            return occurs

        def proportion(dic):
            n_of_doc = count_doc(dic)
            result = {}
            for k, v in dic.items():
                sub = []
                total = sum(v.values())
                for item, freq in v.items():
                    tf = freq / total
                    idf = math.log(k_cluster / n_of_doc[item])
                    sub.append((item, round(tf * idf, 4)))

                sub.sort(key=lambda tup: tup[1], reverse=True)
                result[k] = sub

            return result

        def display(dic):
            for k, v in dic.items():
                print("Group", k)
                if dic == tl:
                    print(v)
                else:
                    print(v[0:3])

        display(tl)
        print()
        new_k = proportion(kd)
        display(new_k)
        print()
        new_g = proportion(gd)
        display(new_g)
        print()
        new_d = proportion(dd)
        display(new_d)
        print()
        new_c = proportion(cd)
        display(new_c)
        print()
        new_a = proportion(ad)
        display(new_a)

    def load(self):
        overview_stem = self.token_stem('overview')
        overview_matrix = get_tfidf(overview_stem)
        # self.elbow(overview_matrix)
        # self.silhouette(overview_matrix,6)
        c = self.clustering(overview_matrix, 9)
        # self.cluster_analysis(c, 9)  # just to display a result of analysis
        return c


class User:
    def __init__(self, d, cluster_):
        self.short_data = d[:100]  # same length as used in PreProcess class
        self.data = d
        self.clusters = cluster_
        self.like = []
        self.not_watched = []
        self.dislike = []
        self.base = self.fetch

    @staticmethod
    def answer():
        return input(">>> ")

    def extract(self):
        """
        randomly select two movies from each clusters obtained from PreProcess class
        :return: dic: {cluster:movie index number}
        """
        result = {}
        for group, movies in self.clusters.items():
            try:
                result[group] = random.sample(movies, 2)
            except ValueError:
                result[group] = random.sample(movies, 1)
        return result

    @property
    def fetch(self):
        """
        This function fetches actual title name and year from original data.
        It also takes 'tagline' and 'overview' just in case user wants to see details of movies
        :return: dic: {movie_index_numer: tuple of movie information}
        """
        extracted_dic = self.extract()
        result = {}
        for key, value in extracted_dic.items():
            for i in value:
                row = self.short_data.iloc[i]
                information = (row['title'], row['release_date'], row['tagline'], row['overview'])
                result[i] = information
        return result

    def eval(self, s, mov_num):
        """
        It takes user's input and allocate it to different tasks.
        :param s: string(user's input)
        :param mov_num: movie index number
        :return: it does not return anything
        """
        if s == 'q':
            sys.exit(2)
        elif s == 'l':
            self.like.append(mov_num)
        elif s == 'n':
            self.not_watched.append(mov_num)
        elif s == 'd':
            self.dislike.append(mov_num)
        elif s == 's':
            print("tagline: ", self.base[mov_num][2])
            print()
            print("Description ... ")
            text = self.base[mov_num][3]
            sent_list = sent_tokenize(text)
            for sent in sent_list:
                print(sent)
            print()
            time.sleep(2)
            print("Did you like it? [please enter (l) OR (n) OR(d)")
            self.eval(self.answer(), mov_num)
        else:
            print("Invalid command. Try agian!")
            self.eval(self.answer(), mov_num)

    def user_query(self):
        return {'liked': self.like, 'disliked': self.dislike, 'not': self.not_watched}

    def play(self):
        print("Hello, Welcome to Sujong's movie recommender.")
        print("If you want to quit the program, please press (q) to exit.")
        print()
        print("I will show some examples of famous movies.")
        print("If you have watched AND actually liked it, please type (l) for like. ")
        print("If you have not watched, please type (n) for not watched.")
        print("If you have watched BUT actually you didn't like it, plese type (d) for dislike.")
        print("If you are not sure about movies, you can press (s) to see details.")
        time.sleep(3)
        print()
        print()
        print("start!")
        print()
        for i, (title, date, tagline, desc) in self.base.items():
            print("Have you watched <{0} ({1})> and liked it ?".format(title, date[:4]))
            self.eval(self.answer(), i)
            print()


class ModelCosineSim:
    """
    Recommender system's first model with cosine similarity (gensim.doc2vec.most_similar)
    Base data is taken from files ( .model) and they are generated from 'doc2vec_features.py'
    """
    def __init__(self, user_query):
        self.user_query = user_query
        self.genres_m = Doc2Vec.load('genres.model')
        self.keywords_m = Doc2Vec.load('keywords.model')
        self.overview_m = Doc2Vec.load('overview.model')
        self.title_m = Doc2Vec.load('title.model')
        self.tagline_m = Doc2Vec.load('tagline.model')
        self.cast_m = Doc2Vec.load('cast.model')
        self.crew_m = Doc2Vec.load('crew.model')
        self.production_m = Doc2Vec.load('production_companies.model')

    def because_you_liked(self, preference):
        """
        It makes a dictionary of similar movies with their similarity scores.
        :param preference: user's query
        :return: dic: {query: {similar movie: similarity score}}
        """
        adict = {}
        for ind in preference:
            ind = int(ind)
            genres_s = self.genres_m.docvecs.most_similar(ind, topn=20)
            keywords_s = self.keywords_m.docvecs.most_similar(ind, topn=20)
            overview_s = self.overview_m.docvecs.most_similar(ind, topn=20)
            title_s = self.title_m.docvecs.most_similar(ind, topn=20)
            tagline_s = self.tagline_m.docvecs.most_similar(ind, topn=20)
            cast_s = self.cast_m.docvecs.most_similar(ind, topn=20)
            crew_s = self.crew_m.docvecs.most_similar(ind, topn=20)
            production_s = self.production_m.docvecs.most_similar(ind, topn=20)
            list_of_list = genres_s, keywords_s, overview_s, title_s, tagline_s, cast_s, crew_s, production_s
            temp_dict = {}
            for feature in list_of_list:
                for tup in feature:
                    if tup[0] not in temp_dict:
                        temp_dict[tup[0]] = tup[1]
                    else:
                        if temp_dict[tup[0]] < tup[1]:
                            temp_dict[tup[0]] = tup[1]
            adict[ind] = temp_dict
        return adict

    @staticmethod
    def put_together(adict):
        """
        Integrate recommended movies from the dictionary obtained from 'because_you_like' method.
        It overwrites the similarity score with higher ones.
        :param adict: dictionary obtained from 'because_you_like' method.
        :return: dic: {recommended movie: (highest) cosine similarity score}
        """
        result = {}
        for query, recommendations in adict.items():
            for recommendation, sim in recommendations.items():
                if recommendation not in result:
                    result[recommendation] = sim
                else:
                    if result[recommendation] < sim:
                        result[recommendation] = sim
        return result

    @staticmethod
    def subtract(subtract_from, subtract_to):
        """
        This is to subtract watched movies from integrated movie recommendation.
        :param subtract_from: integrated recommended movies
        :param subtract_to: for example movies that user disliked
        :return: subtracted list (filtered)
        """
        really_liked = {}
        for query, recommendations in subtract_from.items():
            temp = {}
            for recommendation, sim in recommendations.items():
                if recommendation not in subtract_to:
                    temp[recommendation] = sim
            really_liked[query] = temp
        return really_liked

    @staticmethod
    def filter_real(real_thing, refer):
        """
        This is to make a dictionary of user's liked movie and recommended movies.
        Similarity score here is very important.
        For example:
        Let's say user query '0' had recommendation {'1': 0.5,'2': 0.5} and '3' = {'1': 0.3,'5': 0.5}
        The recommended movie '1' should not appear both user queries.
        In this function, it retuns a dictionary like {0: {'1': 0.5, '2: 0.5}, '3': {'5': 0.5}}
        It made the movie number '1' belong to user query '1' not '3' because user query 0's similarity score of
        movie '1' is higher (0.5) than user query 3's (0.3)
        :param real_thing: dictionary obtained from subtract {movie_index: maximum similarity score}
        :param refer: original result obtained from put_together
        :return: totally filtered dictionary without any overlapping of recommended movies
        """
        filtered_really_liked = {}
        for query, recoms in real_thing.items():
            temp = {}
            for recom, sim in recoms.items():
                if sim >= refer[recom]:
                    temp[recom] = sim
            filtered_really_liked[query] = temp
        return filtered_really_liked

    @staticmethod
    def collect_top_n(semi_final, n=50):
        """
        It taked top N the most similar movies
        :param semi_final: dict from 'filtered_real'
        :param n: top N number
        :return: same key as semi_final and movies with top N similarity scores
        """
        result = {}
        for k1, v1 in semi_final.items():
            li = [(k2, v2) for k2, v2 in v1.items()]
            li.sort(key=operator.itemgetter(1), reverse=True)
            di = dict(li)
            if len(li) >= n:
                di = dict(li[:n])
            result[k1] = di
        return result

    @property
    def main(self):
        liked = self.because_you_liked(self.user_query['liked'])
        put_liked = self.put_together(liked)
        disliked = self.because_you_liked(self.user_query['disliked'])
        put_disliked = self.put_together(disliked)
        liked_minus_disliked = self.subtract(liked, put_disliked)
        subtraction_liked = self.subtract(liked_minus_disliked, self.user_query['liked'])
        subtraction_disliked = self.subtract(subtraction_liked, self.user_query['disliked'])
        really_liked = subtraction_disliked
        filtered_recommendations = self.filter_real(really_liked, put_liked)
        filtered_n_largest = self.collect_top_n(filtered_recommendations)
        self.display(filtered_n_largest)
        return filtered_n_largest

    @staticmethod
    def display(final):
        for k, v in final.items():
            movie_data = data.iloc[k]
            temp = sorted(v, key=v.get, reverse=True)
            print()
            print()
            time.sleep(5)
            print("Because you liked <{} ({})>, maybe you also like: ".
                  format(movie_data.title, movie_data.release_date[:4]))
            print()
            alist = []
            for i in temp:
                recom_data = data.iloc[int(i)]
                text = "<{} ({})>".format(recom_data.title, recom_data.release_date[:4])
                alist.append(text)
            print(', '.join(alist))


class ModelKmeans:
    """
    This is the other recommendation model but using k-means clustering.
    The input vector is not tf-idf in this case, because this is an actual "model".
    So, it takes already-clustered information from other file ('train_kmeans.py')
    Unlike ModelCosineSim, it does not subtract similar movies of disliked queries.
    If liked query and disliked query are in the same cluster, we can not just remove all the movies there.
    So, it subtracts only watched movies.
    """
    def __init__(self, user_query):
        self.clusters = pickle.load(open('cluster_total.pkl', 'rb'))
        self.user_query = user_query

    def main(self):
        clusters_per_liked = {}
        liked_per_cluster = {}
        because_you_like = {}
        for l in self.user_query['liked']:
            clusters_per_liked[l] = self.clusters[l]
        for k, v in clusters_per_liked.items():
            if v not in liked_per_cluster:
                liked_per_cluster[v] = [k]
            else:
                liked_per_cluster[v] += [k]

        for k, v in liked_per_cluster.items():
            temp = [i for i, j in enumerate(list(self.clusters)) if j == k]
            because_you_like[tuple(v)] = temp
        result = self.subtraction(because_you_like)
        self.display(result)
        return result

    def subtraction(self, byldict):
        like = self.user_query['liked']
        dislike = self.user_query['disliked']
        watched = like + dislike
        result = {}
        for k, v in byldict.items():
            temp = []
            for r in v:
                if r not in watched:
                    temp.append(r)
            result[k] = temp
        return result

    @staticmethod
    def display(final_dict):
        result = []
        for k, v in final_dict.items():
            query_temp = []
            recom_temp = []
            for query in k:
                t1 = data.iloc[query].title
                d1 = data.iloc[query].release_date
                query_temp.append("<{} ({})>".format(t1, d1))
            for recom in v:
                t2 = data.iloc[recom].title
                d2 = data.iloc[recom].release_date
                recom_temp.append("<{} ({})>".format(t2, d2))
            result.append((query_temp, recom_temp))
        for (a, b) in result:
            byl = ', '.join(a)
            print()
            print("Because you liked {}, maybe you also like: ".format(byl))
            print()
            rec = ', '.join(b)
            print(rec)
            time.sleep(5)
            print()


class GetRankedRelevance:
    """
    This class is to prepare for the basic evaluation data.
    It takes already filled rating matrix from other file (make_dummy_eval.py and fill_dummy.py)
    Here are several types of movie index.
    1) index of original tmdb data
    2) tmdb id
    3) movieId from movielens dataset
    4) index of rating matrix obtained from fill_dummy.py
    """
    def __init__(self, user_query):
        self.liked = user_query['liked']
        self.disliked = user_query['disliked']
        self.rating_matrix = pickle.load(open('estimated_ratings.pkl', 'rb'))
        avail_user_id = pickle.load(open('user_id.pkl', 'rb'))
        self.avail_user_id = dict(enumerate(avail_user_id))
        self.avail_movie_id_inv = pickle.load(open('movie_eval_inverse.pkl', 'rb'))
        # no_rating_data indicates the movie index of data where there are no ratings available
        self.no_rating_data = np.where(~self.rating_matrix.any(axis=0))[0]
        self.m_tmdb_dict_inv = pickle.load(open('tmdb2movielens.pkl', 'rb'))

    def nan_data(self):
        return self.no_rating_data

    def original_2_movielens_ind(self, s):
        """
        mapping original data's index(1) to index of rating matrix(4)
        :param s: (1) index of original tmdb data
        :return: (4) index of rating matrix obtained from fill_dummy.py
        """
        tmdb_id = index_id_dict[s]
        movie_id = self.m_tmdb_dict_inv[tmdb_id]
        matrix_ind = self.avail_movie_id_inv[movie_id]
        return matrix_ind

    def find_interested_users(self, query, n=50):
        """
        with user's query(liked) take N users who rated high.
        :param query: user's query
        :param n: top N
        :return: dic: {user: rating}
        """
        adict = {}
        arr = self.rating_matrix[:, query]
        sim_users = arr.argsort()[-n:][::-1]
        for u in sim_users:
            rate = arr[u]
            adict[u] = rate
        return adict

    def users_mov_ratings(self, users_dict, n=100):
        """
        After finding interested users from 'find_interested_users',
        collect movies of which rating is entered high from the user.
        :param users_dict: {user; rating}
        :param n: top N movies
        :return: {user: {movie: rating}}
        """
        sim_movie_dict_per_user = {}
        for u in users_dict.keys():
            temp = {}
            arr = self.rating_matrix[u]
            rel_movies = arr.argsort()[-n:][::-1]
            for mov in rel_movies:
                rate = arr[mov]
                temp[mov] = rate
            sim_movie_dict_per_user[u] = temp
        return sim_movie_dict_per_user

    @staticmethod
    def weighting(users_dict):
        """
        from user's dictionary obtained by 'find_interested_users',
        it calculates weight (range 0-1) of each user.
        :param users_dict: dictionary obtained by 'find_interested_users'
        :return: dic: {user: weight}
        """
        weight_dict = {}
        for u, rating in users_dict.items():
            weight_dict[u] = rating / np.sum(list(users_dict.values()))
        return weight_dict

    @staticmethod
    def weighted_ratings(weight_vector, sim_users_sim_movs):
        """
        It applies weight information to similar movie's ratings.
        :param weight_vector: weight_dict from 'weighting' function
        :param sim_users_sim_movs: obtained from 'users_mov_ratings'
        :return: basically same structure as sim_users_sim_movs but different ratings (* weight)
        """
        # convert the original ratings to weighted ratings
        weighted_dict = {}
        for u, sim_dict in sim_users_sim_movs.items():
            temp = {}
            w = weight_vector[u]
            for sim_mov, sim_rating in sim_dict.items():
                temp[sim_mov] = sim_rating * w
            weighted_dict[u] = temp
        return weighted_dict

    @staticmethod
    def total_weighted_ratings(wr):
        """
        It integrates the result of weighted ratings.
        :param wr: result from weighted rating
        :return: dic: {movie: (highest) weighted rating}
        """
        result = {}
        for k1, v1 in wr.items():
            for k2, v2 in v1.items():
                for mov, rat in v2.items():
                    # mov : movie, rat: rating
                    if mov not in result:
                        result[mov] = rat
                    elif rat > result[mov]:
                        result[mov] = rat
        return result

    @staticmethod
    def set_ranking(total_ranked):
        """
        Just sorting
        :param total_ranked: total_weighted_ratings
        :return: a list of tuples reverse sorted.
        """
        li = [(k, v) for k, v in total_ranked.items()]
        li.sort(key=operator.itemgetter(1), reverse=True)
        return li

    def processing(self, converted_preferece):
        # iterate liked movies and get sim movies with weighted ratings
        adict = {}
        for mov in converted_preferece:
            if mov not in self.no_rating_data:
                sim_users_per_movie = self.find_interested_users(mov, 2)
                relevant_movies = self.users_mov_ratings(sim_users_per_movie)
                weights = self.weighting(sim_users_per_movie)
                weighted_matrix = self.weighted_ratings(weights, relevant_movies)
                adict[mov] = weighted_matrix
        return adict

    def converting_org_mov(self, li):
        """
        Mapping (4) to (1)
        :param li: user query
        :return: list of movielens' rating matrix index
        """
        alist = []
        for i in li:
            try:
                ind = self.original_2_movielens_ind(i)
                alist.append(ind)
            except KeyError:
                pass
        return alist

    def main(self):
        conv_liked = self.converting_org_mov(self.liked)
        conv_disliked = self.converting_org_mov(self.disliked)
        sim_liked = self.processing(conv_liked)
        # {movie: {user: {movie: sim}}}
        sim_disliked = self.processing(conv_disliked)
        # To calculate subtraction
        total_sim_liked = self.total_weighted_ratings(sim_liked)
        total_sim_disliked = self.total_weighted_ratings(sim_disliked)
        subtract_disliked = {k: v for k, v in total_sim_liked.items() if k not in total_sim_disliked}
        complete_rel = {k: v for k, v in subtract_disliked.items() if k not in self.liked + self.disliked}
        ranked_relevant = self.set_ranking(complete_rel)
        return ranked_relevant


class Evaluation:
    """
    In this class, it calculates precision, recall, f1-score and mean average precision.
    Just like in GetRankedRelevace, there are 4 index types.
    1) index of original tmdb data
    2) tmdb id
    3) movieId from movielens dataset
    4) index of rating matrix obtained from fill_dummy.py
    """
    def __init__(self, result, ranked_rel, not_available_data):
        self.result = result
        self.relevant = ranked_rel
        self.avail_movie_id = pickle.load(open('movie_eval.pkl', 'rb'))
        self.avail_movie_id_inv = pickle.load(open('movie_eval_inverse.pkl', 'rb'))
        self.m_tmdb_dict = pickle.load(open('movielens2tmdb.pkl', 'rb'))
        self.m_tmdb_dict_inv = pickle.load(open('tmdb2movielens.pkl', 'rb'))
        self.not_available_data = not_available_data

    def movielens_ind_2_original_ind(self, s):
        """
        just for mapping movielens rating's matrix index to original tmdb data's index
        :param s: (4)
        :return: (1)
        """
        movielens_id = self.avail_movie_id[s]
        tmdb_id = self.m_tmdb_dict[int(movielens_id)]
        original_ind = id_index_dict[tmdb_id]
        return str(original_ind)

    def nan_handling(self):
        """
        Convert not available rating data to original tmdb movie index
        :return: list of different index format
        """
        alist = []
        for i in self.not_available_data:
            alist.append(self.movielens_ind_2_original_ind(i))
        return alist

    def original_ind_2_movielend_ind(self, li):
        """
        mapping index from (1) to (4)
        :param li: list of (1) type indices
        :return: list of (4) type indices
        """
        alist = []
        for original_ind in li:
            tmdb_id = index_id_dict[original_ind]
            movielens_id = self.m_tmdb_dict_inv[tmdb_id]
            rating_ind = self.avail_movie_id_inv[movielens_id]
            alist.append(rating_ind)
        return alist

    def put_together(self):
        alist = []
        for query, recoms in self.result.items():
            # for K-means clustering model
            if not isinstance(recoms, list):
                for recom, sim in recoms.items():
                    if query not in self.nan_handling():
                        alist.append(str(recom))
            # for cosine similarity model
            else:
                for recom in recoms:
                    if not set(query).intersection(list(self.nan_handling())):
                        alist.append(str(recom))
        return alist

    @staticmethod
    def make_binary(true, pred):
        """
        To make document binary with its rank
        :param true: relevant (obtained from movielens rating)
        :param pred: retrieved (actual recommendations)
        :return: list of tuples [(rank,boolean)]
        """
        result = []
        for i in true:
            if i in pred:
                result.append(1)
            else:
                result.append(0)
        result = list(enumerate(result, 1))
        return result

    @staticmethod
    def pk(binary_true):
        """
        Triskelion's comment at https://www.kaggle.com/c/avito-prohibited-content/discussion/9584
        Calcualate precision at K
        :param binary_true: 'make_binary' method
        :return: list of P@K for all K
        """
        result = []
        retrieved = 0
        true_positive = 0
        for i in binary_true:
            retrieved += 1
            true_positive += i[1]
            result.append(true_positive/float(retrieved))
        return result

    @staticmethod
    def avg_precision(binary_true, precisions):
        """
        Triskelion's comment at https://www.kaggle.com/c/avito-prohibited-content/discussion/9584
        :param binary_true: 'make_binary' method
        :param precisions: list of P@K for all K
        :return: average precision @ all K
        """
        avg_precisions = []
        for r in range(1, len(binary_true) + 1):
            avg_precisions.append(sum(precisions[:r]) / float(r))
        return avg_precisions

    def make_eval_data(self):
        rel_labels = [x for (x, y) in self.relevant]
        y_true = []
        for i in rel_labels:
            try:
                y_true.append(self.movielens_ind_2_original_ind(i))
            except KeyError:
                pass
        y_pred = self.put_together()
        trueleng, predleng = len(y_true), len(y_pred)
        true_positive = len(set(y_pred).intersection(set(y_true)))

        precision = true_positive / predleng
        recall = true_positive / trueleng
        f1score = (2 * precision * recall) / (precision + recall)
        print()
        print("="*20, "Evaluation", "="*20)
        print("precision:{}, recall:{}, f1-score:{}"
              .format(round(precision, 7), round(recall, 7), round(f1score, 7)))

        binary_pair = self.make_binary(y_true, y_pred)
        pk_list = self.pk(binary_pair)
        apk = self.avg_precision(binary_pair, pk_list)
        mean_apk = np.mean(apk)
        print("Mean Average Precision : {} ".format(mean_apk))


if __name__ == "__main__":
    X = PreProcess(data)
    clusters = X.load()
    Y = User(data, clusters)
    Y.play()
    user = Y.user_query()

    #sample user input for evaluation
    #user = {'disliked': [35, 72, 7, 82, 3, 89, 93], 'not': [74, 48], 'liked': [98, 22, 71, 12, 15, 21, 86, 25, 95]}

    print("Choose the model (1) cosine (2) kmeans. Please type (1) or (2). ")
    ask_model = input(">>> ")
    time.sleep(1)
    print("System will analyze your preference and recommend other movies...")
    time.sleep(4)

    if ask_model == '1':
        Z = ModelCosineSim(user)
        print("This is my recommendation!")
        res = Z.main

    elif ask_model == '2':
        print("This is my recommendation!")
        Z = ModelKmeans(user)
        res = Z.main()
    else:
        print("Invalid command. Exit the program.")
        sys.exit(2)

    print()
    ask_evaluation = input("Do you want evaluation? \t Please type (y) or (n). \n>>> ")
    if ask_evaluation == 'y':
        G = GetRankedRelevance(user)
        rel = G.main()
        E = Evaluation(res, rel, G.nan_data())
        E.make_eval_data()
        print()
        print("Thank you for using Sujoung's recommender.")
    elif ask_evaluation == 'n':
        print("Thank you for using Sujoung's recommender.")
    else:
        print("Invalid command. Exit the program.")
        sys.exit(2)
