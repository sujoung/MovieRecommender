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
from nltk import word_tokenize
from nltk.tokenize import sent_tokenize
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from heapq import nlargest
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer

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

index_id_dict = {x: y for x, y in enumerate(data['id'])}  # (x = index : y = movie id(tmbd))
id_index_dict = {y: x for x, y in enumerate(data['id'])}  # (y = movie id : x = index)


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
                    while counter < 3:
                        if new_name not in dic:
                            minilist.append(new_name)
                            dic[new_name] = 1
                            counter += 1
                        else:
                            dic[new_name] += 1
                            counter += 1
                else:
                    while counter < 2:
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
    n_largest = nlargest(n, dic, key=dic.get)
    assert isinstance(n_largest, list)
    return n_largest


genr = popular('genres')
keyw = popular('keywords')
comp = popular('production_companies')
acto = popular('cast')
dito = popular('crew')


def get_tfidf(list_of_stems):
    v = TfidfVectorizer(min_df=1)
    idf = v.fit_transform(list_of_stems)
    tfidf = idf.toarray()
    return tfidf


class PreProcess:
    def __init__(self, dataset):
        self.data = dataset[:100]
        self.stemmer = SnowballStemmer("english")

    def token_stem(self, attribute):
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
        elbow = KElbowVisualizer(KMeans(), k=10)
        elbow.fit(matrix)
        elbow.poof()

    @staticmethod
    def silhouette(matrix, k):
        model_kmeans = KMeans(n_clusters=k, max_iter=200)
        silhouette = SilhouetteVisualizer(model_kmeans)
        silhouette.fit(matrix)
        silhouette.poof()

    @staticmethod
    def clustering(matrix, k):
        dic = {}
        # model_kmeans = MiniBatchKMeans(n_clusters=k).fit(matrix)
        model_kmeans = KMeans(n_clusters=k).fit(matrix)
        kmeans = list(model_kmeans.labels_)
        for i in range(len(kmeans)):
            if kmeans[i] not in dic:
                dic[kmeans[i]] = [i]
            else:
                dic[kmeans[i]] += [i]
        return dic

    @staticmethod
    def cluster_analysis(self, cluster_dic, k_cluster):
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
            genres = [g for v in dic.values() for g in v.keys()]
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
        # self.cluster_analysis(clusters,9)  # just to display a result of analysis
        return c


class User:
    def __init__(self, d, cluster_):
        self.short_data = d[:100]
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
        result = {}
        for group, movies in self.clusters.items():
            try:
                result[group] = random.sample(movies, 2)
            except ValueError:
                result[group] = random.sample(movies, 1)
        return result

    @property
    def fetch(self):
        extracted_dic = self.extract()
        result = {}
        for key, value in extracted_dic.items():
            for i in value:
                row = self.short_data.iloc[i]
                information = (row['title'], row['release_date'], row['tagline'], row['overview'])
                result[i] = information
        return result

    def eval(self, s, mov_num):
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
        print("System will analyze your preference and recommend other movies...")
        time.sleep(4)
        print()
        print("This is my recommendation.")


class Model:
    def __init__(self, user_query):
        self.user_query = user_query
        self.g_model = Doc2Vec.load('genre.model')
        self.k_model = Doc2Vec.load('keyword.model')
        self.d_model = Doc2Vec.load('director.model')
        self.c_model = Doc2Vec.load('company.model')
        self.a_model = Doc2Vec.load('actor.model')
        self.t_model = Doc2Vec.load('title.model')
        self.l_model = Doc2Vec.load('tagline.model')

    def because_you_liked(self, preference):
        adict = {}
        for ind in preference:
            ind = int(ind)
            sim_genre = self.g_model.docvecs.most_similar(ind, topn=20)
            sim_keyword = self.k_model.docvecs.most_similar(ind, topn=20)
            sim_director = self.d_model.docvecs.most_similar(ind, topn=20)
            sim_company = self.c_model.docvecs.most_similar(ind, topn=20)
            sim_actor = self.a_model.docvecs.most_similar(ind, topn=20)
            sim_title = self.t_model.docvecs.most_similar(ind, topn=20)
            sim_tagline = self.l_model.docvecs.most_similar(ind, topn=20)
            list_of_list = sim_genre, sim_keyword, sim_director, sim_company, sim_actor, sim_title, sim_tagline
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
        # for example: self.liked and self.disliked
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
            time.sleep(5)
            print("Because you liked <{} ({})>, maybe you also like: ".
                  format(movie_data.title, movie_data.release_date[:4]))
            alist = []
            for i in temp:
                recom_data = data.iloc[int(i)]
                text = "{} ({})".format(recom_data.title, recom_data.release_date[:4])
                alist.append(text)
            print(', '.join(alist))


class GetRankedRelevance:
    def __init__(self, user_query):
        self.liked = user_query['liked']
        self.disliked = user_query['disliked']
        self.rating_matrix = pickle.load(open('estimated_ratings.pkl', 'rb'))
        avail_user_id = pickle.load(open('user_id.pkl', 'rb'))
        self.avail_user_id = dict(enumerate(avail_user_id))
        self.avail_movie_id_inv = pickle.load(open('movie_eval_inverse.pkl', 'rb'))
        self.no_rating_data = np.where(~self.rating_matrix.any(axis=0))[0]
        self.m_tmdb_dict_inv = pickle.load(open('tmdb2movielens.pkl', 'rb'))

    def nan_data(self):
        return self.no_rating_data

    def original_2_movielens_ind(self, s):
        tmdb_id = index_id_dict[s]
        movie_id = self.m_tmdb_dict_inv[tmdb_id]
        matrix_ind = self.avail_movie_id_inv[movie_id]
        return matrix_ind

    def find_interested_users(self, query, n=50):
        adict = {}
        arr = self.rating_matrix[:, query]
        sim_users = arr.argsort()[-n:][::-1]
        for u in sim_users:
            rate = arr[u]
            adict[u] = rate
        return adict

    def users_mov_ratings(self, users_dict, n=100):
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
        weight_dict = {}
        for u, rating in users_dict.items():
            weight_dict[u] = rating / np.sum(list(users_dict.values()))
        return weight_dict

    @staticmethod
    def weighted_ratings(weight_vector, sim_users_sim_movs):
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
        # wr = result from weighted rating
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
    def __init__(self, result, ranked_rel, not_available_data):
        self.result = result
        self.relevant = ranked_rel
        self.avail_movie_id = pickle.load(open('movie_eval.pkl', 'rb'))
        self.avail_movie_id_inv = pickle.load(open('movie_eval_inverse.pkl', 'rb'))
        self.m_tmdb_dict = pickle.load(open('movielens2tmdb.pkl', 'rb'))
        self.m_tmdb_dict_inv = pickle.load(open('tmdb2movielens.pkl', 'rb'))
        self.not_available_data = not_available_data

    def movielens_ind_2_original_ind(self, s):
        movielens_id = self.avail_movie_id[s]
        tmdb_id = self.m_tmdb_dict[int(movielens_id)]
        original_ind = id_index_dict[tmdb_id]
        return str(original_ind)

    def nan_handling(self):
        alist = []
        for i in self.not_available_data:
            alist.append(self.movielens_ind_2_original_ind(i))
        return alist

    def original_ind_2_movielend_ind(self, li):
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
            for recom, sim in recoms.items():
                if query not in self.nan_handling():
                    alist.append(str(recom))
        return alist

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


if __name__ == "__main__":
    X = PreProcess(data)
    clusters = X.load()
    Y = User(data, clusters)
    Y.play()
    user = Y.user_query()
    Z = Model(user)
    res = Z.main
    ask_evaluation = input("Do you want evaluation? \t type (y) or (n) \n>>>")
    if ask_evaluation == 'y':
        G = GetRankedRelevance(user)
        rel = G.main()
        E = Evaluation(res, rel, G.nan_data())
        E.make_eval_data()
    elif ask_evaluation == 'n':
        print("Thank you for using Sujoung's recommender.")
    else:
        print("Invalid command. Quit the program.")
