import numpy as np
import pandas as pd
import math
import operator
import json
import sys
import string
import random
import re
import time
import warnings
import pickle
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn import metrics
from nltk import word_tokenize
from nltk.tokenize import sent_tokenize
from sklearn.cluster import KMeans
from sklearn.metrics import precision_recall_fscore_support
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from heapq import nlargest
from sklearn.decomposition import PCA
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

index_id_dict = {x:y for x,y in enumerate(data['id'])} #(x = index : y = movie id(tmbd))
id_index_dict = {y:x for x,y in enumerate(data['id'])} #(y = movie id : x = index)


# collect important words

def fxn():
    warnings.warn("deprecated", FutureWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()


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


def top(dic,n):
    n_largest = nlargest(n, dic, key=dic.get)
    assert isinstance(n_largest, list)
    return n_largest

genr = popular('genres')
keyw = popular('keywords')
comp = popular('production_companies')
#cont = popular('production_countries')
acto = popular('cast')
dito = popular('crew')


class Preprocess:
    def __init__(self,dataset):
        self.data = dataset[:100]
        self.stemmer = SnowballStemmer("english")

    def token_stem(self,attribute):
        stopw = stopwords.words("english")
        stopp = list(string.punctuation)
        stop = stopw + stopp
        # return 4803 of vectors
        overview_data = self.data[attribute]
        filtered_data = []
        for example in overview_data:
            temp = []
            if isinstance(example,str):
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

    def get_tfidf(self, list_of_stems):
        v = TfidfVectorizer(min_df=1)
        idf = v.fit_transform(list_of_stems)
        tfidf = idf.toarray()
        return tfidf

    def elbow(self,matrix):
        title_dic = {a:b for a,b in enumerate(self.data['id'])}
        elbow = KElbowVisualizer(KMeans(), k=10)
        elbow.fit(matrix)
        elbow.poof()

    def silhouette(self,matrix,k):
        model_kmeans = KMeans(n_clusters=k, max_iter=200)
        silhouette = SilhouetteVisualizer(model_kmeans)
        silhouette.fit(matrix)
        silhouette.poof()

    def clustering(self,matrix,k):
        dic = {}
        #model_kmeans = MiniBatchKMeans(n_clusters=k).fit(matrix)
        model_kmeans = KMeans(n_clusters=k).fit(matrix)
        kmeans = list(model_kmeans.labels_)
        for i in range(len(kmeans)):
           if kmeans[i] not in dic:
               dic[kmeans[i]] = [i]
           else:
               dic[kmeans[i]] += [i]
        return dic

    def cluster_analysis(self,cluster_dic,k_cluster):
        tl = {} #title
        kd = {} #keywords
        gd = {} #genres
        dd = {} #directors
        cd = {} #companies
        ad = {} #actors

        for key,value in cluster_dic.items():
            temp_t = [] #title
            temp_k = {} #keywords
            temp_g = {} #genres
            temp_d = {} #directors
            temp_c = {} #companies
            temp_a = {} #actors
            #print("group", key)
            #print("title\tkeyword\tgenre\tdirector\tcompany\tactor")
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
            genres = [genre for v in dic.values() for genre in v.keys()]
            occurs = {g: genres.count(g) for g in set(genres)}
            return occurs

        def proportion(dic):
            n_of_doc = count_doc(dic)
            res = {}
            for k,v in dic.items():
                sub = []
                total = sum(v.values())
                for item, freq in v.items():
                    tf = freq/total
                    idf = math.log(k_cluster/n_of_doc[item])
                    sub.append((item,round(tf*idf,4)))

                sub.sort(key=lambda tup:tup[1], reverse=True)
                res[k] = sub

            return res

        def display(dic):
            for k,v in dic.items():
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
        overview_matrix = self.get_tfidf(overview_stem)
        #self.elbow(overview_matrix)
        #self.silhouette(overview_matrix,6)
        clusters = self.clustering(overview_matrix,9)
        #self.cluster_analysis(clusters,9)  # just to display a result of analysis
        return clusters


class User:
    def __init__(self, data, clusters):
        self.short_data = data[:100]
        self.data = data
        self.clusters = clusters
        self.like = []
        self.not_watched = []
        self.dislike = []

    def answer(self):
        return input(">>> ")

    def extract(self):
        res = {}
        for group,movies in self.clusters.items():
            try:
                res[group] = random.sample(movies,2)
            except ValueError:
                res[group] = random.sample(movies,1)
        return res

    def fetch(self):
        extracted_dic = self.extract()
        res = {}
        for key,value in extracted_dic.items():
            for i in value:
                row = self.short_data.iloc[i]
                information = (row['title'],row['release_date'],row['tagline'],row['overview'])
                res[i] = information
        return res

    def eval(self,string,mov_num):
        if string == 'q':
            sys.exit(2)

        elif string == 'l':
            self.like.append(mov_num)

        elif string == 'n':
            self.not_watched.append(mov_num)

        elif string == 'd':
            self.dislike.append(mov_num)

        elif string == 's':
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
            self.eval(self.answer(),mov_num)

        else:
            print("Invalid command. Try agian!")
            self.eval(self.answer(),mov_num)

    def user_query(self):
        return {'liked':self.like, 'disliked': self.dislike, 'not': self.not_watched}

    def play(self):
        self.base = self.fetch()
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
        for i,(title,date,tagline,desc) in self.base.items():
            print("Have you watched <{0} ({1})> and liked it ?".format(title,date[:4]))
            self.eval(self.answer(),i)
            print()
        print("System will analyze your preference and recommend other movies...")
        time.sleep(4)
        print()
        print("This is my recommendation.")

class Model:
    def __init__(self,user_query):
        self.user_query = user_query
        self.g_model = Doc2Vec.load('genre.model')
        self.k_model = Doc2Vec.load('keyword.model')
        self.d_model = Doc2Vec.load('director.model')
        self.c_model = Doc2Vec.load('company.model')
        self.a_model = Doc2Vec.load('actor.model')
        self.t_model = Doc2Vec.load('title.model')
        self.l_model = Doc2Vec.load('tagline.model')

    def because_you_liked(self,preference):
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
            list_of_list = sim_genre, sim_keyword, sim_director, sim_company, \
                           sim_actor, sim_title, sim_tagline
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

    def put_together(self,byl_dict):
        res = {}
        for query, recoms in byl_dict.items():
            for recom, sim in recoms.items():
                if recom not in res:
                    res[recom] = sim
                else:
                    if res[recom] < sim:
                        res[recom] = sim

        return res

    def subtract(self,subtract_from,subtract_to):
        #for example: self.liked and self.disliked
        really_liked = {}
        for query, recoms in subtract_from.items():
            temp = {}
            for recom, sim in recoms.items():
                if recom not in subtract_to:
                    temp[recom] = sim
            really_liked[query] = temp
        return really_liked

    def filter_real(self,real_thing,refer):
        filtered_really_liked = {}
        for query, recoms in real_thing.items():
            temp = {}
            for recom, sim in recoms.items():
                if sim >= refer[recom]:
                    temp[recom] = sim
            filtered_really_liked[query] = temp
        return filtered_really_liked

    def collect_topN(self, semi_final, n=50):
        res = {}
        for k1, v1 in semi_final.items():
            li = [(k2, v2) for k2, v2 in v1.items()]
            li.sort(key=operator.itemgetter(1), reverse=True)
            di = dict(li)
            if len(li) >= n:
                di = dict(li[:n])
            res[k1] = di
        return res

    def main(self):
        self.liked = self.because_you_liked(self.user_query['liked'])
        self.put_liked = self.put_together(self.liked)
        self.disliked = list(self.put_together(self.because_you_liked
                                               (self.user_query['disliked'])))
        really_liked = self.subtract(self.liked, self.disliked)
        really_liked = self.subtract(really_liked, self.user_query['liked'])
        really_liked = self.subtract(really_liked, self.user_query['disliked'])
        self.really_liked = really_liked
        self.filtered_recom = self.filter_real(self.really_liked,self.put_liked)
        self.filtered_n_largest = self.collect_topN(self.filtered_recom)
        self.display(self.filtered_n_largest)
        return self.filtered_n_largest

    def display(self,final):
        for k,v in final.items():
            movie_data = data.iloc[k]
            temp = sorted(v, key=v.get, reverse=True)
            print()
            print("Because you liked <{} ({})>, maybe you also like: ".
                  format(movie_data.title, movie_data.release_date[:4]))
            alist = []
            for i in temp:
                recom_data = data.iloc[int(i)]
                text = "{} ({})".format(recom_data.title, recom_data.release_date[:4])
                alist.append(text)
            print(', '.join(alist))


# {index number in rating matrix : movieID}
# converting process : index numb -> movieID -> tmdbID -> original data's index
# self.avail_movie_id -> self.m_tmdb_dict -> id_ind_dict(global)

# from original data's index to np matrix index number
# index_id dict <- self.m_tmdb_dict_inv <- self.avai_movie_id_inv
class GetRankedRelevance:
    def __init__(self, user_query):
        self.liked = user_query['liked']
        self.disliked = user_query['disliked']
        self.rating_matrix = pickle.load(open('estimated_ratings.pkl', 'rb'))
        avail_user_id = pickle.load(open('user_id.pkl', 'rb'))
        self.avail_user_id = dict(enumerate(avail_user_id))
        self.avail_movie_id_inv = pickle.load(open('movie_eval_inverse.pkl','rb'))
        self.no_rating_data = np.where(~self.rating_matrix.any(axis=0))[0]
        self.m_tmdb_dict_inv = pickle.load(open('tmdb2movielens.pkl','rb'))

    def nan_data(self):
        return self.no_rating_data

    def org2molenind(self,string):
        tmdb_id = index_id_dict[string]
        movie_id = self.m_tmdb_dict_inv[tmdb_id]
        matrix_ind = self.avail_movie_id_inv[movie_id]
        return matrix_ind

    def find_interested_users(self,query,n=50):
        adict = {}
        arr = self.rating_matrix[:,query]
        sim_users = arr.argsort()[-n:][::-1]
        for user in sim_users:
            rate = arr[user]
            adict[user] = rate
        return adict

    def users_mov_ratings(self,users_dict,n=100):
        sim_movie_dict_per_user = {}
        for user in users_dict.keys():
            temp = {}
            arr = self.rating_matrix[user]
            rel_movies = arr.argsort()[-n:][::-1]
            for mov in rel_movies:
                rate = arr[mov]
                temp[mov] = rate
            sim_movie_dict_per_user[user] = temp
        return sim_movie_dict_per_user

    def weighting(self,users_dict):
        weight_dict = {}
        for user,rating in users_dict.items():
            weight_dict[user] = rating/np.sum(list(users_dict.values()))
        return weight_dict

    def weighted_ratings(self, weight_vector, sim_users_sim_movs):
        # convert the original ratings to weighted ratings
        weighted_dict = {}
        for user, sim_dict in sim_users_sim_movs.items():
            temp = {}
            w = weight_vector[user]
            for sim_mov, sim_rating in sim_dict.items():
                temp[sim_mov] = sim_rating * w
            weighted_dict[user] = temp
        return weighted_dict

    def total_weighted_ratings(self,wr):
        # wr = result from weighted rating
        res = {}
        for k1,v1 in wr.items():
            for k2,v2 in v1.items():
                for mov,rat in v2.items():
                    if mov not in res:
                        res[mov] = rat
                    elif rat > res[mov]:
                        res[mov] = rat
        return res


    def set_ranking(self,total_ranked):
        li = [(k, v) for k, v in total_ranked.items()]
        li.sort(key=operator.itemgetter(1), reverse=True)
        return li

    def processing(self,converted_preferece):
        # iterate liked movies and get sim movies with weighted ratings
        adict = {}
        for mov in converted_preferece:
            if mov not in self.no_rating_data:
                sim_users_per_movie = self.find_interested_users(mov,2)
                relevant_movies = self.users_mov_ratings(sim_users_per_movie)
                weights = self.weighting(sim_users_per_movie)
                weighted_matrix = self.weighted_ratings(weights, relevant_movies)
                adict[mov] = weighted_matrix
        return adict

    def converting_org_mov(self, li):
        alist = []
        for i in li:
            try:
                ind = self.org2molenind(i)
                alist.append(ind)
            except KeyError:
                pass
        return alist

    def main(self):
        conv_liked = self.converting_org_mov(self.liked)
        conv_disliked = self.converting_org_mov(self.disliked)
        sim_liked = self.processing(conv_liked)
        #{movie: {user: {movie: sim}}}
        sim_disliked = self.processing(conv_disliked)
        # To calculate subtraction
        total_sim_liked = self.total_weighted_ratings(sim_liked)
        total_sim_disliked = self.total_weighted_ratings(sim_disliked)
        subtract_disliked = {k:v for k,v in total_sim_liked.items() if k not in total_sim_disliked}
        self.complete_rel = {k:v for k,v in subtract_disliked.items() if k not in self.liked + self.disliked}
        ranked_relevant = self.set_ranking(self.complete_rel)
        return ranked_relevant


class Evaluation:
    def __init__(self,result,ranked_rel,not_available_data):
        self.result = result
        self.relevant = ranked_rel
        self.avail_movie_id = pickle.load(open('movie_eval.pkl', 'rb'))
        self.avail_movie_id_inv = pickle.load(open('movie_eval_inverse.pkl', 'rb'))
        self.m_tmdb_dict = pickle.load(open('movielens2tmdb.pkl', 'rb'))
        self.m_tmdb_dict_inv = pickle.load(open('tmdb2movielens.pkl', 'rb'))
        self.not_available_data = not_available_data


    def movieind2org(self,string):
        movielens_id = self.avail_movie_id[string]
        #print(movielens_id)
        tmdb_id = self.m_tmdb_dict[int(movielens_id)]
        #print(tmdb_id)
        original_ind = id_index_dict[tmdb_id]
        #print(original_ind)
        return str(original_ind)

    def nan_handling(self):
        alist = []
        for i in self.not_available_data:
            alist.append(self.movieind2org(i))
        return alist

    def org2movieind(self,li):
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
        rel_labels = [ x for (x,y) in self.relevant]
        #print("original result before mapping",rel_labels)
        y_true = []
        for i in rel_labels:
            try:
                y_true.append(self.movieind2org(i))
            except KeyError:
                pass

        y_pred = self.put_together()

        trueleng, predleng = len(y_true),len(y_pred)

        true_positive = len(set(y_pred).intersection(set(y_true)))
        precision = true_positive/predleng
        recall = true_positive/trueleng
        f1score = (2 * precision * recall) / (precision + recall)
        print("precision:{}, recall:{}, f1-score:{}"
              .format(precision,recall,f1score))


if __name__ == "__main__":
    X = Preprocess(data)
    clusters = X.load()
    Y = User(data,clusters)
    Y.play()
    user = Y.user_query()
    Z = Model(user)
    res = Z.main()
    G = GetRankedRelevance(user)
    rel = G.main()
    E = Evaluation(res,rel,G.nan_data())
    E.make_eval_data()


