import pandas as pd
import numpy as np
import pickle
import sujoungs_recommender as mainfile

data = mainfile.data
links = pd.read_csv("links.csv")
mapping = dict(zip(links.movieId, links.tmdbId))
mapping2 = dict(zip(links.tmdbId, links.movieId))
movieIdSeries = links['movieId']
original = pd.read_csv("ratings.csv")
ratings = original.drop(columns=['timestamp'])
ratings = ratings.set_index(['userId', 'movieId']).rating.unstack().\
    reindex(columns=movieIdSeries). \
    stack(dropna=False). \
    reset_index(name='rating')
ratings['tmdbId'] = ratings['movieId'].map(mapping)
ratings = ratings[ratings['tmdbId'].isin(list(data['id']))]
idList = set(ratings.userId)

print("Creating matrix finished")

normalized_ratings = {}
for i in idList:
    get_rating = ratings[ratings.userId == i]['rating']
    get_mean = np.mean(get_rating)
    get_std = np.std(get_rating)
    if str(get_mean) != 'nan':
        nr1 = ((get_rating - get_mean) / get_std) + 5
        nr2 = nr1.to_dict()
        normalized_ratings.update(nr2)

print("Normailization finished")

new_dict = ratings.to_dict()
new_dict['rating'] = normalized_ratings
ratings = pd.DataFrame(new_dict)


ratings = ratings.fillna(0)
movie_lookup = list(ratings[ratings['userId'] == 1]['movieId'])
movie_lookup = dict(enumerate(movie_lookup))
movie_lookup2 = {v: k for k, v in movie_lookup.items()}
vectordict = {}

print("Overwrite dataframe finished")

for i in idList:
    vector = list(ratings[ratings['userId'] == i]['rating'])
    vectordict[i] = vector
    print("User Id {} information collected".format(i))

user_id = []
sparse_ratings = []

for k, v in vectordict.items():
    if np.sum(v) > 0:
        user_id.append(k)
        sparse_ratings.append(v)

print("ready to pickle variables")

pickle.dump(user_id, open('user_id.pkl', 'wb'))
pickle.dump(np.array(sparse_ratings), open("dummy_eval.pkl", "wb"))
pickle.dump(movie_lookup, open('movie_eval.pkl', 'wb'))
pickle.dump(movie_lookup2, open('movie_eval_inverse.pkl', 'wb'))
pickle.dump(mapping, open('movielens2tmdb.pkl', 'wb'))
pickle.dump(mapping2, open('tmdb2movielens.pkl', 'wb'))

print("All tasks done!")
