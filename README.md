# Movie Recommender system

language : python
source : tmdb 5000 database from kaggle, movielens data(ml-latest-small)

## Q. How to run?
A.
1) Download all data in a folder
2) Open terminal
3) cd YOUR PATH
------------------------------------------------------------
------If you downloaded only .csv and .py files ------------
------------------------------------------------------------
4) python3 doc2vec_features.py
5) python3 train_kmeans.py
6) python3 make_dummy_eval.py
7) python3 fill_dummpy.py
* It takes quite long time, prepare a movie and watch it...
------------------------------------------------------------

------------------------------------------------------------
-----If you already have everything (.model, .pkl)----------
------------------------------------------------------------
8) python3 sujoungs_recommender.py
-------------------------------------------------------------

## Make sure that you have .csv files.
(tmdb_5000_movies.csv, tmdb_5000_credits.csv ratings.csv, links.csv)

## Packages used:
* numpy
* pandas
* json
* gensim
* scikit-learn
* nltk
* heapq
* yellowbrick
