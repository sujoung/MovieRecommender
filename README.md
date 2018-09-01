# Movie Recommender system

* Language : python3
* Source : tmdb 5000 database from kaggle, movielens data(ml-latest-small)
* Description : This is a movie recommender system. When you execute program, system asks you what movie you like and dislike and based on your preference, it recommends similar movies. For evaluation, actual users' ratings dataset is used. Using Pearson correlation coefficient, the missing parts of the rating data are filled.
* This is an indivisual project of Uppsala University's Information Retrieval Course.

## There are two types of models (Choose one of them)
* cosine similarity model : extract features, find N most similar movies, concatenate
* k-means clustering model : integrate feature information into one string, clustering

## There are two types of evaluation
* Precision, Recall, F1 score
* Mean average precision


## If you have only.py files

1) Please download .csv files from

* https://www.kaggle.com/tmdb/tmdb-movie-metadata - > size: 9 MB
* https://grouplens.org/datasets/movielens/latest/ - > ml-latest-small.zip (size: 1 MB)
2) Type commands below on terminal
```
python3 doc2vec_features.py
python3 train_kmeans.py
python3 make_dummy_eval.py
python3 fill_dummpy.py
```
* It takes quite long time, prepare a movie and watch it...
3) Execute the main file
```
python3 sujoungs_recommender.py
```


## If you already have everything (.model, .pkl)
simply execute the main file
```
python3 sujoungs_recommender.py
```

## Make sure that you have all the .csv files needed
* tmdb_5000_movies.csv
* tmdb_5000_credits.csv
* ratings.csv
* links.csv


## Packages used:
* numpy
* pandas
* json
* gensim
* scikit-learn
* nltk
* heapq
* yellowbrick
