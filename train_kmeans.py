import re
import pickle
import numpy as np
from string import punctuation
from sujoungs_recommender import data
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk import word_tokenize
from nltk.corpus import stopwords
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer
from sklearn.cluster import KMeans

concatenated_list = []
original_data = data.fillna(0)

stop_set = set(punctuation).union(set(stopwords.words("english")))
multiple_punc = re.compile('[{}]+'.format(punctuation))
df = []


def processing():
    for x in range(len(original_data)):
        genres = str(original_data.iloc[x].genres)
        keywords = str(original_data.iloc[x].keywords)
        overview = str(original_data.iloc[x].overview)
        title = str(original_data.iloc[x].title)
        tagline = str(original_data.iloc[x].tagline)
        cast = str(original_data.iloc[x].cast)
        crew = str(original_data.iloc[x].crew)
        production_companies = str(original_data.iloc[x].production_companies)
        total_attributes = [genres, keywords, overview, title, tagline, cast, crew, production_companies]
        temp = ' '.join(total_attributes)
        temp1 = multiple_punc.sub(' ', temp)
        temp2 = [y for y in word_tokenize(temp1) if y not in stop_set]
        temp3 = ' '.join(temp2)
        df.append(temp3)
        print("Iteration {} done!".format(x))
        pickle.dump(df, open('total_string_tmdb.pkl', 'wb'))


try:
    df = pickle.load(open('total_string_tmdb.pkl', 'rb'))
except FileNotFoundError:
    processing()


# line 46-74 from Deepak Mishra's article at Medium
# https://medium.com/@mishra.thedeepak/doc2vec-simple-implementation-example-df2afbbfbad5
def make_doc2vec(dt):
    tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(dt)]

    max_epochs = 200
    vec_size = 20
    alpha = 0.05

    model = Doc2Vec(size=vec_size,
                    alpha=alpha,
                    min_alpha=0.00025,
                    min_count=0,
                    dm=1)

    model.build_vocab(tagged_data)

    for epoch in range(max_epochs):
        print('iteration {0}'.format(epoch))
        model.train(tagged_data,
                    total_examples=model.corpus_count,
                    epochs=model.iter)
        # decrease the learning rate
        model.alpha -= 0.0002
        # fix the learning rate, no decay
        model.min_alpha = model.alpha

    model.save("data_d2v.model")
    print("Model Saved")


try:
    d2vmodel = Doc2Vec.load("data_d2v.model")
except FileNotFoundError:
    make_doc2vec(df)
    d2vmodel = Doc2Vec.load("data_d2v.model")

empty_matrix = []
for ind in range(len(data)):
    empty_matrix.append(list(d2vmodel.docvecs[ind]))

X = np.array(empty_matrix)


def elbow_method(matrix, k):
    elbow = KElbowVisualizer(KMeans(), k=k)
    elbow.fit(matrix)
    elbow.poof()


def silhouette_method(matrix, k):
    model_kmeans = KMeans(n_clusters=k, max_iter=200)
    silhouette = SilhouetteVisualizer(model_kmeans)
    silhouette.fit(matrix)
    silhouette.poof()


kmeans = KMeans(n_clusters=30, random_state=0).fit(X)
result = kmeans.labels_
pickle.dump(result, open('cluster_total.pkl', 'wb'))
