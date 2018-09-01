import pickle
from train_kmeans import stop_set, multiple_punc
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk import word_tokenize
from sujoungs_recommender import data


def clean_strings(li):
    result = []
    for string in li:
        temp1 = multiple_punc.sub(' ', string)
        temp2 = [y for y in word_tokenize(temp1) if y not in stop_set]
        temp3 = ' '.join(temp2)
        result.append(str(temp3))
    return result

try:
    genres = pickle.load(open('genres.pkl','rb'))
    keywords = pickle.load(open('keywords.pkl','rb'))
    overview = pickle.load(open('overview.pkl','rb'))
    title = pickle.load(open('title.pkl','rb'))
    tagline = pickle.load(open('tagline.pkl','rb'))
    cast = pickle.load(open('cast.pkl','rb'))
    crew = pickle.load(open('crew.pkl','rb'))
    production_companies = pickle.load(open('production_companies.pkl','rb'))

except FileNotFoundError:
    genres = list(data['genres'].dropna(axis=0))
    keywords = list(data['keywords'].dropna(axis=0))
    overview = list(data['overview'].dropna(axis=0))
    title = list(data['title'].dropna(axis=0))
    tagline = list(data['tagline'].dropna(axis=0))
    cast = list(data['cast'].dropna(axis=0))
    crew = list(data['crew'].dropna(axis=0))
    production_companies = list(data['production_companies'].dropna(axis=0))  # filter out Nan value(numpy nan)

    print("Cleaning text")
    genres = clean_strings(genres)
    print(" task 1 out of 8 done")
    keywords = clean_strings(keywords)
    print(" task 2 out of 8 done")
    overview = clean_strings(overview)
    print(" task 3 out of 8 done")
    title = clean_strings(title)
    print(" task 4 out of 8 done")
    tagline = clean_strings(tagline)
    print(" task 5 out of 8 done")
    cast = clean_strings(cast)
    print(" task 6 out of 8 done")
    crew = clean_strings(crew)
    print(" task 7 out of 8 done")

    cleaned_text_list = {"genres": genres, "keywords": keywords, "overview": overview,
                         "title": title, "tagline": tagline, "cast": cast, "crew": crew,
                         "production_companies": production_companies}

    for k, v in cleaned_text_list.items():
        pickle.dump(v, open(k+".pkl", 'wb'))
        print("{} variable saved".format(k))

# from Deepak Mishra's article at Medium
# https://medium.com/@mishra.thedeepak/doc2vec-simple-implementation-example-df2afbbfbad5


class Train:
    def __init__(self, li, modelname):
        self.li = li
        self.filename = modelname

    def main(self):
        base = self.li
        tagged_data = [TaggedDocument(doc, [str(i)]) for i, doc in enumerate(base)]
        max_epochs = 200
        vec_size = 20
        alpha = 0.075
        model = Doc2Vec(vector_size=vec_size,
                        alpha=alpha,
                        min_alpha=0.00025,
                        dm=0)  # dm=0 ; PV-DBOW, dm=1 : PV-DM
        model.build_vocab(tagged_data)
        for epoch in range(max_epochs):
            print("Epoch {} out of {}".format(epoch, max_epochs))
            model.train(tagged_data,
                        total_examples=model.corpus_count,
                        epochs=model.iter)
            model.alpha -= 0.0002
            model.min_alpha = model.alpha
        model.save(self.filename+'.model')


Genres = Train(genres, 'genres')
Keywords = Train(keywords, 'keywords')
Overview = Train(overview, 'overview')
Title = Train(title, 'title')
Tagline = Train(tagline, 'tagline')
Cast = Train(cast, 'cast')
Crew = Train(crew, 'crew')
ProductionCompanies = Train(production_companies, 'production_companies')

print("Start saving feature information")
Genres.main()
print("task 1 out of 8 done")
Keywords.main()
print("task 2 out of 8 done")
Overview.main()
print("task 3 out of 8 done")
Title.main()
print("task 4 out of 8 done")
Tagline.main()
print("task 5 out of 8 done")
Cast.main()
print("task 6 out of 8 done")
Crew.main()
print("task 7 out of 8 done")
ProductionCompanies.main()
print("task 8 out of 8 done")
print("All task finished")
