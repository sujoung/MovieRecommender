import pandas as pd
import json
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk import word_tokenize

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

genr = popular('genres')
keyw = popular('keywords')
comp = popular('production_companies')
acto = popular('cast')
dito = popular('crew')
title = list(data['title'])
tag = list(data['tagline'].dropna(axis=0)) #filter out Nan value(numpy nan)

class Train:
    def __init__(self,li,modelname):
        self.li = li
        self.filename = modelname

    def convert2normalstr(self):
        res = []
        for i in self.li:
            res.append(' '.join(str(i).lower()))
        return res

    def main(self):
        if any(isinstance(i,list) for i in self.li):
            self.data = self.convert2normalstr()
        else:
            self.data = self.li
        tagged_data = [TaggedDocument(doc,[str(i)]) for i, doc in enumerate(self.data)]
        max_epochs = 100
        vec_size = 20
        alpha = 0.025
        model = Doc2Vec(vector_size=vec_size,
                        alpha = alpha,
                        min_alpha= 0.00025,
                        dm = 0) # dm=0 ; PV-DBOW, dm=1 : PV-DM
        model.build_vocab(tagged_data)
        for epoch in range(max_epochs):
            model.train(tagged_data,
                        total_examples=model.corpus_count,
                        epochs=model.iter)
            model.alpha -= 0.0002
            model.min_alpha = model.alpha
        model.save(self.filename+'.model')

G = Train(genr[1], 'genre')
K = Train(keyw[1], 'keyword')
D = Train(dito[1], 'director')
C = Train(comp[1], 'company')
A = Train(acto[1], 'actor')
T = Train(title,'title')
L = Train(tag,'tagline')

G.main()
print("task 1 out of 7 done")
K.main()
print("task 2 out of 7 done")
D.main()
print("task 3 out of 7 done")
C.main()
print("task 4 out of 7 done")
A.main()
print("task 5 out of 7 done")
T.main()
print("task 6 out of 7 done")
L.main()
print("task 7 out of 7 done")
print("All task finished")

