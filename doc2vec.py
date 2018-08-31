from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import kmeans as k

genr, keyw, comp = k.genr, k.keyw, k.comp
acto, dito = k.acto, k.dito
data = k.data
title = list(data['title'])
tag = list(data['tagline'].dropna(axis=0))  # filter out Nan value(numpy nan)


class Train:
    def __init__(self, li, modelname):
        self.li = li
        self.filename = modelname

    def convert2normalstr(self):
        res = []
        for i in self.li:
            res.append(' '.join(str(i).lower()))
        return res

    def main(self):
        if any(isinstance(i, list) for i in self.li):
            base = self.convert2normalstr()
        else:
            base = self.li
        tagged_data = [TaggedDocument(doc, [str(i)]) for i, doc in enumerate(base)]
        max_epochs = 100
        vec_size = 20
        alpha = 0.025
        model = Doc2Vec(vector_size=vec_size,
                        alpha=alpha,
                        min_alpha=0.00025,
                        dm=0)  # dm=0 ; PV-DBOW, dm=1 : PV-DM
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
T = Train(title, 'title')
L = Train(tag, 'tagline')

print("Start saving feature information")
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
