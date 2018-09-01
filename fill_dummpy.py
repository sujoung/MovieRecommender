import numpy as np
import pickle
import heapq

dummy = pickle.load(open('dummy_eval.pkl', 'rb'))
id_ = pickle.load(open('user_id.pkl', 'rb'))
index2id = dict(enumerate(id_))

# From Waylon Flinn's answer: https://stackoverflow.com/a/20687984/8626681
# base similarity matrix (all dot products)
# replace this with dummy.dot(dummy.T).toarray() for sparse representation
similarity = np.dot(dummy, dummy.T)

# squared magnitude of preference vectors (number of occurrences)
square_mag = np.diag(similarity)

# inverse squared magnitude
inv_square_mag = 1 / square_mag

# if it doesn't occur, set it's inverse magnitude to zero (instead of inf)
inv_square_mag[np.isinf(inv_square_mag)] = 0

# inverse of the magnitude
inv_mag = np.sqrt(inv_square_mag)

# cosine similarity (elementwise multiply by inverse magnitudes)
cosine = similarity * inv_mag
cosine = cosine.T * inv_mag


# Return the index of top N similarities
def top_n(array, n):
    return heapq.nlargest(n, range(len(array)), array.take)


def similar_user(cosine_matrix, n):
    dic = {}
    for i in range(len(cosine_matrix)):
        temp = top_n(cosine_matrix[i], n+1)
        temp.pop(0)
        dic[i] = temp
    return dic


user_comp = similar_user(cosine, 60)
pickle.dump(user_comp, open('user.pkl', 'wb'))


def estimation(dummy_matrix, comp, cosim_mat):
    for user in range(len(dummy_matrix)):  # type: int
        for mov_ind in range(len(dummy_matrix[user])):
            if dummy_matrix[user, mov_ind] == 0:
                sim_list = comp[user]
                sim_rating_list = []
                for sim_user in sim_list:
                    sim_rating = dummy_matrix[sim_user, mov_ind]
                    if sim_rating:
                        sim_rating_list.append(
                            (sim_rating, cosim_mat[user, sim_user]))
                if sim_rating_list:
                    base = sum([sim for (rating, sim) in sim_rating_list])
                    main = sum([rating*sim for (rating, sim) in sim_rating_list])

                    if base > 0 and main > 0:
                        est_rating = main/base
                        dummy_matrix[user, mov_ind] = est_rating
    return dummy_matrix


complete_matrix = estimation(dummy, user_comp, cosine)
pickle.dump(complete_matrix, open('estimated_ratings.pkl', 'wb'))
