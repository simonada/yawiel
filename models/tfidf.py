import numpy as np
import collections

def raw_frequency(term, doc):
    count = 0

    if isinstance(doc, str):
        for word in doc.split():
            if term == word:
                count = count + 1

    return count


def get_most_freq_term( doc):
    doc_freq = dict([word, raw_frequency(word, doc)] for word in doc.split())
    value, count = collections.Counter(doc_freq).most_common(1)[0]

    return count


def compute_tf( term, doc):
    if (raw_frequency(term, doc) > 0):
        return (1 + np.log10(raw_frequency(term, doc))) / (1 + np.log10(get_most_freq_term(doc)))
    else:
        return 0


def compute_idf(term, N, index):
    return np.log10(N / get_docs_count(index, term))

def get_docs_count(index, term):
        return len(index[term])

def isInDoc( term, docID, index):
    if docID in index[term]:
        return True
    return False


def compute_tf_idf(term, doc, docID, N):
    if (isInDoc(term, docID) == False):
        return 0
    else:
        return compute_tf(term, doc) * compute_idf(term, N)

    ##Vectorization


def vectorize(query, index, index_values):
    vec = np.zeros((len(index)), dtype=int)

    for term in query:
        ind = index_values.index(term)
        vec[ind] = 1
    return vec.tolist()

    ##Vector normalization


def get_euclidean_norm( vector):
    vector_sum = np.sum(np.square(vector))
    return np.sqrt(vector_sum)


def normalize_vector(vector):
    euclid_norm = get_euclidean_norm(vector)
    vec = np.array(vector)
    if euclid_norm == 0:
        euclid_norm = 1
    normalized_vector = vec / euclid_norm

    return normalized_vector.tolist()

    ##Euclidean distance, note that both vectors are length normalized -> Normalized Euclidean Distance

def compute_euclidean_distance( vector1, vector2):
    v1 = np.array(vector1)
    v2 = np.array(vector2)

    return np.sqrt(np.sum((v1 - v2) ** 2))


def dot_product(vector1, vector2):
    return np.dot(np.array(vector1), np.array(vector2))


def compute_cosine_similarity_without_norm( vector1, vector2):
    return dot_product(vector1, vector2)


def compute_cosine_similarity_with_norm(vector1, vector2):
    return dot_product(vector1, vector2) / (
            get_euclidean_norm(vector1) * get_euclidean_norm(vector2))


def compute_cosine_distance_unnormalized_inputs(vector1, vector2):
    return 1 - compute_cosine_similarity_with_norm(vector1, vector2)


def compute_cosine_distance_normalized_inputs( vector1, vector2):
    return 1 - dot_product(vector1, vector2)