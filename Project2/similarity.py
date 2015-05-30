import numpy as np


def vector_norm(v):
    """
    Returns the euclidean norm of a vector.
    """
    return np.sqrt(np.sum(np.square(v)))


def filter_common(v1, v2):
    """
    Returns new vectors (a, b) after filtering any
    indices where an element in a or b <= 0.
    """
    v1_new = []
    v2_new = []
    for i, x in enumerate(v1):
        y = v2[i]
        if y > 0 and x > 0:
            v1_new.append(x)
            v2_new.append(y)

    return np.array(v1_new), np.array(v2_new)


def cosine_similarity(a, b):
    """
    Cosine similarity between two vectors.
    Returns a float in [-1, 1].
    """
    a_new, b_new = filter_common(a, b)

    sim = np.dot(a_new, b_new)

    norm_a = vector_norm(a_new)
    norm_b = vector_norm(b_new)
    if norm_a != 0 and norm_b != 0:
        sim /= (norm_a * norm_b)
    else:
        sim = 0

    if sim > 1:
        sim = 1
    elif sim < -1:
        sim = -1

    return sim


def adj_cosine_similarity(a, b, users):
    """
    Returns a float in [-1, 1].
    a and b are items, containing user ratings.
    """
    if not hasattr(adj_cosine_similarity, 'avgs'):
        filtered_users = [
            [x for x in u if x > 0] for u in users]
        adj_cosine_similarity.avgs = [np.mean(u) for u in filtered_users]

    avgs = adj_cosine_similarity.avgs
    a_adj = np.subtract(a, avgs)
    b_adj = np.subtract(b, avgs)
    a_new, b_new = filter_common(a_adj, b_adj)

    return cosine_similarity(a_new, b_new)


def pearson_correlation(a, b):
    """
    Computes the pearson correlation between two vectors.
    Returns a float in [-1, 1].
    """
    a_new, b_new = filter_common(a, b)
    a_mean = a_new.mean()
    b_mean = b_new.mean()
    a_adj = np.subtract(a_new, a_mean[np.newaxis])
    b_adj = np.subtract(b_new, b_mean[np.newaxis])
    num = np.dot(a_adj, b_adj)
    sum_sq_a = np.dot(a_adj, a_adj)
    sum_sq_b = np.dot(b_adj, b_adj)
    denom = np.sqrt(sum_sq_a * sum_sq_b)
    if denom == 0:
        return 0
    return num/denom
