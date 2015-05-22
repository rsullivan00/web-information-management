import numpy as np


def vector_norm(v):
    """
    Returns the euclidean norm of a vector.
    """
    return np.sqrt(np.sum(np.square(v)))


def filter_common(v1, v2):
    """
    Returns new vectors (a, b) after filtering any
    indices where an element in a or b is 0 or None.
    """
#    a = [x for i, x in enumerate(v1) if x != 0 and v2[i] != 0]
#    b = [x for i, x in enumerate(v2) if x != 0 and v1[i] != 0]
#    return np.array(a), np.array(b)

    v1_new = []
    v2_new = []
    for i, x in enumerate(v1):
        y = v2[i]
        if y and x:
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


def pearson_correlation(a, b):
    """
    Computes the pearson correlation between two vectors.
    Returns a float in [-1, 1].
    """
    a_new, b_new = filter_common(a, b)
    mean_a = np.mean(a_new)
    mean_b = np.mean(b_new)
    a_adj = np.subtract(a_new, mean_a)
    b_adj = np.subtract(b_new, mean_b)

    w = cosine_similarity(a_adj, b_adj)
    return w
