import numpy as np
from similarity import (
    cosine_similarity,
    pearson_correlation,
    adj_cosine_similarity
)


def train_users(users, train_file='train.txt'):
    training = open('train.txt', 'r')
    training = training.read().strip().split('\n')
    for i, line in enumerate(training):
        users[i] = [int(x) for x in line.split()]


def score_batch_pearson_base(users, user, user_id, movie_ids, p=None):
    weights = [pearson_correlation(user, u) for
               u in users]
    if p is not None:
        weights = [w * np.abs(w)**(p-1) for w in weights]
    user_averages = [np.average([r for r in u if r > 0]) for u in users]

    ratings = []
    r_avg = np.average([x for x in user.values() if x > 0])
    for movie_id in movie_ids:
        sum_w = 0
        rating = 0
        for w, u_other, user_avg in zip(weights, users, user_averages):
            u_rating = u_other[movie_id]
            if u_rating == 0:
                continue

            sum_w += np.abs(w)
            rating += (w * (u_rating - user_avg))

        if sum_w != 0:
            rating = r_avg + (rating/sum_w)
        else:
            # If no relevant info was found, guess an average score.
            rating = r_avg

        ratings.append(rating)

    return ratings


def clean_rating(rating):
    rating = int(np.rint(rating))
    if rating > 5:
        print(rating)
        rating = 5
    elif rating < 1:
        print(rating)
        rating = 1

    return rating


def clean_ratings(ratings):
    return [clean_rating(r) for r in ratings]


def score_batch_pearson(users, user, user_id, movie_ids):
    ratings = score_batch_pearson_base(users, user, user_id, movie_ids)
    return clean_ratings(ratings)


def score_batch_pearson_iuf(users, user, user_id, movie_ids):
    if not hasattr(score_batch_pearson_iuf, 'run'):
        score_batch_pearson_iuf.run = True
        m = len(users)
        for i in range(1000):
            m_j = len([0 for u in users if u[i] != 0])
            if m_j == 0:
                # Do nothing
                continue
            iuf = np.log(m/m_j)
            for u in users:
                u[i] *= iuf

    ratings = score_batch_pearson_base(users, user, user_id, movie_ids)

    return clean_ratings(ratings)


def score_batch_pearson_case(users, user, user_id, movie_ids):
    ratings = score_batch_pearson_base(users, user, user_id, movie_ids, p=2.5)
    return clean_ratings(ratings)


def score_batch_pearson_case_iuf(users, user, user_id, movie_ids):
    if not hasattr(score_batch_pearson_case_iuf, 'run'):
        print('Applying iuf')
        score_batch_pearson_case_iuf.run = True
        m = len(users)
        for i in range(1000):
            m_j = len([0 for u in users if u[i] != 0])
            if m_j == 0:
                # Do nothing
                continue
            iuf = np.log(m/m_j)
            for u in users:
                u[i] *= iuf

    ratings = score_batch_pearson_base(users, user, user_id, movie_ids, p=2.5)

    return clean_ratings(ratings)


def score_batch_cosine(users, user, user_id, movie_ids):
    weights = [cosine_similarity(user, u) for
               u in users]

    ratings = []
    for movie_id in movie_ids:
        sum_w = 0
        rating = 0

        for w, u_other in zip(weights, users):
            u_rating = u_other[movie_id]
            if u_rating == 0:
                continue

            sum_w += w
            rating += (w * u_rating)

        if sum_w != 0:
            rating /= sum_w
        else:
            # If no relevant info was found, guess a score of 3.
            rating = 3

        rating = int(np.rint(rating))
        ratings.append(rating)

    return clean_ratings(ratings)


def score_batch_item_centered(users, user, user_id, movie_ids):
    items = np.array(users).T
    ratings = []
    user_items = list(user.keys())
    user_averages = [np.average([r for r in u if r > 0]) for u in users]

    for movie_id in movie_ids:
        item = items[movie_id]
        i_ratings = [r for r in item if r > 0]
        if len(i_ratings) > 0:
            r_avg = np.average(i_ratings)
        else:
            r_avg = 3

        weights = [adj_cosine_similarity(items[i], item, users)
                   for i in user_items]
        sum_w = 0
        rating = 0

        for w, i, user_avg in zip(weights, user_items, user_averages):
            u_rating = user[i]
            sum_w += np.abs(w)
            rating += (w * (u_rating - user_avg))

        if sum_w != 0:
            rating = r_avg + (rating/sum_w)
        else:
            # If no relevant info was found, guess an average score.
            rating = r_avg

        rating = int(np.rint(rating))
        ratings.append(rating)

    return clean_ratings(ratings)


def score_batch_item(users, user, user_id, movie_ids):
    items = np.array(users).T
    ratings = []
    user_items = list(user.keys())
    for movie_id in movie_ids:
        item = items[movie_id]
        weights = [adj_cosine_similarity(items[i], item, users)
                   for i in user_items]
        sum_w = 0
        rating = 0

        for w, i in zip(weights, user_items):
            u_rating = user[i]

            sum_w += np.abs(w)
            rating += (w * u_rating)

        if sum_w != 0:
            rating /= sum_w
        else:
            # If no relevant info was found, guess a score of 3.
            rating = 3

        rating = int(np.rint(rating))
        ratings.append(rating)

    return clean_ratings(ratings)


def process_stored_data(users, user, user_id, movie_ids, results):
    if len(movie_ids) > 0:
        ratings = score_batch_item_centered(users, user, user_id, movie_ids)

        for m_id, r in zip(movie_ids, ratings):
            if r < 1 or r > 5:
                raise Exception('Rating %d' % r)
            results.append((user_id+1, m_id+1, r))


def test_dataset(users, dataset_file):
    dataset = open(dataset_file, 'r').read().strip().split('\n')
    dataset = [data.split() for data in dataset]
    dataset = [[int(e) for e in data] for data in dataset]
    current_user_id = dataset[0][0] - 1
    current_user = {}
    movie_ids = []
    results = []
    for user_id, movie_id, rating in dataset:
        user_id -= 1
        movie_id -= 1
        print('User %d' % user_id, end='\r')

        # If it's a new user, process buffer and reinitialize.
        if user_id != current_user_id:
            process_stored_data(
                users,
                current_user,
                current_user_id,
                movie_ids,
                results
            )
            current_user_id = user_id
            current_user = {}
            movie_ids = []

        if rating == 0:
            movie_ids.append(movie_id)
        else:
            current_user[movie_id] = rating

    process_stored_data(
        users,
        current_user,
        current_user_id,
        movie_ids,
        results
    )

    return results


def log_results(results, logfile):
    fout = open(logfile, 'w')
    for result in results:
        fout.write(' '.join(str(x) for x in result) + '\n')


def test_all(users):
    print('Processing test5')
    results = test_dataset(users, 'test5.txt')
    log_results(results, 'result5.txt')
    print('Processing test10')
    results = test_dataset(users, 'test10.txt')
    log_results(results, 'result10.txt')
    print('Processing test20')
    results = test_dataset(users, 'test20.txt')
    log_results(results, 'result20.txt')


def main():
    num_users = 200
    num_movies = 1000
    users = [[0] * num_movies] * num_users
    train_users(users, 'train.txt')
    test_all(users)

main()
