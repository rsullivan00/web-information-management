import numpy as np
from similarity import cosine_similarity


def train_users(users, train_file='train.txt'):
    training = open('train.txt', 'r')
    training = training.read().strip().split('\n')
    for i, line in enumerate(training):
        users[i] = [int(x) for x in line.split()]


def predict_score_batch(users, user_id, movie_ids):
    weights = [cosine_similarity(users[user_id], u) for
               u in users[:user_id]]

    ratings = []
    for movie_id in movie_ids:
        sum_w = 0
        rating = 3
        # Only compare with previous users.
        for i in range(user_id):
            user = users[i]
            if user[movie_id] == 0 or np.isnan(user[movie_id]):
                continue

            w = weights[i]
            sum_w += w
            rating += (w * user[movie_id])

        if sum_w != 0:
            rating /= sum_w

        rating = int(np.rint(rating))

        # Account for results outside of valid range.
        if rating > 5:
            rating = 5
        elif rating < 1:
            rating = 1
        ratings.append(rating)

    return ratings


def process_stored_data(users, user_id, movie_ids, results):
    if len(movie_ids) > 0:
        ratings = predict_score_batch(users, user_id, movie_ids)
        for m_id, r in zip(movie_ids, ratings):
            if r < 1 or r > 5:
                raise Exception('Rating %d' % r)
            users[user_id][m_id] = r
            results.append((user_id, m_id+1, r))


def test_dataset(users, dataset_file):
    """
    Blah
    """
    dataset = open(dataset_file, 'r').read().strip().split('\n')
    dataset = [data.split() for data in dataset]
    dataset = [[int(e) for e in data] for data in dataset]
    current_user_id = dataset[0][0] - 1
    movie_ids = []
    ratings = []
    results = []
    for user_id, movie_id, rating in dataset:
        user_id -= 1
        movie_id -= 1
        if user_id == current_user_id:
            if rating == 0:
                movie_ids.append(movie_id)
                ratings.append(rating)
            else:
                users[user_id][movie_id] = rating
            continue

        process_stored_data(users, user_id, movie_ids, results)

        # Initialize new batch
        print('User %d' % user_id, end='\r')
        current_user_id = user_id
        if rating == 0:
            movie_ids = [movie_id]
            ratings = [rating]
        else:
            users[user_id][movie_id] = rating
            movie_ids = []
            ratings = []

    process_stored_data(users, dataset[-1][0]-1, movie_ids, results)

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
    num_users = 500
    num_movies = 1000
    users = [[0] * num_movies] * num_users
    train_users(users)
    test_all(users)

main()
