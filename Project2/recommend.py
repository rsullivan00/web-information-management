import numpy as np
from similarity import cosine_similarity, pearson_correlation


def train_users(users, train_file='train.txt'):
    training = open('train.txt', 'r')
    training = training.read().strip().split('\n')
    for i, line in enumerate(training):
        users[i] = [int(x) for x in line.split()]


def predict_score_batch_pearson(users, user_id, movie_ids):
    weights = [pearson_correlation(users[user_id], u) for
               u in users[:200]]
    user_averages = [np.average(u) for u in users[:200]]

    ratings = []
    r_avg = np.average([x for x in users[user_id] if x > 0])
    for movie_id in movie_ids:
        sum_w = 0
        rating = 0
        # Only compare with previous users.
        for i, w in enumerate(weights):
            user = users[i]
            if user[movie_id] == 0:
                continue

            user_avg = user_averages[i]

            sum_w += np.abs(w)
            rating += (w * (user[movie_id] - user_avg))

        if sum_w != 0:
            rating = r_avg + (rating/sum_w)
        else:
            # If no relevant info was found, guess an average score.
            rating = r_avg

        rating = int(np.rint(rating))
        if rating > 5:
            rating = 5
        elif rating < 1:
            rating = 1

        ratings.append(rating)

    return ratings


def predict_score_batch_cosine(users, user_id, movie_ids):
    weights = [cosine_similarity(users[user_id], u) for
               u in users[:200]]

    ratings = []
    for movie_id in movie_ids:
        sum_w = 0
        rating = 0
        # Only compare with previous users.
        for i, w in enumerate(weights):
            user = users[i]
            if user[movie_id] == 0:
                continue

            sum_w += w
            rating += (w * user[movie_id])

        if sum_w != 0:
            rating /= sum_w
        else:
            # If no relevant info was found, guess a score of 3.
            rating = 3

        rating = int(np.rint(rating))
        ratings.append(rating)

    return ratings


def process_stored_data(users, user_id, movie_ids, results):
    if len(movie_ids) > 0:
        ratings = predict_score_batch_pearson(users, user_id, movie_ids)
        for m_id, r in zip(movie_ids, ratings):
            if r < 1 or r > 5:
                raise Exception('Rating %d' % r)
            users[user_id][m_id] = r
            results.append((user_id+1, m_id+1, r))


def test_dataset(users, dataset_file):
    """
    Blah
    """
    dataset = open(dataset_file, 'r').read().strip().split('\n')
    dataset = [data.split() for data in dataset]
    dataset = [[int(e) for e in data] for data in dataset]
    current_user_id = dataset[0][0] - 1
    movie_ids = []
    results = []
    for user_id, movie_id, rating in dataset:
        user_id -= 1
        movie_id -= 1
        print('User %d' % user_id, end='\r')

        # If it's a new user, process buffer and reinitialize.
        if user_id != current_user_id:
            process_stored_data(users, current_user_id, movie_ids, results)
            current_user_id = user_id
            movie_ids = []

        if rating == 0:
            movie_ids.append(movie_id)
        else:
            movie_ids.append(movie_id)
            users[user_id][movie_id] = rating

    process_stored_data(users, current_user_id, movie_ids, results)

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
    train_users(users, 'train.txt')
    test_all(users)

main()
