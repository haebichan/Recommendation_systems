# Building Recommendation Systems for Movie Recommender 

I have built a recommendation system based off data from the
[MovieLens dataset](http://grouplens.org/datasets/movielens/). It includes movie
information, user information, and the users' ratings. My goal was to build a
recommendation system and to suggest movies to users.

The **movies data** and **user data** are in `data/movies.dat` and `data/users.dat`.

The **ratings data** can be found in `data/training.csv`. The users' ratings have been broken into a training and test set for you (to obtain the testing set, we have split the 20% of **the most recent** ratings).


## Target Goal

My target was to provide a rating for each of those `user,movie` pairs. You will submit a csv file with three columns `user,movie,rating` as created by the script `src/run.py` (see below).

My score was measured based on how well I predicted the ratings for the users' ratings compared to the test set. 


## How to implement the recommender

The file `src/recommender.py` is the main template for creating my recommender. 

## How to run the recommender

By executing `src/run.py`, I could create an instance of a `MovieRecommender` class (see file `src/recommender.py`), feed it with the training data and output the results in a file.

