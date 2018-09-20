import logging
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS

class MovieRecommender():
    """Template class for a Movie Recommender system."""

    def __init__(self):
        """Constructs a MovieRecommender"""
        self.logger = logging.getLogger('reco-cs')
        self.spark = SparkSession.builder.getOrCreate()
        self.als = ALS(
            itemCol='movie',
            userCol='user',
            ratingCol='rating',
            nonnegative=True,    
            regParam=0.1,
            rank=10)
        self.model = None
        
    def fit(self, ratings):
        """
        Trains the recommender on a given set of ratings.

        Parameters
        ----------
        ratings : pandas dataframe, shape = (n_ratings, 4)
                  with columns 'user', 'movie', 'rating', 'timestamp'

        Returns
        -------
        self : object
            Returns self.
        """
        self.logger.debug("starting fit")
        ratings_df = self.spark.createDataFrame(ratings)
        self.model = self.als.fit(ratings_df)
        # ...

        self.logger.debug("finishing fit")
        return(self)


    def transform(self, requests):
        """
        Predicts the ratings for a given set of requests.

        Parameters
        ----------
        requests : pandas dataframe, shape = (n_ratings, 2)
                  with columns 'user', 'movie'

        Returns
        -------
        dataframe : a pandas dataframe with columns 'user', 'movie', 'rating'
                    column 'rating' containing the predicted rating
        """
        self.logger.debug("starting predict")
        self.logger.debug("request count: {}".format(requests.shape[0]))

        #requests['rating'] = np.random.choice(range(1, 5), requests.shape[0])
        ratings_df = self.spark.createDataFrame(requests)
        spark_prediction = self.model.transform(ratings_df)
        submission = spark_prediction.toPandas()
        submission.rename(columns={'prediction': 'rating'}, inplace=True)
        self.logger.debug("finishing predict")
        return(submission)


if __name__ == "__main__":
    logger = logging.getLogger('reco-cs')
    logger.critical('you should use run.py instead')
