# kaggle-homesite

This is a relatively simple high-scoring solution for the "Homesite Quote Conversion" contest on Kaggle 
(https://www.kaggle.com/c/homesite-quote-conversion). The task is to predict which insurance quotes will actual result in the sale of a 
policy. We are given a training data set of around 260,000 rows and around 300 features, a mixture of numeric and categorical.

My solution makes use of the powerful xgboost package for gradient-boosted regression trees, and uses only basic feature engineering. 
String-valued columns are one-hot encoded and an integer feature is added representing the day of the week.
The SalesField8 feature is dropped since it has a large number of unique integer values and therefore seems to represent some kind of 
identifier rather than a genuine feature.

We use relatively large values for the gamma and lambda regularisation parameters, which reduces the impact of irrelevant features without
the need to manually test and exclude these. We also use a small eta and large number of rounds.
