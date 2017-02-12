import pandas as pd
import numpy as np
from sklearn import preprocessing
import xgboost as xgb

# read data
train = pd.read_csv('Data\\train.csv',parse_dates=['Original_Quote_Date'])
test = pd.read_csv('Data\\test.csv',parse_dates=['Original_Quote_Date'])


def add_features(data_in):

    data_out = data_in.drop(['QuoteNumber'], axis=1)

    # remove quote conversion flag if it exists
    if 'QuoteConversion_Flag' in data_out.columns:
        data_out = data_out.drop(['QuoteConversion_Flag'], axis=1)

    # binary day of week features
    week_dict = {'Monday':0, 'Tuesday':1, 'Wednesday':2, 'Thursday':3, 'Friday':4, 'Saturday':5, 'Sunday': 6}
    for day in week_dict:
        weekday = lambda x: x.dayofweek
        data_out[day] = (data_in['Original_Quote_Date'].apply(weekday)==week_dict[day]).astype(int)

    return data_out

# add features
train_x = add_features(train)
test_x = add_features(test)

# remove salesfield8
cols_no_sales8 = filter(lambda x: x!= 'SalesField8',train_x.columns)
train_x = train_x[cols_no_sales8]
test_x = test_x[cols_no_sales8]

encoder = {} # dictionary of encoders indexed by col name
all_data = pd.concat((train_x,test_x),axis=0)
cat_feat_list = []
for (ind, col) in enumerate(train_x.columns):
    # encode
    if type(train_x[col].iloc[0]) in [str, np.datetime64, pd.tslib.Timestamp]:
        if type(train_x[col].iloc[0]) == str:
            cat_feat_list.append(ind)
        encoder[col] = preprocessing.LabelEncoder()
        encoder[col].fit(all_data[col])
        all_data[col] = encoder[col].transform(all_data[col])
        train_x[col] = encoder[col].transform(train_x[col])
        test_x[col] = encoder[col].transform(test_x[col])


# replace nulls so we can fit the one-hot encoder
all_data[pd.isnull(all_data)] = -99
train_x[pd.isnull(train_x)] = -99
test_x[pd.isnull(test_x)] = -99
one_hot = preprocessing.OneHotEncoder(categorical_features = cat_feat_list, sparse = False)
one_hot.fit(all_data)
train_x = one_hot.transform(train_x)
test_x = one_hot.transform(test_x)

train_x[train_x == -99] = np.nan
test_x[test_x == -99] = np.nan

train_y = train['QuoteConversion_Flag']
xg_train = xgb.DMatrix(train_x, label = train_y, missing = np.nan)

# fit estimator
param = {'eta': .05, 'subsample': .9, 'lambda': 11, 'gamma':11, 'min_child_weight':1, 'max_depth':7, 'objective':'binary:logistic', 'eval_metric':'auc', 'silent':1, 'seed':0}
est = xgb.train(param,xg_train,num_boost_round=5000, evals=[(xg_train,'train')])

# predict
xg_test = xgb.DMatrix(test_x, missing = np.nan)
predicted = est.predict(xg_test)

# write submission to csv
submit = pd.DataFrame(test['QuoteNumber'])
submit['QuoteConversion_Flag'] = predicted
submit.to_csv('Submission.csv', index=False)

print "done"
