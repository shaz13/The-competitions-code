from __future__ import division
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

print "Importing Train and Test Set"
train = pd.read_csv('data/train.csv', parse_dates=['start_date','creation_date','sell_date'])
test = pd.read_csv('data/test.csv', parse_dates=['start_date','creation_date','sell_date'])

print "Making the alldata set"
alldata= pd.concat([train,test])

alldata.drop([ 'indicator_code', 'status', 'desk_id','libor_rate'], axis=1,inplace=True )

print "Feature Engineering with Dates"
# Feature Engineering with Dates
alldata['start_date_month'] = alldata['start_date'].dt.month
alldata['start_date_year'] = alldata['start_date'].dt.year
alldata['creation_date_month'] = alldata['creation_date'].dt.month
alldata['creation_date_year'] = alldata['creation_date'].dt.year
alldata['sell_date_day'] = alldata['sell_date'].dt.day
alldata['sell_date_month'] = alldata['sell_date'].dt.month
alldata['sell_date_week'] = alldata['sell_date'].dt.weekday_name
alldata['sell_date_year'] = alldata['sell_date'].dt.year
alldata['Period Sell - Create'] =alldata['sell_date'].subtract(alldata['creation_date']).dt.days
alldata['Period Sell - Start'] = alldata['sell_date'].subtract(alldata['start_date']).dt.days

print "Hang on there... Its time to Label Encode things"
# LabelEncoder
obj_cols = [x for x in alldata.columns if alldata[x].dtype == 'object']
for x in obj_cols:
    encoder = LabelEncoder()
    alldata[x] = encoder.fit_transform(alldata[x])
alldata.drop(['creation_date', 'sell_date','start_date','sold','bought'], axis=1, inplace=True)
alldata = alldata.reset_index(drop=True)

print "Dropping Outliers"
# Dropping Outliers
drops = list(alldata[alldata['return'] > 0.20].index)
len_of_drops = len(drops)
print "There are {} outliers".format(len_of_drops)
alldata.drop(drops, inplace=True)

# split the merged data file into train and test respectively
train_feats = alldata[~pd.isnull(alldata['return'])]
test_feats = alldata[pd.isnull(alldata['return'])]


# Main Test set
X_TEST  = test_feats.drop('return', axis=1)

X = train_feats.drop('return', axis=1)
y = train_feats['return']


X.drop(['hedge_value','start_date_year','office_id'],axis=1,inplace=True)
X_TEST.drop(['hedge_value','start_date_year','office_id'],axis=1,inplace=True)

print "Checking the Cross validation Scores"
# Checking the Cross validation Scores
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import ShuffleSplit, cross_val_score
shuffle_split = ShuffleSplit(test_size=.4, train_size=.6, n_splits=5)
import xgboost as xgb
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor

model1=RandomForestRegressor(max_depth=8,n_estimators=180)
model2=GradientBoostingRegressor(max_depth=9,n_estimators=250)
model3= xgb.XGBRegressor(max_depth=9,learning_rate=.3,n_estimators=100,seed=1,subsample=.9,reg_alpha=0.01)

print "Ensembling the models"
model1.fit(X, y)
model2.fit(X, y)
model3.fit(X, y)

preds1 = model1.predict(X_TEST)
preds2 = model2.predict(X_TEST)
preds3 = model3.predict(X_TEST)
preds = (preds1 + preds2 + preds3)/3.0

print "Cooking the submission file"
sample_submission = pd.read_csv('data/sample_submission.csv')
sample_submission['return'] = preds2
sample_submission.to_csv('DEC_24_FINAL_SUB.csv', index=None) # 0.99724
print "Done. Good Luck!"
