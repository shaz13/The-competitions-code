import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm
train=pd.read_csv('data/train.csv')
test=pd.read_csv('data/test.csv')
test_ids=test.transaction_id
y=train.target
train.drop(['target'],axis=1,inplace=True)
alldata=pd.concat([train,test])
alldata.drop(['cat_var_23','cat_var_24','cat_var_25','cat_var_26','cat_var_27','cat_var_28','cat_var_29',
        'cat_var_30','cat_var_31','cat_var_32','cat_var_33','cat_var_34','cat_var_35','cat_var_36',
        'cat_var_37','cat_var_38','cat_var_39','cat_var_40','cat_var_41','cat_var_42'],axis=1,inplace=True)
alldata.cat_var_1.fillna(value='gf',inplace=True)
alldata.cat_var_3.fillna(value='None',inplace=True)
alldata.cat_var_6.fillna(value='zs',inplace=True)
alldata.cat_var_8.fillna(value='dn',inplace=True)

label=LabelEncoder()
for col in tqdm(list(alldata.columns[alldata.columns.str.contains('cat')].values)):
    alldata[col]=label.fit_transform(alldata[col])

test=alldata[alldata.transaction_id.isin(test_ids)]
train=alldata[alldata.transaction_id.isin(test_ids)==False]
train.drop('transaction_id',axis=1,inplace=True)
test.drop('transaction_id',axis=1,inplace=True)

#Stacking script inspired from Kaggle kernels
def Stacking(model,train,y,test,n_fold, shuffle):
    folds=StratifiedKFold(n_splits=n_fold,random_state=123,shuffle=shuffle)
    RETURN_TEST_PROBAS=np.empty((test.shape[0],1),float)
    RETURN_TRAIN_PROBAS=np.empty((0,1),float)
    for train_indices,val_indices in folds.split(train,y.values):
        x_train,x_val=train.iloc[train_indices],train.iloc[val_indices]
        y_train,y_val=y.iloc[train_indices],y.iloc[val_indices]
        eval_set = [(x_val,y_val)]
        model.fit(X=x_train,y=y_train,eval_set=eval_set,eval_metric='auc',early_stopping_rounds=100,verbose=100)
        RETURN_TRAIN_PROBAS=np.append(RETURN_TRAIN_PROBAS,model.predict_proba(x_val)[:,1].reshape(-1,1),axis=0)
        RETURN_TEST_PROBAS=np.append(RETURN_TEST_PROBAS,model.predict_proba(test)[:,1].reshape(-1,1),axis=1)
    RETURN_TEST_PROBAS=np.mean(RETURN_TEST_PROBAS[:,1:],axis=1)
    return RETURN_TEST_PROBAS.reshape(-1,1),RETURN_TRAIN_PROBAS

lightgbm_classifier=LGBMClassifier(colsample_bytree=0.5,learning_rate=0.1,n_estimators=1300,
                      subsample=0.9,num_leaves=80,min_child_weight=1,n_jobs=-1)
TEST_PREDS ,TRAIN_PREDS=Stacking(model=lightgbm_classifier,n_fold=5,shuffle=False,train=train,test=test,y=y) ###

sub = pd.read_csv('data/sample_submissions.csv')
sub['target'] = TEST_PREDS
sub.to_csv('GHv4.csv',index=None)
