# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 16:34:00 2019

@author: 何洪涛
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model,ensemble,model_selection
from sklearn.model_selection import cross_val_score
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import catboost as ct
plt.rcParams['font.sans-serif']='SimHei'
plt.rcParams['axes.unicode_minus']=False
df_tsv1=pd.read_csv(r'D:\data\tsv\P0001_0017.tsv',sep='\t')
df_tsv1['value'].value_counts()
df_train=pd.read_csv(r'D:\data\1_preliminary_list_train.csv')
df_train.loc[df_train['label']=='AD','label']=0
df_train.loc[df_train['label']=='CTRL','label']=1
df_train.loc[df_train['label']=='MCI','label']=2
df_tsv2=pd.read_csv(r'D:\data\tsv\P0004_0025.tsv',sep='\t')
df_tsv3=pd.read_csv(r'D:\data\tsv\P0009_0031.tsv',sep='\t')
df_tsv4=pd.read_csv(r'D:\data\tsv\P0016_0050.tsv',sep='\t')
df_tsv5=pd.read_csv(r'D:\data\tsv\P0018_0052.tsv',sep='\t')
df_egemaps_pre=pd.read_csv(r'D:\data\egemaps_pre.csv')
index_label3=list(df_train[df_train['label']==0]['uuid'])
index_label1=list(df_train[df_train['label']==1]['uuid'])
def printvalue_counts(df):
    
    print(df['speaker'].value_counts())
    a=df['speaker'].value_counts()
    a.plot(kind='bar')
    plt.show()
def lensentence(df):
    a1=len(df)-1
    a=df['start_time'][0]
    b=df['end_time'][a1]
    c=b-a
    return c
train_index=df_train['uuid']
lenA=[]
lenB=[]
lenstop=[]
len_all_time=[]
time_A=[]
time_B=[]
length=[]
count_A_question_mark=[]
count_A_en_mark=[]
count_A_gang=[]
count_B_question_mark=[]
count_B_en_mark=[]
count_B_gang=[]
count_B_shanghai=[]
start_time_end_time_mean=[]
start_time_end_time_min=[]
start_time_end_time_max=[]
start_time_end_time_std=[]
start_time_end_time_median=[]
start_time_end_time_skew=[]
def start_time_end_time(z):
    z['end_time-start_time']=z['end_time']-z['start_time']
    start_time_end_time_mean1=z['end_time-start_time'].mean()
    start_time_end_time_min1=z['end_time-start_time'].min()
    start_time_end_time_max1=z['end_time-start_time'].max()
    start_time_end_time_std1=z['end_time-start_time'].std()
    start_time_end_time_median1=z['end_time-start_time'].median()
    start_time_end_time_skew1=z['end_time-start_time'].skew()
    start_time_end_time_mean.append(start_time_end_time_mean1)
    start_time_end_time_min.append(start_time_end_time_min1)
    start_time_end_time_max.append(start_time_end_time_max1)
    start_time_end_time_median.append(start_time_end_time_median1)
    start_time_end_time_std.append(start_time_end_time_std1)
    start_time_end_time_skew.append(start_time_end_time_skew1)
def scatter(x):
    if x<=lentimeB/4:
        return 1
    elif x>lentimeB/4 and x<=lentimeB/2:
        return 2
    elif x> lentimeB/2 and x<=3*lentimeB/4:
        return 3
    elif x>3*lentimeB/4:
        return 4   
def count_A_features(df):
    count_a_question=0
    count_en=0
    count_gang=0
    a1=df.loc[df['speaker']=='<A>']['value']
    i1=a1.index
    for i in i1:
        if '？' in a1[i]:
            count_a_question=count_a_question+1
        elif '&嗯' in a1[i]:
            count_en=count_en+1
        elif '/' in a1[i]:
            count_gang=count_gang+1
    count_A_gang.append(count_gang)      
    count_A_question_mark.append(count_a_question)
    count_A_en_mark.append(count_en)
def count_B_features(df):
    count_a_question=0
    count_en=0
    count_gang=0
    count_shanghai=0
    a1=df.loc[df['speaker']=='<B>']['value']
    i1=a1.index
    for i in i1:
        if '？' in a1[i]:
            count_a_question=count_a_question+1
        elif '&嗯' in a1[i]:
            count_en=count_en+1
        elif '/' in a1[i]:
            count_gang=count_gang+1
        elif '【上海话】' in a1[i]:
            count_shanghai=count_shanghai+1
    count_B_shanghai.append(count_shanghai)
    count_B_gang.append(count_gang)      
    count_B_question_mark.append(count_a_question)
    count_B_en_mark.append(count_en)
    
def counts_features(df):
    a1=df['speaker'].value_counts()
    A=a1['<A>']
    B=a1['<B>']
    stop=a1.sum()-A-B
    lenA.append(A)
    lenB.append(B)
    lenstop.append(stop)
    c=lensentence(df)
    len_all_time.append(c)
for index in train_index:
    i='.tsv'
    a='D:\\data\\tsv\\'+index+i
    df=pd.read_csv(a,sep='\t')
    counts_features(df)
    time_a_start=df.loc[df['speaker']=='<A>','start_time']
    time_a_end=df.loc[df['speaker']=='<A>','end_time']
    time_a=np.subtract(time_a_end,time_a_start).sum()
    time_A.append(time_a)
    time_b_start=df.loc[df['speaker']=='<B>','start_time']
    time_b_end=df.loc[df['speaker']=='<B>','end_time']
    time_b=np.subtract(time_b_end,time_b_start).sum()
    time_B.append(time_b)
    long=len(df)
    length.append(long)
    count_A_features(df)
    count_B_features(df)
    start_time_end_time(df)
y_train=df_train['label'].astype('int')
df_train['lenA']=pd.Series(lenA)
df_train['lenB']=pd.Series(lenB)
df_train['lenstop']=pd.Series(lenstop)
df_train['all_time']=pd.Series(len_all_time)
A_subtract_B=np.subtract(lenA,lenB)
df_train['A-B']=pd.Series(A_subtract_B)
df_train['time_A']=pd.Series(time_A)
df_train['time_B']=pd.Series(time_B)
df_train['talk_length']=pd.Series(length)
time_A_subtract_B=np.subtract(time_A,time_B)
df_train['time_A_subtract_B']=pd.Series(time_A_subtract_B)
df_train['count_A_question_mark']=pd.Series(count_A_question_mark)
df_train['count_A_en_mark']=pd.Series(count_A_en_mark)
df_train['count_A_gang']=pd.Series(count_A_gang)
df_train['count_B_question_mark']=pd.Series(count_B_question_mark)
df_train['count_B_en_mark']=pd.Series(count_B_en_mark)
df_train['count_B_gang']=pd.Series(count_B_gang)
df_train['shanghaihua']=pd.Series(count_B_shanghai)
#df_train['stop_in_talk']=pd.Series(np.divide(df_train['lenstop'],df_train['talk_length']))
df_train['start_time_end_time_mean']=pd.Series(start_time_end_time_mean)
df_train['start_time_end_time_min']=pd.Series(start_time_end_time_min)
df_train['start_time_end_time_max']=pd.Series(start_time_end_time_max)
df_train['start_time_end_time_std']=pd.Series(start_time_end_time_std)
df_train['start_time_end_time_median']=pd.Series(start_time_end_time_median)
df_train['start_time_end_time_skew']=pd.Series(start_time_end_time_skew)
q1=df_train.time_B
q2=df_train.lenB
df_train['lenbchutimeb']=pd.Series(np.divide(q2,q1))
q1_a=df_train.time_A
q2_a=df_train.lenA
df_train['lenachutimea']=pd.Series(np.divide(q2_a,q1_a))
lentimeB=df_train.time_B.max()-df_train.time_B.min()
#df_train['time_B']=df_train['time_B'].apply(scatter)
lentimeB=df_train.time_A.max()-df_train.time_A.min()
#df_train['time_A']=df_train['time_A'].apply(scatter)
df_train=pd.merge(df_train,df_egemaps_pre,how='left',on=['uuid'])
df_train.drop(['uuid','label','A-B'],axis=1,inplace=True)
df_1700=pd.read_csv(r'D:\data\egemaps\P0001_0017.csv',sep=';')
df_1700.drop(['name','frameTime'],axis=1,inplace=True)
name=[]
for index in list(df_1700.columns):
    i1=index+'mean'
    name.append(i1)
a1=df_1700.mean()
a1=pd.DataFrame(a1).T
a1.columns=name
for index in train_index[1:]:
    i='.csv'
    a='D:\\data\\egemaps\\'+index+i
    df=pd.read_csv(a,sep=';')
    df.drop(['name','frameTime'],axis=1,inplace=True)
    a2=df.mean()
    a2=pd.DataFrame(a2).T
    a2.columns=name
    a1=pd.concat([a1,a2],axis=0,ignore_index=True)
df_train=pd.concat([df_train,a1],axis=1)
name=[]
for index in list(df_1700.columns):
    i1=index+'std'
    name.append(i1)
a1=df_1700.std()
a1=pd.DataFrame(a1).T
a1.columns=name
for index in train_index[1:]:
    i='.csv'
    a='D:\\data\\egemaps\\'+index+i
    df=pd.read_csv(a,sep=';')
    df.drop(['name','frameTime'],axis=1,inplace=True)
    a2=df.std()
    a2=pd.DataFrame(a2).T
    a2.columns=name
    a1=pd.concat([a1,a2],axis=0,ignore_index=True)
df_train=pd.concat([df_train,a1],axis=1)
name=[]
for index in list(df_1700.columns):
    i1=index+'median'
    name.append(i1)
a1=df_1700.median()
a1=pd.DataFrame(a1).T
a1.columns=name
for index in train_index[1:]:
    i='.csv'
    a='D:\\data\\egemaps\\'+index+i
    df=pd.read_csv(a,sep=';')
    df.drop(['name','frameTime'],axis=1,inplace=True)
    a2=df.median()
    a2=pd.DataFrame(a2).T
    a2.columns=name
    a1=pd.concat([a1,a2],axis=0,ignore_index=True)
df_train=pd.concat([df_train,a1],axis=1)
F0Var=[]
F0qujian=[]
Loundnessqujian=[]
for index in train_index:
    i='.csv'
    a='D:\\data\\egemaps\\'+index+i
    df=pd.read_csv(a,sep=';')
    F0Var1=df['F0semitoneFrom27.5Hz_sma3nz'].var()
    F0qujian1=df['F0semitoneFrom27.5Hz_sma3nz'].max()-df['F0semitoneFrom27.5Hz_sma3nz'].min()
    Loundnessqujian1=df['Loudness_sma3'].max()-df['Loudness_sma3'].min()
    F0Var.append(F0Var1)
    F0qujian.append(F0qujian1)
    Loundnessqujian.append(Loundnessqujian1)
df_train['F0Var']=pd.Series(F0Var)
df_train['F0qujian']=pd.Series(F0qujian)
df_train['Loundnessqujian']=pd.Series(Loundnessqujian)
x_train=df_train
df_test=pd.read_csv(r'D:\data\1_preliminary_list_test.csv')
test_index=df_test.uuid
lenA=[]
lenB=[]
lenstop=[]
len_all_time=[]
time_A=[]
time_B=[]
length=[]
count_A_question_mark=[]
count_A_en_mark=[]
count_A_gang=[]
count_B_question_mark=[]
count_B_en_mark=[]
count_B_gang=[]
count_B_shanghai=[]
start_time_end_time_mean=[]
start_time_end_time_min=[]
start_time_end_time_max=[]
start_time_end_time_std=[]
start_time_end_time_median=[]
start_time_end_time_skew=[]
for index in test_index:
    i='.tsv'
    a='D:\\data\\tsv\\'+index+i
    df=pd.read_csv(a,sep='\t')
    counts_features(df)
    time_a_start=df.loc[df['speaker']=='<A>','start_time']
    time_a_end=df.loc[df['speaker']=='<A>','end_time']
    time_a=np.subtract(time_a_end,time_a_start).sum()
    time_A.append(time_a)
    time_b_start=df.loc[df['speaker']=='<B>','start_time']
    time_b_end=df.loc[df['speaker']=='<B>','end_time']
    time_b=np.subtract(time_b_end,time_b_start).sum()
    time_B.append(time_b)
    long=len(df)
    length.append(long)
    count_A_features(df)
    count_B_features(df)
    start_time_end_time(df)
df_test['lenA']=pd.Series(lenA)
df_test['lenB']=pd.Series(lenB)
df_test['lenstop']=pd.Series(lenstop)
df_test['all_time']=pd.Series(len_all_time)
A_subtract_B=np.subtract(lenA,lenB)
df_test['A-B']=pd.Series(A_subtract_B)
df_test['time_A']=pd.Series(time_A)
df_test['time_B']=pd.Series(time_B)
df_test['talk_length']=pd.Series(length)
time_A_subtract_B=np.subtract(time_A,time_B)
df_test['time_A_subtract_B']=pd.Series(time_A_subtract_B) 
df_test['count_A_question_mark']=pd.Series(count_A_question_mark)
df_test['count_A_en_mark']=pd.Series(count_A_en_mark)
df_test['count_A_gang']=pd.Series(count_A_gang)
df_test['count_B_question_mark']=pd.Series(count_B_question_mark)
df_test['count_B_en_mark']=pd.Series(count_B_en_mark)
df_test['count_B_gang']=pd.Series(count_B_gang)
df_test['shanghaihua']=pd.Series(count_B_shanghai)
#df_test['stop_in_talk']=pd.Series(np.divide(df_test['lenstop'],df_test['talk_length']))
df_test['start_time_end_time_mean']=pd.Series(start_time_end_time_mean)
df_test['start_time_end_time_min']=pd.Series(start_time_end_time_min)
df_test['start_time_end_time_max']=pd.Series(start_time_end_time_max)
df_test['start_time_end_time_std']=pd.Series(start_time_end_time_std)
df_test['start_time_end_time_median']=pd.Series(start_time_end_time_median)
df_test['start_time_end_time_skew']=pd.Series(start_time_end_time_skew)
q1=df_test.time_B
q2=df_test.lenB
df_test['lenbchutimeb']=pd.Series(np.divide(q2,q1))
q1_a=df_test.time_A
q2_a=df_test.lenA
df_test['lenachutimea']=pd.Series(np.divide(q2_a,q1_a))
lentimeB=df_test.time_B.max()-df_test.time_B.min()
#df_test['time_B']=df_test['time_B'].apply(scatter)
lentimeB=df_test.time_A.max()-df_test.time_A.min()
#df_test['time_A']=df_test['time_A'].apply(scatter)
df_test=pd.merge(df_test,df_egemaps_pre,how='left',on=['uuid'])
df_test.drop(['label','uuid','A-B'],axis=1,inplace=True)
df_0038=pd.read_csv(r'D:\data\egemaps\P0012_0038.csv',sep=';')
df_0038.drop(['name','frameTime'],axis=1,inplace=True)
name=[]
for index in list(df_0038.columns):
    i1=index+'mean'
    name.append(i1)
a1=df_0038.mean()
a1=pd.DataFrame(a1).T
a1.columns=name
for index in test_index[1:]:
    i='.csv'
    a='D:\\data\\egemaps\\'+index+i
    df=pd.read_csv(a,sep=';')
    df.drop(['name','frameTime'],axis=1,inplace=True)
    a2=df.mean()
    a2=pd.DataFrame(a2).T
    a2.columns=name
    a1=pd.concat([a1,a2],axis=0,ignore_index=True)
df_test=pd.concat([df_test,a1],axis=1)
name=[]
for index in list(df_0038.columns):
    i1=index+'std'
    name.append(i1)
a1=df_0038.std()
a1=pd.DataFrame(a1).T
a1.columns=name
for index in test_index[1:]:
    i='.csv'
    a='D:\\data\\egemaps\\'+index+i
    df=pd.read_csv(a,sep=';')
    df.drop(['name','frameTime'],axis=1,inplace=True)
    a2=df.std()
    a2=pd.DataFrame(a2).T
    a2.columns=name
    a1=pd.concat([a1,a2],axis=0,ignore_index=True)
df_test=pd.concat([df_test,a1],axis=1)
name=[]
for index in list(df_0038.columns):
    i1=index+'median'
    name.append(i1)
a1=df_0038.median()
a1=pd.DataFrame(a1).T
a1.columns=name
for index in test_index[1:]:
    i='.csv'
    a='D:\\data\\egemaps\\'+index+i
    df=pd.read_csv(a,sep=';')
    df.drop(['name','frameTime'],axis=1,inplace=True)
    a2=df.median()
    a2=pd.DataFrame(a2).T
    a2.columns=name
    a1=pd.concat([a1,a2],axis=0,ignore_index=True)
df_test=pd.concat([df_test,a1],axis=1)
F0Var=[]
F0qujian=[]
Loundnessqujian=[]
for index in test_index:
    i='.csv'
    a='D:\\data\\egemaps\\'+index+i
    df=pd.read_csv(a,sep=';')
    F0Var1=df['F0semitoneFrom27.5Hz_sma3nz'].var()
    F0qujian1=df['F0semitoneFrom27.5Hz_sma3nz'].max()-df['F0semitoneFrom27.5Hz_sma3nz'].min()
    Loundnessqujian1=df['Loudness_sma3'].max()-df['Loudness_sma3'].min()
    F0Var.append(F0Var1)
    F0qujian.append(F0qujian1)
    Loundnessqujian.append(Loundnessqujian1)
df_test['F0Var']=pd.Series(F0Var)
df_test['F0qujian']=pd.Series(F0qujian)
df_test['Loundnessqujian']=pd.Series(Loundnessqujian)
x_test=df_test
#regr=linear_model.LogisticRegression(penalty='l1',C=1.0)
#regr.fit(x_train,y_train)
#score=cross_val_score(regr,x_train,y_train,scoring='recall_macro',cv=5)
#print('逻辑回归')
#print(score.mean())
#y_test_label=regr.predict(x_test)
#y_test_label=pd.Series(y_test_label).replace([1,0],['CTRL','AD'])
#y_test_label=pd.DataFrame(data=y_test_label,columns=['label'])
#submibt=pd.concat([test_index,y_test_label],axis=1)
#submibt.to_csv(r'D:\reach\submit_7_13.csv',index=False)
##随机森林
#clf=ensemble.RandomForestClassifier(max_depth=6,min_samples_split=5,n_estimators=400,n_jobs=-1,random_state=100)
#clf.fit(x_train,y_train)
#score=cross_val_score(clf,x_train,y_train,scoring='recall_macro',cv=5)
#print('随机森林')
#print(score.mean())
#y_test_label=clf.predict(x_test)
#y_test_label=pd.Series(y_test_label).replace([1,0],['CTRL','AD'])
#y_test_label=pd.DataFrame(data=y_test_label,columns=['label'])
#submibt=pd.concat([test_index,y_test_label],axis=1)
#submibt.to_csv(r'D:\reach\submit_7_19.csv',index=False)
#clf1=lgb.LGBMClassifier(boosting_type='gbdt',learning_rate=0.01,objective='binary',feature_fraction=0.8,bagging_fraction=0.8,bagging_freq=5,num_leaves=1000,verbose=-1,max_depth=-1,seed=42)
#clf1.fit(x_train,y_train)
#pre=clf1.predict(x_test)  
#score=cross_val_score(clf1,x_train,y_train,scoring='recall_macro',cv=5)
#print(score.mean())
x_train1,x_eval2,y_train1,y_eval2=train_test_split(df_train,y_train,test_size=0.2,random_state=10)
lgb_train1=lgb.Dataset(df_train,y_train)
lgb_train=lgb.Dataset(x_train1,y_train1,free_raw_data=False)
lgb_eval=lgb.Dataset(x_eval2,y_eval2,free_raw_data=False,reference=lgb_train)
params = {
            'boosting_type': 'gbdt',
            'objective': 'binary',
            'metric': 'auc',
            'learning_rate': 0.01,
            'max_depth':-1,
 
            'max_bin':9,
            'min_data_in_leaf':12,
 
            'feature_fraction': 0.8,
            'bagging_fraction': 1,
            'bagging_freq':6,
            'num_leaves': 100,
            'verbose': -1,
            'lambda_l1': 0,
            'lambda_l2': 0,
            'min_split_gain': 0
}
gbm=lgb.train(params,lgb_train,num_boost_round=300,valid_sets=[lgb_train, lgb_eval],early_stopping_rounds=200,verbose_eval=400)
preds_offline=gbm.predict(x_test, num_iteration=gbm.best_iteration)
print(preds_offline)
result=[1 if i>0.5 else 0 for i in preds_offline]
print(sum(result))
label=pd.DataFrame(result,columns=['label']).replace([1,0],['CTRL','AD'])
submit=pd.concat([test_index,label],axis=1)
submit.to_csv(r'D:\reach\submit_723.csv',index=False)
from catboost import Pool,CatBoostClassifier
train_pool=Pool(x_train1,y_train1)
eval_pool=Pool(x_eval2,y_eval2)
test_pool=Pool(x_test)
model=CatBoostClassifier(iterations=200,depth=10,learning_rate=0.1,loss_function='Logloss',logging_level='Verbose',eval_metric='AUC')
model.fit(train_pool,eval_set=eval_pool,early_stopping_rounds=200)
model.predict(test_pool,prediction_type='Class')





