# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 23:30:47 2018

@author: t4nis
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 23:09:23 2018

@author: t4nis
"""


import pandas as pd
import csv
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import resample
from sklearn.metrics import recall_score
from imblearn.over_sampling import SMOTE
from sklearn.metrics import precision_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_score
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve



data = pd.read_csv("C:/Users/t4nis/Desktop/652/Datasets_Chicago/Data_dummy.csv")
df=pd.DataFrame(data)


################################################CD_D#########################################################################
#keep required Variables
df_cd = df[['CD_D', 'Arrest_D', 'Domestic_D', 'District_1', 'District_2', 'District_3', 'District_4', 'District_5', 'District_6', 'District_7', 'District_8', 'District_9',  'District_10', 'District_11', 'District_12', 'District_13', 'District_14', 'District_15', 'District_16', 'District_17', 'District_18', 'District_19', 'District_20', 'District_22', 'District_23', 'District_24', 'District_25']]
print(df_cd['CD_D'].value_counts())
#Up sampling
df_maj = df_cd[df_cd.CD_D==0]
df_min = df_cd[df_cd.CD_D==1]

df_min_down = resample(df_min, replace=True, n_samples=2805450, random_state=123)
df_down=pd.concat([df_maj,df_min_down])
print(df_down['CD_D'].value_counts())


#Split Dataset
y=df_down.CD_D
x = df_down.drop('CD_D', axis=1)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=123)


#Logistic Regression
logistic = LogisticRegression().fit(x_train, y_train)
pred = logistic.predict(x_test)
print(accuracy_score(y_test,pred))
print(confusion_matrix(y_test,pred))
print(cross_val_score(logistic, x, y, scoring='accuracy', cv = 5).mean()*100)
pred_log_prob = logistic.predict_proba(x_test)
precision,recall, thresholds = precision_recall_curve(y_test,pred_log_prob[:,1]) 
print("AUC Score for Logistic")
auc_log_cd=auc(recall,precision)
print("AUC Score for Logistic ",auc_log_cd)


#RF
rf = RandomForestClassifier()
RF_1 = rf.fit(x_train, y_train)
pred_RF = RF_1.predict(x_test)
print(accuracy_score(y_test, pred_RF))
print(confusion_matrix(y_test, pred_RF))
#recall=recall_score(y_test, pred_RF)
#precision=precision_score(y_test, pred_RF)
pred_RF_prob = RF_1.predict_proba(x_test)
precision,recall, thresholds = precision_recall_curve(y_test,pred_RF_prob[:,1]) 
auc=auc(recall,precision)
print(auc)



##########################################################Assault_D##############################################################3
#keep required Variables
df_Assault = df[['Assault_D', 'Arrest_D', 'Domestic_D', 'District_1', 'District_2', 'District_3', 'District_4', 'District_5', 'District_6', 'District_7', 'District_8', 'District_9',  'District_10', 'District_11', 'District_12', 'District_13', 'District_14', 'District_15', 'District_16', 'District_17', 'District_18', 'District_19', 'District_20', 'District_22', 'District_23', 'District_24', 'District_25']]
print(df_Assault['Assault_D'].value_counts())
#Up sampling
df_maj_A = df_Assault[df_Assault.Assault_D==0]
df_min_A = df_Assault[df_Assault.Assault_D==1]

df_min_down_A = resample(df_min_A, replace=True, n_samples=3124629, random_state=123)
df_down_A=pd.concat([ df_maj_A, df_min_down_A,])
print(df_down_A['Assault_D'].value_counts())
#Split Dataset
y_A=df_down_A.Assault_D
x_A = df_down_A.drop('Assault_D', axis=1)
x_train_A,x_test_A,y_train_A,y_test_A=train_test_split(x_A,y_A,test_size=0.25,random_state=123)



#Logistic Regression
logistic_A = LogisticRegression().fit(x_train_A, y_train_A)
pred_A = logistic_A.predict(x_test_A)
print(accuracy_score(y_test_A,pred_A))
print(confusion_matrix(y_test_A,pred_A))
pred_log_prob_A = logistic.predict_proba(x_test)
precision_A,recall_A, thresholds = precision_recall_curve(y_test,pred_log_prob_A[:,1]) 

auc=auc(recall_A,precision_A)
print("AUC Score for Logistic ",auc)







#RF
rf_A = RandomForestClassifier()
RF_1_A = rf_A.fit(x_train_A, y_train_A)
pred_RF_A = RF_1_A.predict(x_test_A)
print(accuracy_score(y_test_A, pred_RF_A))
print(confusion_matrix(y_test_A, pred_RF_A))
recall_score(y_test_A, pred_RF_A)



#######################################################NARCOTICS_D################################################


#keep required Variables
df_Narcotics = df[['Narcotics_D', 'Arrest_D', 'Domestic_D', 'District_1', 'District_2', 'District_3', 'District_4', 'District_5', 'District_6', 'District_7', 'District_8', 'District_9',  'District_10', 'District_11', 'District_12', 'District_13', 'District_14', 'District_15', 'District_16', 'District_17', 'District_18', 'District_19', 'District_20', 'District_22', 'District_23', 'District_24', 'District_25']]
print(df_Narcotics['Narcotics_D'].value_counts())
#Up sampling
df_maj_N = df_Narcotics[df_Narcotics.Narcotics_D==0]
df_min_N = df_Narcotics[df_Narcotics.Narcotics_D==1]

df_min_down_N = resample(df_min_N, replace=True, n_samples=2866212, random_state=123)
df_down_N=pd.concat([ df_maj_N, df_min_down_N,])
print(df_down_N['Narcotics_D'].value_counts())
#Split Dataset
y_N=df_down_N.Narcotics_D
x_N = df_down_N.drop('Narcotics_D', axis=1)
x_train_N,x_test_N,y_train_N,y_test_N=train_test_split(x_N,y_N,test_size=0.25,random_state=123)

#Logistic Regression
logistic_N = LogisticRegression().fit(x_train_N, y_train_N)
pred_N = logistic_N.predict(x_test_N)
print(accuracy_score(y_test_N,pred_N))
print(confusion_matrix(y_test_N,pred_N))
print(cross_val_score(logistic_N, x_N, y_N, scoring='accuracy', cv = 5).mean()*100)

#RF
rf_N = RandomForestClassifier()
RF_1_N = rf_N.fit(x_train_N, y_train_N)
pred_RF_N = RF_1_N.predict(x_test_N)
print(accuracy_score(y_test_N, pred_RF_N))
print(confusion_matrix(y_test_N, pred_RF_N))
recall_score(y_test_N, pred_RF_N)
print(cross_val_score(rf_N, x_N, y_N, scoring='accuracy', cv = 3).mean()*100)


########################################################Robbery_D##########################################################
#keep required Variables
df_Robbery = df[['Robbery_D', 'Arrest_D', 'Domestic_D', 'District_1', 'District_2', 'District_3', 'District_4', 'District_5', 'District_6', 'District_7', 'District_8', 'District_9',  'District_10', 'District_11', 'District_12', 'District_13', 'District_14', 'District_15', 'District_16', 'District_17', 'District_18', 'District_19', 'District_20', 'District_22', 'District_23', 'District_24', 'District_25']]
print(df_Robbery['Robbery_D'].value_counts())

#Up sampling
df_maj_R = df_Robbery[df_Robbery.Robbery_D==0]
df_min_R = df_Robbery[df_Robbery.Robbery_D==1]

df_min_down_R = resample(df_min_R, replace=True, n_samples=3269937, random_state=123)
df_down_R=pd.concat([df_maj_R, df_min_down_R])
print(df_down_R['Robbery_D'].value_counts())

#Split Dataset
y_R=df_down_R.Robbery_D
x_R = df_down_R.drop('Robbery_D', axis=1)
x_train_R,x_test_R,y_train_R,y_test_R=train_test_split(x_R,y_R,test_size=0.25,random_state=123)



#Logistic Regression
logistic_R = LogisticRegression().fit(x_train_R, y_train_R)
pred_R = logistic_R.predict(x_test_R)
print(accuracy_score(y_test_R,pred_R))
print(confusion_matrix(y_test_R,pred_R))


#RF
rf_R = RandomForestClassifier()
RF_1_R = rf_R.fit(x_train_R, y_train_R)
pred_RF_R = RF_1_R.predict(x_test_R)
print(accuracy_score(y_test_R, pred_RF_R))
print(confusion_matrix(y_test_R, pred_RF_R))
recall_score(y_test_R, pred_RF_R)
print(cross_val_score(rf_N, x, y, scoring='accuracy', cv = 5).mean()*100)



######################################################MVT_D #############################################################
#keep required Variables
df_MVT= df[['MVT_D', 'Arrest_D', 'Domestic_D', 'District_1', 'District_2', 'District_3', 'District_4', 'District_5', 'District_6', 'District_7', 'District_8', 'District_9',  'District_10', 'District_11', 'District_12', 'District_13', 'District_14', 'District_15', 'District_16', 'District_17', 'District_18', 'District_19', 'District_20', 'District_22', 'District_23', 'District_24', 'District_25']]
print(df_MVT['MVT_D'].value_counts())

#Up sampling
df_maj_M = df_MVT[df_MVT.MVT_D ==0]
df_min_M = df_MVT[df_MVT.MVT_D ==1]

df_min_down_M = resample(df_min_M, replace=True, n_samples=3222482, random_state=123)
df_down_M=pd.concat([df_maj_M, df_min_down_M])
print(df_down_M['MVT_D'].value_counts())
#Split Dataset
y_M=df_down_M.MVT_D 
x_M = df_down_M.drop('MVT_D', axis=1)
x_train_M,x_test_M,y_train_M,y_test_M=train_test_split(x_M,y_M,test_size=0.25,random_state=123)



#Logistic Regression
logistic_M = LogisticRegression().fit(x_train_M, y_train_M)
pred_M = logistic_M.predict(x_test_M)
print(accuracy_score(y_test_M,pred_M))
print(confusion_matrix(y_test_M,pred_M))


#RF
rf_M = RandomForestClassifier()
RF_1_M = rf_M.fit(x_train_M, y_train_M)
pred_RF_M = RF_1_M.predict(x_test_M)
print(accuracy_score(y_test_M, pred_RF_M))
print(confusion_matrix(y_test_M, pred_RF_M))
recall_score(y_test_M, pred_RF_M)

#########################################Homicide_D####################################################

df_Hom= df[['Homicide_D', 'Arrest_D', 'Domestic_D', 'District_1', 'District_2', 'District_3', 'District_4', 'District_5', 'District_6', 'District_7', 'District_8', 'District_9',  'District_10', 'District_11', 'District_12', 'District_13', 'District_14', 'District_15', 'District_16', 'District_17', 'District_18', 'District_19', 'District_20', 'District_22', 'District_23', 'District_24', 'District_25']]
print(df_Hom['Homicide_D'].value_counts())



#Split Dataset
y_H=df_Hom.Homicide_D 
x_H = df_Hom.drop('Homicide_D', axis=1)
x_train_H,x_test_H,y_train_H,y_test_H=train_test_split(x_H,y_H,test_size=0.25,random_state=123)

#SMOTE analysis
smt=SMOTE()
x_train_sam, y_train_sam=smt.fit_sample(x_train_H, y_train_H)

#Logistic Regression
logistic_H = LogisticRegression().fit(x_train_sam, y_train_sam)
pred_H = logistic_H.predict(x_test_H)
print(accuracy_score(y_test_H,pred_H))
print(confusion_matrix(y_test_H,pred_H))


#RF
rf_H = RandomForestClassifier()
RF_1_H = rf_H.fit(x_train_sam, y_train_sam)
pred_RF_H = RF_1_H.predict(x_test_H)
print(accuracy_score(y_test_H, pred_RF_H))
print(confusion_matrix(y_test_H, pred_RF_H))
recall_score(y_test_H, pred_RF_H)

########################################Theft##########################################################
df_theft= df[['Theft_D', 'Arrest_D', 'Domestic_D', 'District_1', 'District_2', 'District_3', 'District_4', 'District_5', 'District_6', 'District_7', 'District_8', 'District_9',  'District_10', 'District_11', 'District_12', 'District_13', 'District_14', 'District_15', 'District_16', 'District_17', 'District_18', 'District_19', 'District_20', 'District_22', 'District_23', 'District_24', 'District_25']]
print(df_theft['Theft_D'].value_counts())

y_t=df_theft.Theft_D 
x_t = df_theft.drop('Theft_D', axis=1)
x_train_t,x_test_t,y_train_t,y_test_t=train_test_split(x_t,y_t,test_size=0.25,random_state=123)

#Logistic Regression
logistic_t = LogisticRegression().fit(x_train_t, y_train_t)
pred_t = logistic_t.predict(x_test_t)
print(accuracy_score(y_test_t,pred_t))
print(confusion_matrix(y_test_t,pred_t))
#recall_score(y_test_t, pred_t)
#RF
rf_t = RandomForestClassifier()
RF_1_t = rf_t.fit(x_train_t, y_train_t)
pred_RF_t = RF_1_t.predict(x_test_t)
print(accuracy_score(y_test_t, pred_RF_t))
print(confusion_matrix(y_test_t, pred_RF_t))
#recall_score(y_test_t, pred_RF_t)

p_c=precision_score(y_test,pred)
r_c=recall_score(y_test,pred)

p_c_rf=precision_score(y_test,pred_RF)
r_c_rf=recall_score(y_test,pred_RF)

p_a=precision_score(y_test_A,pred_A)
r_a=recall_score(y_test_A,pred_A)

p_a_rf=precision_score(y_test_A,pred_RF_A)
r_a_rf=recall_score(y_test_A,pred_RF_A)

p_n=precision_score(y_test_N,pred_N)
r_n=recall_score(y_test_N,pred_N)

p_n_rf=precision_score(y_test_N,pred_RF_N)
r_n_rf=recall_score(y_test_N,pred_RF_N)

p_r=precision_score(y_test_R,pred_R)
r_r=recall_score(y_test_R,pred_R)

p_r_rf=precision_score(y_test_R, pred_RF_R)
r_r_rf=recall_score(y_test_M,pred_RF_M)

p_m=precision_score(y_test_M,pred_M)
r_m=recall_score(y_test_M,pred_M)

p_m_rf=precision_score(y_test_M,pred_RF_M)
r_m_rf=recall_score(y_test_M,pred_RF_M)

p_t=precision_score(y_test_t,pred_t)
r_t=recall_score(y_test_t,pred_t)

p_t_rf=precision_score(y_test_t,pred_RF_t)
r_t_rf=recall_score(y_test_t,pred_RF_t)