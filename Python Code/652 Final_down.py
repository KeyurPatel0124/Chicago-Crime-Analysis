# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 15:36:55 2019

@author: patel
"""


import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.utils import resample
from sklearn.metrics import recall_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc
from sklearn.metrics import average_precision_score
from matplotlib import pyplot

data = pd.read_csv("C:/Users/patel/Documents/Stevens Assignments/BIA 652/Data_Dummy.csv")
data.head()
df=pd.DataFrame(data)

sam = ['CD_D', 'Arrest_D', 'Domestic_D', 'District_1', 'District_2', 'District_3', 'District_4', 'District_5', 'District_6', 'District_7', 'District_8', 'District_9',  'District_10', 'District_11', 'District_12', 'District_13', 'District_14', 'District_15', 'District_16', 'District_17', 'District_18', 'District_19', 'District_20', 'District_22', 'District_23', 'District_24', 'District_25']
print("This is:",sam[0],sam[1])
q = sam[0]
print(sam)

#Making subset for Criminal Damage
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
####################################################################################################################

#Making subset for Assault
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
############################################################################################################

#Making subset for Narcotics
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
#############################################################################################################################

#Making subset for Robbery
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
##################################################################################################################################

#MAking subset for MVT
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
#################################################################################################################################

#MAking subset for Homicide
df_Hom= df[['Homicide_D', 'Arrest_D', 'Domestic_D', 'District_1', 'District_2', 'District_3', 'District_4', 'District_5', 'District_6', 'District_7', 'District_8', 'District_9',  'District_10', 'District_11', 'District_12', 'District_13', 'District_14', 'District_15', 'District_16', 'District_17', 'District_18', 'District_19', 'District_20', 'District_22', 'District_23', 'District_24', 'District_25']]
print(df_Hom['Homicide_D'].value_counts())

#Split Dataset
y_H=df_Hom.Homicide_D
x_H = df_Hom.drop('Homicide_D', axis=1)
x_train_H,x_test_H,y_train_H,y_test_H=train_test_split(x_H,y_H,test_size=0.25,random_state=123)

#SMOTE analysis
smt=SMOTE()
x_train_sam, y_train_sam=smt.fit_sample(x_train_H, y_train_H)
#########################################################################################################################################

#Making subset for Theft
df_theft= df[['Theft_D', 'Arrest_D', 'Domestic_D', 'District_1', 'District_2', 'District_3', 'District_4', 'District_5', 'District_6', 'District_7', 'District_8', 'District_9',  'District_10', 'District_11', 'District_12', 'District_13', 'District_14', 'District_15', 'District_16', 'District_17', 'District_18', 'District_19', 'District_20', 'District_22', 'District_23', 'District_24', 'District_25']]
print(df_theft['Theft_D'].value_counts())

y_t=df_theft.Theft_D
x_t = df_theft.drop('Theft_D', axis=1)
x_train_t,x_test_t,y_train_t,y_test_t=train_test_split(x_t,y_t,test_size=0.25,random_state=123)


#Function for Logistic Regression
def logisticfn(x_train, y_train, x_test, y_test):
    logistic = LogisticRegression().fit(x_train, y_train)
    probs_l = logistic.predict_proba(x_test)
    probs_l = probs_l[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_test, probs_l)
    print(auc(recall, precision))
    print("AUC Score for {} is {}",sam[0],auc(recall, precision))
    pyplot.plot([0, 1], [0.5, 0.5], linestyle='--')
    pyplot.plot(recall, precision, marker='.')
    pyplot.show()
    
    
#Function for Random Forest
def randomFC(x_train, y_train, x_test, y_test):
    rf = RandomForestClassifier()
    RF_1 = rf.fit(x_train, y_train)
    probs = RF_1.predict_proba(x_test)
    probs = probs[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_test, probs)
    print(auc(recall, precision))
    pyplot.plot([0, 1], [0.5, 0.5], linestyle='--')
    pyplot.plot(recall, precision, marker='.')
    pyplot.show()


print("Logistic for Criminal Damage")
logisticfn(x_train,y_train,x_test,y_test)
print("Logistic for Assault")
logisticfn(x_train_A,y_train_A,x_test_A,y_test_A)
print("Logistic for Narcotics")
logisticfn(x_train_N,y_train_N,x_test_N,y_test_N)
print("Logistic for Robbery")
logisticfn(x_train_R,y_train_R,x_test_R,y_test_R)
print("Logistic for MVT")
logisticfn(x_train_M,y_train_M,x_test_M,y_test_M)
print("Logistic for Theft")
logisticfn(x_train_t,y_train_t,x_test_t,y_test_t)
print("Logistic for Homicide")
logisticfn(x_train_H,y_train_H,x_test_H,y_test_H)


print("Random Forest for Criminal Damage")
randomFC(x_train,y_train,x_test,y_test)
print("Random Forest for Assault")
randomFC(x_train_A,y_train_A,x_test_A,y_test_A)
print("Random Forest for Narcotics")
randomFC(x_train_N,y_train_N,x_test_N,y_test_N)
print("Random Forest for Robbery")
randomFC(x_train_R,y_train_R,x_test_R,y_test_R)
print("Random Forest for MVT")
randomFC(x_train_M,y_train_M,x_test_M,y_test_M)
print("Random Forest for Homicide")
randomFC(x_train_H,y_train_H,x_test_H,y_test_H)
print("Random Forest for Theft")
randomFC(x_train_t,y_train_t,x_test_t,y_test_t)





    
    
