This repository includes the link for the Chicago Crime Dataset along with the SAS code and the Python code. The repository also includes the Tableau dashboard.
This will take you to the [Dataset](https://atom.io/packages/hyperlink-helper).  

Since the data was categorical we took an approach of converting it to dummy columns for each District, Domestic and Arrest column.

In addition to this, the dataset seemed to be biased for crime happening and not happening so we used an approach of Sampling the dataset to make the variables even.

**Sampling the dataset in Python for Primary_Type = Criminal Damage**
```
#Keeping the required variables for Criminal Damage
df_cd = df[['CD_D', 'Arrest_D', 'Domestic_D', 'District_1', 'District_2', 'District_3', 'District_4', 'District_5', 'District_6', 'District_7', 'District_8', 'District_9',  'District_10', 'District_11', 'District_12', 'District_13', 'District_14', 'District_15', 'District_16', 'District_17', 'District_18', 'District_19', 'District_20', 'District_22', 'District_23', 'District_24', 'District_25']]

#Counts of 0s and 1s before Sampling
print(df_cd['CD_D'].value_counts())
```
**Count of 0s and 1s before Sampling**

| Variables |   Counts   |
| :-------- | :-------   |
|  0        |    2805450 |
|  1        |    695147  |

**Count of 0s and as after Sampling**
```
df_min_down = resample(df_min, replace=True, n_samples=2805450, random_state=123)
df_down=pd.concat([df_maj,df_min_down])
print(df_down['CD_D'].value_counts())
```

| Variables  | Counts    |
| :------------- | :------------- |
|0     | 2805450     |
|1|2805450|


After Sampling the dataset and getting the even number of samples we split the dataset into training and testing dataset in the ratio of 75:25. i.e. 75% into training and 25% in testing datasets.
```
#Split Dataset
y=df_down.CD_D
x = df_down.drop('CD_D', axis=1)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=123)
```

After sub setting the dataset into training and testing for each type of crime we decided to move forward with Logistic Regression and Random Forest Classifier to test the samples.

The following code shows logistic regression with Confusion matrix, precision vs recall curve and AUC curve for Primary_Type = Criminal Damage.
```
#Logistic Regression
def logistic(x_train,y_train,x_test,y_test,x):
    logistic = LogisticRegression().fit(x_train, y_train)
    pred = logistic.predict(x_test)
    print(confusion_matrix(y_test,pred))
    pred_log_prob = logistic.predict_proba(x_test)
    precision,recall, thresholds = precision_recall_curve(y_test,pred_log_prob[:,1])
    print("AUC Score for Logistic Regression for {} is {}".format(df_names[x],(auc(recall,precision))))
    pyplot.plot([0, 1], [0.5, 0.5], linestyle='--')
    pyplot.plot(recall, precision, marker='.')
    pyplot.show()
```

**Results for Logistic Regression**
```
#Analysis
logistic(x_train,y_train,x_test, y_test,0)
```
**Results**

|     |       Predicted     |
| :------------- | :------------- |
|Actual|0     |1       |  
| 0|283590     | 416948     |
|1|79265|622922

**AUC Score for Logistic Regression for Criminal Damage is 0.626**

The following code shows Random Forest Classifier with Confusion matrix, precision vs recall curve and AUC curve for Primary_Type = Criminal Damage.

```
def RandomForest(x_train,y_train,x_test,y_test,x):
    rf = RandomForestClassifier()
    RF_1 = rf.fit(x_train, y_train)
    pred_RF = RF_1.predict(x_test)
    print(confusion_matrix(y_test, pred_RF))
    pred_RF_prob = RF_1.predict_proba(x_test)
    precision,recall, thresholds = precision_recall_curve(y_test,pred_RF_prob[:,1])
    print("AUC Score for Random Forest for {} is {}".format(df_names[x],(auc(recall,precision))))
    pyplot.plot([0, 1], [0.5, 0.5], linestyle='--')
    pyplot.plot(recall, precision, marker='.')
    pyplot.show()
```

**Results for Random Forest**
```
RandomForest(x_train,y_train,x_test, y_test,0)
```
**Results**

|     |       Predicted     |
| :------------- | :------------- |
|Actual|0     |1       |  
| 0|  279219  | 421319  |
|1|  72189 |  629998   |

**AUC Score for Random Forest for Criminal Damage is 0.6333**
