
For our coursework of Multivariate Analysis, our Professor Khasha Dehnad suggested that we analyze crimes in United States. After looking at state government websites we found a dataset for the City of Chicago which had Reported Crimes from 2001 to Present. The dataset is available at this [link](https://catalog.data.gov/dataset/crimes-2001-to-present-398a4).

The dataset has over 6.7 million rows and 22 variables. Since the dataset only has Categorical variables we couldn't apply any linear machine learning application to predict. We decided to appply Logistic Regression for this problem to find out the probability of a crime occurring.

Influential variables includes

Primary_type:- Its the type of crime being reported. For eg. Theft, Robbery, Assault

Arrest :- A boolean value for which an Arrest is made for the reported crimes

Domestic :- A boolean value for if the reported crime is a domestic offense or not

District(Target Variable) :- Categorical value which corresponds to the police district in Chicago

**Data Pre-Processing**

We subset part of the data and analyze it on Excel. We notice that there are some null values in district, Location, X_coordinate, Y_coordinate, Ward, Community_area and District 21 & 31 have less than 200 values which make it useless for this problem. We remove these null values in SAS

    ```
    data Clean;
    	set TestTime;
    	if district = . or Location = '' or X_coordinate = . or Y_coordinate = .
    	or Ward = . or Community_area = . or District = 21 or District = 31 then delete;
    run;

    ```

  The Date variable also has the time part in it. We split that variable to separate Date and Time.

      ```
      data TestTime;
        set Data;
        Date_r=datepart(Date);
        Time_r=timepart(Date);
        format Date_r yymmdd10. Time_r time20.;
      run;
      ```

   We then subset the dataset for Major Offenses. We picked the most recurring crimes.

   ```
   data Data_major_offenses;
   	set Clean;
	  if Primary_type = 'ASSAULT' OR Primary_type ='NARCOTICS OR Primary_type ='ROBBERY' OR Primary_type ='THEFT' OR Primary_type ='MOTOR VEHICLE THEFT' OR Primary_type ='CRIMINAL DAMAGE' OR Primary_type = 'HOMICIDE';
   run;
   ```

   To work with categorical data we need to convert them into dummy variables.

   ```
   /*Creating Dummy variables for District_1-Dictrict_25*/
   DATA Data_Dummy;
     set Data_major_offenses;

     ARRAY dummys {*} 3.  District_1 - District_25;
     DO i=1 TO 25;			      
       dummys(i) = 0;
     END;
     dummys(District) = 1;		
   RUN;

   /*Create Dummy variables for Primary_Type, Arrest and Domestic*/
   data data_dummy;
   	set data_dummy;
   	if Primary_type = 'ASSAULT' then Assault_D = 1;
   	else Assault_D = 0;

   	if Primary_type = 'NARCOTICS' then Narcotics_D = 1;
   	else Narcotics_D = 0;

   	if Primary_type = 'THEFT' then Theft_D = 1;
   	else Theft_D = 0;

   	if Primary_type = 'HOMICIDE' then Homicide_D = 1;
   	else Homicide_D = 0;

   	if Primary_type = 'CRIMINAL DAMAGE' then CD_D = 1;
   	else CD_D = 0;

   	if Primary_type = 'ROBBERY' then Robbery_D = 1;
   	else Robbery_D = 0;

   	if Primary_type = 'MOTOR VEHICLE THEFT' then MVT_D = 1;
   	else MVT_D = 0;

   	if Arrest = "true" then Arrest_D=1;
   	else Arrest_D=0;

   	if Domestic = "true" then Domestic_D=1;
   	else Domestic_D=0;

   	run;

   ```
We then moved onto analyzing the data. Since the dataset is huge and we decided to work on subsets of the data to the full dataset, we had a doubt that this would be an imbalanced class problem.

To confirm our suspicion we run **proc freq** on Criminal Damage

| CD_D | Frequency | Percent | CumulativeFrequency | CumulativePercent |
|------|-----------|---------|---------------------|-------------------|
| 0    | 2805450   | 80.14   | 2805450             | 80.14             |
| 1    | 695147    | 19.86   | 3500597             | 100.00            |

From the 80-20 ratio we can confirm our suspicions.

Application of any machine learning algorithms will give us an output with very high accuracy, which we know is biased towards 0.
To counter this, we need to apply sampling technique. We used Downsampling in SAS.

  ```
  proc surveyselect data = data_dummy1 out = cd method = srs sampsize=(695147,695147) seed = 9876;
  	strata CD_D;
  run;

  /*Output of Downsampling*/
  proc freq data=cd;
  	tables CD_D;
  run;
  ```

**Output**

  | CD_D | Frequency | Percent | CumulativeFrequency | CumulativePercent |
|------|-----------|---------|---------------------|-------------------|
| 0    | 695147    | 50.00   | 695147              | 50.00             |
| 1    | 695147    | 50.00   | 1390294             | 100.00            |

From the output we can see that the subset has been successfully downsampled and we can go on to apply Logistic Regression.
