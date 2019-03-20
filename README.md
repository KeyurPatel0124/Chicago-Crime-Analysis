
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
