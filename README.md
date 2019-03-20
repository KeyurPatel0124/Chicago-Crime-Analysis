
This repository includes the link for the Chicago Crime Dataset along with the SAS code and the Python code. The repository also includes the Tableau dashboard.
This will take you to the [Dataset](https://atom.io/packages/hyperlink-helper).  

Since the data was categorical we took a approach of converting it to dummy columns for each District, Domestic and Arrest column.

In addition to this, the dataset seemed to be biased for crime happening and not happening so we used an approach of Sampling the dataset to make the variables even.

Following is the code for down sampling the dataset in SAS:

**Sample Code**

```/*Down-Sampling Data for Criminal Damage*/
Title "Sampling and Logistic for Criminal Damage Train to Test";
proc sort data =data_dummy out=data_dummy1;
	by CD_D;
run;
proc freq data=data_dummy;
	Tables CD_D;
run;
proc surveyselect data = data_dummy1 out = cd method = srs sampsize=(695147,695147) seed = 9876;
	strata CD_D;
run;
proc freq data=cd;
	tables CD_D;
run;
```
