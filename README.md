# projectRossman
# Project Summary -
The objective of the Rossmann Sales Prediction project is to develop a predictive model that accurately forecasts daily sales for Rossmann stores. Accurate sales predictions can help in 
optimizing inventory management, staffing, and other operational aspects to enhance business efficiency and profitability.
# Problem Statement
Rossmann operates over 3,000 drug stores in 7 European countries. Currently, Rossmann store managers are tasked with predicting their daily sales for up to six weeks in advance. Store 
sales are influenced by many factors, including promotions, competition, school and state holidays, seasonality, and locality. With thousands of individual managers predicting sales 
based on their unique circumstances, the accuracy of results can be quite varied. You are provided with historical sales data for 1,115 Rossmann stores. The task is to forecast the 
"Sales" column for the test set. Note that some stores in the dataset were temporarily closed for refurbishment.
# Know your data
We are having 2 tables Store table and Rossmann store data table. Store table have data which affect the sales of store. Rossmann store table having sales data of store. 
Number of rows in store table are 1115 and having 10 columns. Number of rows in Rossmann store data table are 1017209 and having 9 columns. Store table having columns which 
have null values and data types used are int,object,float. Rossmann store data table have no null values and data type used are int,object. 
There are no duplicated values in both tables. Store table column having maximum 544 null values from 1017209 values. Rossmann store data table having no null values.
# Understanding Your Variables
Index(['Store', 'StoreType', 'Assortment', 'CompetitionDistance',
       'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear', 'Promo2',
       'Promo2SinceWeek', 'Promo2SinceYear', 'PromoInterval'],
      dtype='object')
Index(['Store', 'DayOfWeek', 'Date', 'Sales', 'Customers', 'Open', 'Promo',
       'StateHoliday', 'SchoolHoliday'],
      dtype='object')
# Variables Description
Store Store Number, StoreType Type of Store, Assortment ** Mix of items retailer sale, **CompetitionDistance Competative shop near to this shop, CompetitionOpenSinceMonth Competative shop open month, CompetitionOpenSinceYear Competative shop open year, Promo2 Promotion held or not, Promo2SinceWeek ** Promotion started week, **Promo2SinceYear Promotion started year, PromoInterval Promotion held on month,

Store Type of Store, DayOfWeek Number of days in week shop opens, Date Sales value Date, Sales Sales at Date, Customers Number of customer, Open Whether shop open or close, Promo Promotion held or not, StateHoliday Whether state holiday or not, SchoolHoliday Whether school holiday or not.

Id - an Id that represents a (Store, Date) duple within the test set

Store - a unique Id for each store

Sales - the turnover for any given day (this is what you are predicting)

Customers - the number of customers on a given day

Open - an indicator for whether the store was open: 0 = closed, 1 = open

StateHoliday - indicates a state holiday. Normally all stores, with few exceptions, are closed on state holidays. Note that all schools are closed on public holidays and weekends. a = public holiday, b = Easter holiday, c = Christmas, 0 = None

SchoolHoliday - indicates if the (Store, Date) was affected by the closure of public schools

StoreType - differentiates between 4 different store models: a, b, c, d

Assortment - describes an assortment level: a = basic, b = extra, c = extended

CompetitionDistance - distance in meters to the nearest competitor store

CompetitionOpenSince[Month/Year] - gives the approximate year and month of the time the nearest competitor was opened

Promo - indicates whether a store is running a promo on that day

Promo2 - Promo2 is a continuing and consecutive promotion for some stores: 0 = store is not participating, 1 = store is participating

Promo2Since[Year/Week] - describes the year and calendar week when the store started participating in Promo2

PromoInterval - describes the consecutive intervals Promo2 is started, naming the months the promotion is started anew. E.g. "Feb,May,Aug,Nov" means each round starts in February, May, August, November of any given year for that store

# Hypothesis Testing
1. Null Hypothesis (H0): There is no significant difference in sales between stores with longer promotion durations and those with shorter promotion durations.

Alternative Hypothesis (H1): Stores with longer promotion durations have significantly higher sales than those with shorter promotion durations.
2.Null Hypothesis (H0): Stores with higher sales have no significant difference in competitor presence compared to stores with lower sales.

Alternative Hypothesis (H1): Stores with higher sales have significantly different competitor presence compared to stores with lower sales.
3. Null Hypothesis (H0): There is no significant difference in sales between stores with a larger assortment of products and those with a smaller assortment.

Alternative Hypothesis (H1): Stores with a larger assortment of pro![sales](https://github.com/BitanHazra/projectRossman/assets/46105628/c2f52e8b-45de-4489-b78a-de268a670d56)
ducts have significantly different sales compared to those with a smaller assortment.
# handling missing
Promo2SinceWeek , Promo2SinceYear , PromoInterval:Missing values are filled with 0. We cannot fill Nan values with mean or standard deviation as it can effect our result. Promo directly affect our sales data so it better to fill it with 0.

CompetitionOpenSinceMonth','CompetitionOpenSinceYear:Store whose CompetitionDistance is 0 we fill Nan values with 0 as if there is no competetor means no month value and no year value.

CompetitionOpenSinceMonth','CompetitionOpenSinceYear: Store whose CompetitionDistance is not 0 we fill Nan values with mean as there is a competition but forget to inset the values.
# categorical encoding
StateHoliday, StoreType, Assortment, PromoInterval :This columns values are code with 0,1,2,3.

# Which ML model did you choose from the above created models as your final prediction model and why?
Answer Here.

Random Forest Regressor Random Forest Regressor model choose as the best model as its Mean Squared Error for Random Forest Regressor is 0.07081097203680194
![reg](https://github.com/BitanHazra/projectRossman/assets/46105628/96bdeed9-c1db-45c4-ac7c-a1b19cf9bdba)
![promo2](https://github.com/BitanHazra/projectRossman/assets/46105628/3651fa6b-4362-4b4a-9e32-7d6c0608f28e)
![page1](https://github.com/BitanHazra/projectRossman/assets/46105628/b05bef7c-a7ea-4d1f-ac02-2929c991ceb8)
![outliers](https://github.com/BitanHazra/projectRossman/assets/46105628/046fc6fc-484f-455d-8e0a-2acf9cdcb840)
![heatmapmissing](https://github.com/BitanHazra/projectRossman/assets/46105628/b9776462-0181-4983-b015-d5a2120ae538)
![dsfds](https://github.com/BitanHazra/projectRossman/assets/46105628/003cb22d-c889-4464-a733-a0f103634b4c)
![describe](https://github.com/BitanHazra/projectRossman/assets/46105628/f6153612-e3e1-4684-b072-95782cbf2c75)
![datadesc](https://github.com/BitanHazra/projectRossman/assets/46105628/fd94d15c-f701-4329-9d07-59447cbd44c7)
![avgsalespromo](https://github.com/BitanHazra/projectRossman/assets/46105628/c4ad0082-4104-4bb1-b1f8-357efee655e2)
![avgsales](https://github.com/BitanHazra/projectRossman/assets/46105628/983cff3a-a860-4a16-b6d4-cded274bcc70)
