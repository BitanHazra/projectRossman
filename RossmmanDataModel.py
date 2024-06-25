# Import Libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Load Dataset
store_data = pd.read_csv('/content/sample_data/store.csv')
r_store_data = pd.read_csv('/content/sample_data/Rossmann Stores Data.csv')

# Dataset First Look
store_data.head()

r_store_data.head()

r_store_data[r_store_data['Open']==0]

r_store_data['Store'].unique()

# Dataset Rows & Columns count
print(f'Numbers of rows in store data is {store_data.shape[0]} and columns is {store_data.shape[1]}')
print(f'Numbers of rows in Rossmann store data is {r_store_data.shape[0]} and columns is {r_store_data.shape[1]}')

# Dataset Info
store_data.info()

r_store_data.info()

# Dataset Duplicate Value Count
print(f'Dataset store duplicated values= {store_data.duplicated().sum()}')
print(f'Dataset Rossmann store Data duplicated values= {r_store_data.duplicated().sum()}')

# Missing Values/Null Values Count
store_data.isna().sum()

r_store_data.isna().sum()

# Visualizing the missing values
#Visualizing missing values in a dataset can help you understand the extent and pattern of the missing data.
plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
sb.heatmap(store_data.isna(),cbar=False, cmap='viridis', yticklabels=False)
plt.title('Heatmap of Missing Values')
plt.subplot(1,2,2)
sb.heatmap(r_store_data.isna(),cbar=False, cmap='viridis', yticklabels=False)
plt.title('Heatmap of Missing Values')
plt.show()

# Dataset Columns
print(store_data.columns)
print(r_store_data.columns)

# Dataset Describe
store_data.describe()

r_store_data.describe()

# Check Unique Values for each variable.
store_data['Store'].unique()
store_data['StoreType'].unique()
store_data['Assortment'].unique()
store_data['CompetitionDistance'].unique()
store_data['CompetitionOpenSinceMonth'].unique()
store_data['CompetitionOpenSinceYear'].unique()
store_data['Promo2'].unique()
store_data['Promo2SinceWeek'].unique()
store_data['Promo2SinceYear'].unique()
store_data['PromoInterval'].unique()
r_store_data['Store'].unique()
r_store_data['DayOfWeek'].unique()
r_store_data['Sales'].unique()
r_store_data['Customers'].unique()

# knowing the data types
store_data.dtypes
# knowing the data types
r_store_data.dtypes
# knowing the column names
store_data.columns
# knowing the column names
r_store_data.columns
print(store_data.isna().sum())
print('------'*20)
print(f'Percentage of missing values\n {store_data.isnull().mean()*100}')
print(r_store_data.isna().sum())
print('------'*20)
print(f'Percentage of missing values\n {r_store_data.isnull().mean()*100}')

plt.figure(figsize=(6,4))
for index, column in enumerate(r_store_data[['Sales','Customers']]):
  plt.subplot(1,2,index+1)
  sb.boxplot(r_store_data[column])
  plt.xlabel(column)
  plt.ylabel('Values')
  plt.show()
plt.figure(figsize=(6,4))
for index, column in enumerate(store_data[['CompetitionDistance']]):
  plt.subplot(1,2,index+1)
  sb.boxplot(store_data[column])
  plt.xlabel(column)
  plt.ylabel('Values')
  plt.show()

r_store_data['StateHoliday'].replace(0,'0',inplace=True)
r_store_data['Date'] = pd.to_datetime(r_store_data['Date'])
r_store_data['Year'] =  r_store_data['Date'].dt.year
r_store_data['month'] =  r_store_data['Date'].dt.month
r_store_data = r_store_data[r_store_data['Open']==1]
r_store_data

#merge table
merge_df = pd.merge(r_store_data,store_data,how='left',on='Store')
merge_df

# Chart - 1 visualization code
avg_dayofweek_sales = r_store_data.groupby(['DayOfWeek'])['Sales'].mean().reset_index()
plt.figure(figsize=(5,4))
sb.barplot(x='DayOfWeek',y='Sales',data=avg_dayofweek_sales,label='avg sales')
plt.title('Average sales for day of week')
plt.show()

# Chart - 2 visualization code Sales	Promo	StateHoliday	SchoolHoliday	Year	month
promo_data = r_store_data[r_store_data['Promo'] == 1]
promo_data = promo_data.groupby(['Year', 'month'])['Sales'].mean().reset_index()
no_promo_data = r_store_data[r_store_data['Promo'] == 0]
no_promo_data = no_promo_data.groupby(['Year', 'month'])['Sales'].mean().reset_index()

plt.figure(figsize=(25, 6))

unique_years = no_promo_data['Year'].unique()

for index, year in enumerate(unique_years):
    filtered_no_promo = no_promo_data[no_promo_data['Year'] == year]
    filtered_promo = promo_data[promo_data['Year'] == year]

    ax = plt.subplot(1, len(unique_years), index + 1)

    # Set the positions of the bars
    bar_width = 0.4
    months = filtered_no_promo['month']
    no_promo_positions = range(len(months))
    promo_positions = [x + bar_width for x in no_promo_positions]

    # Plot the no promotion data
    ax.bar(no_promo_positions, filtered_no_promo['Sales'], width=bar_width, color='red', label='No Promo')

    # Plot the promotion data
    ax.bar(promo_positions, filtered_promo['Sales'], width=bar_width, color='blue', label='Promo')

    # Set the x-axis labels to be the month names
    ax.set_xticks([r + bar_width / 2 for r in no_promo_positions])
    ax.set_xticklabels(months)

    ax.set_title(f"Average Sales Data for {year}")
    ax.legend()

plt.tight_layout()
plt.show()

# Chart - 3 visualization code
state_holiday = r_store_data.groupby(['StateHoliday'])['Sales'].mean().reset_index()
state_holiday['StateHoliday'].replace({'0':'None','a':'public holiday','b':'Easter holiday','c':'Christmas'},inplace=True)
plt.figure(figsize=(4,4))
sb.barplot(x='StateHoliday',y='Sales',data=state_holiday,label='Sales')
plt.legend()
plt.xticks(rotation=45)
plt.ylabel('Average Sales')
plt.title('Average Sales in each State holiday')
plt.show()

# Chart - 4 visualization code
school_holiday = r_store_data.groupby(['SchoolHoliday'])['Sales'].mean().reset_index()
school_holiday['SchoolHoliday'].replace({0:'No Holiday',1:'holiday'},inplace=True)
plt.figure(figsize=(4,4))
sb.barplot(x='SchoolHoliday',y='Sales',data=school_holiday,label='Sales')
plt.legend()
plt.xticks(rotation=45)
plt.ylabel('Average Sales')
plt.title('Average Sales in each School holiday')
plt.show()

# Chart - 5 visualization code
storeType = merge_df.groupby(['StoreType'])['Sales'].mean().reset_index()
plt.figure(figsize=(4,4))
sb.barplot(x='StoreType',y='Sales',data=storeType,label='Sales')
plt.legend()
plt.ylabel('Average Sales')
plt.title('Average sale in each Store Type')
plt.show()

# Chart - 6 visualization code
count_comp = merge_df.groupby(['StoreType'])['CompetitionDistance'].count().reset_index()
plt.figure(figsize=(4,4))
sb.barplot(x='StoreType',y='CompetitionDistance',data=count_comp,label='Sales')
plt.legend()
plt.ylabel('Number of competeror')
plt.title('Number of competeter in each store type')
plt.show()

# Chart - 7 visualization code
count_comp = merge_df.groupby(['Assortment'])['Sales'].count().reset_index()
plt.figure(figsize=(4,4))
sb.barplot(x='Assortment',y='Sales',data=count_comp,label='Sales')
plt.legend()
plt.ylabel('Sales count')
plt.title('Assortment')
plt.show()

# Chart - 8 visualization code
promo_data = merge_df[merge_df['Promo2'] == 1]
promo_data = promo_data.groupby(['Year', 'month'])['Sales'].mean().reset_index()
no_promo_data = merge_df[merge_df['Promo2'] == 0]
no_promo_data = no_promo_data.groupby(['Year', 'month'])['Sales'].mean().reset_index()

plt.figure(figsize=(25, 6))

unique_years = no_promo_data['Year'].unique()

for index, year in enumerate(unique_years):
    filtered_no_promo = no_promo_data[no_promo_data['Year'] == year]
    filtered_promo = promo_data[promo_data['Year'] == year]

    ax = plt.subplot(1, len(unique_years), index + 1)

    # Set the positions of the bars
    bar_width = 0.4
    months = filtered_no_promo['month']
    no_promo_positions = range(len(months))
    promo_positions = [x + bar_width for x in no_promo_positions]

    # Plot the no promotion data
    ax.bar(no_promo_positions, filtered_no_promo['Sales'], width=bar_width, color='red', label='No Promo2')

    # Plot the promotion data
    ax.bar(promo_positions, filtered_promo['Sales'], width=bar_width, color='blue', label='Promo2')

    # Set the x-axis labels to be the month names
    ax.set_xticks([r + bar_width / 2 for r in no_promo_positions])
    ax.set_xticklabels(months)

    ax.set_title(f"Average Sales Data for {year}")
    ax.legend()

plt.tight_layout()
plt.show()

# Chart - 9 visualization code
mean_sales_promo2 = merge_df.groupby(['Promo2SinceWeek'])['Sales'].mean().reset_index()
plt.figure(figsize=(15,8))
sb.barplot(x='Promo2SinceWeek',y='Sales',data=mean_sales_promo2)
plt.title('Average sales Promo2SinceWeek')
plt.xlabel('Promo2 Since Week')
plt.ylabel('Sales')
plt.show()


from scipy import stats

median_promo_duration = merge_df['Promo2SinceWeek'].median()

# Subset data
long_promo_sales = merge_df[merge_df['Promo2SinceWeek'] > median_promo_duration]['Sales']
short_promo_sales = merge_df[merge_df['Promo2SinceWeek'] <= median_promo_duration]['Sales']

# Calculate means and standard deviations
long_promo_mean = long_promo_sales.mean()
short_promo_mean = short_promo_sales.mean()
long_promo_std = long_promo_sales.std()
short_promo_std = short_promo_sales.std()

# Sample sizes
n_long_promo = len(long_promo_sales)
n_short_promo = len(short_promo_sales)

# Calculate the z-statistic
z_stat = (long_promo_mean - short_promo_mean) / np.sqrt((long_promo_std**2 / n_long_promo) + (short_promo_std**2 / n_short_promo))

# Calculate the p-value for a one-tailed test
p_value = 1 - stats.norm.cdf(z_stat)

# Print results
print(f"Z-statistic: {z_stat}")
print(f"P-value: {p_value}")

# Decision
alpha = 0.05
if p_value < alpha:
    print("Reject the null hypothesis.")
else:
    print("Fail to reject the null hypothesis.")


from scipy import stats

median_sales = merge_df['Sales'].median()

# Subset data
high_sales_competitors = merge_df[merge_df['Sales'] > median_sales]['CompetitionDistance']
low_sales_competitors = merge_df[merge_df['Sales'] <= median_sales]['CompetitionDistance']

# Calculate means and standard deviations
high_sales_mean = high_sales_competitors.mean()
low_sales_mean = low_sales_competitors.mean()
high_sales_std = high_sales_competitors.std()
low_sales_std = low_sales_competitors.std()

# Sample sizes
n_high_sales = len(high_sales_competitors)
n_low_sales = len(low_sales_competitors)

# Calculate the z-statistic
z_stat = (high_sales_mean - low_sales_mean) / np.sqrt((high_sales_std**2 / n_high_sales) + (low_sales_std**2 / n_low_sales))

# Calculate the p-value for a two-tailed test
p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

# Print results
print(f"Z-statistic: {z_stat}")
print(f"P-value: {p_value}")

# Decision
alpha = 0.05
if p_value < alpha:
    print("Reject the null hypothesis.")
else:
    print("Fail to reject the null hypothesis.")

from scipy import stats

# Subset data
high_sales = merge_df[merge_df['StoreType'] == 'a']['Sales']
low_sales = merge_df[merge_df['StoreType'] == 'b']['Sales']

# Calculate means and standard deviations
high_sales_mean = high_sales.mean()
low_sales_mean = low_sales.mean()
high_sales_std = high_sales.std()
low_sales_std = low_sales.std()

# Sample sizes
n_high_sales = len(high_sales)
n_low_sales = len(low_sales)

# Calculate the z-statistic
z_stat = (high_sales_mean - low_sales_mean) / np.sqrt((high_sales_std**2 / n_high_sales) + (low_sales_std**2 / n_low_sales))

# Calculate the p-value for a two-tailed test
p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

# Print results
print(f"Z-statistic: {z_stat}")
print(f"P-value: {p_value}")

# Decision
alpha = 0.05
if p_value < alpha:
    print("Reject the null hypothesis.")
else:
    print("Fail to reject the null hypothesis.")

# Handling Missing Values & Missing Value Imputation
merge_df['Promo2SinceWeek'].fillna(0,inplace=True)
merge_df['Promo2SinceYear'].fillna(0,inplace=True)
merge_df['PromoInterval'].fillna(0,inplace=True)

merge_df['CompetitionDistance'].fillna(0,inplace=True)
merge_df[merge_df['CompetitionDistance']==0][['CompetitionOpenSinceMonth','CompetitionOpenSinceYear']].fillna(0,inplace=True)
merge_df['CompetitionOpenSinceMonth'].fillna(merge_df['CompetitionOpenSinceMonth'].mean(),inplace=True)
merge_df['CompetitionOpenSinceYear'].fillna(merge_df['CompetitionOpenSinceYear'].mean(),inplace=True)

# Encode your categorical columns
merge_df['StateHoliday'] = merge_df['StateHoliday'].map({'0':0,'a':1,'b':2,'c':3})
merge_df['StoreType'] = merge_df['StoreType'].map({'a':1,'b':2,'c':3,'d':4})
merge_df['Assortment'] = merge_df['Assortment'].map({'a':1,'b':2,'c':3})
merge_df['PromoInterval'] = merge_df['PromoInterval'].map({0:0, 'Jan,Apr,Jul,Oct':1, 'Feb,May,Aug,Nov':2, 'Mar,Jun,Sept,Dec':3})

# Select your features wisely to avoid overfitting
merge_df.drop(columns=['Promo2', 'Promo2SinceWeek', 'CompetitionDistance', 'CompetitionOpenSinceMonth'], inplace=True)
merge_df['day'] = merge_df['Date'].dt.day
merge_df.drop(columns=['Date'], inplace=True)

# Scaling your data
merge_df['CompetitionOpenSinceYear'] = merge_df['CompetitionOpenSinceYear'].astype('int64')
merge_df['Promo2SinceYear'] = merge_df['Promo2SinceYear'].astype('int64')
from scipy import stats
merge_df['Sales_log'],_ = stats.boxcox(merge_df['Sales']+1)
merge_df['Customers_log'],_ = stats.boxcox(merge_df['Customers']+1)
merge_df.drop(columns=['Sales','Customers'],inplace=True)
merge_df.drop(columns=['Year'],inplace=True)
merge_df.drop(columns=['Open'],inplace=True)
merge_df

X = merge_df.drop(columns=['Sales_log'])
Y = merge_df['Sales_log']
from sklearn.model_selection import train_test_split
train_x,test_x,train_y,test_y = train_test_split(X,Y,test_size=0.2,random_state=42)


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,accuracy_score,r2_score,mean_absolute_error
l_model = LinearRegression()
l_model.fit(train_x,train_y)
pred_y = l_model.predict(test_x)
mse = mean_squared_error(test_y, pred_y)

r2 = r2_score(test_y, pred_y)

mae = mean_absolute_error(test_y, pred_y)

rmse = np.sqrt(mse)

print(f'Mean Squared Error for LinearRegression is {mse}')
print(f'R-squared for LinearRegression is {r2}')
print(f'Mean Absolute Error for LinearRegression is {mae}')
print(f'Root Mean Squared Error for LinearRegression is {rmse}')


results_df = pd.DataFrame({'Actual': test_y, 'Predicted': pred_y})

plt.figure(figsize=(10, 6))

sb.scatterplot(x='Actual', y='Predicted', data=results_df)
plt.plot([results_df['Actual'].min(), results_df['Actual'].max()],
         [results_df['Actual'].min(), results_df['Actual'].max()],
         color='red', lw=2)

plt.title('Actual vs Predicted Values')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.show()


from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression

l_model = LinearRegression()

k = 5
kf = KFold(n_splits=k, random_state=42, shuffle=True)

cv_results = cross_val_score(l_model, X, Y, cv=kf, scoring='r2')

print(f'Cross-Validation R-squared scores: {cv_results}')
print(f'Mean R-squared: {cv_results.mean()}')
print(f'Standard Deviation of R-squared: {cv_results.std()}')

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

rf_model = RandomForestRegressor(random_state=42)

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, scoring='r2', n_jobs=-1)

grid_search.fit(X, Y)

best_params = grid_search.best_params_
best_score = grid_search.best_score_

print(f'Best Hyperparameters: {best_params}')
print(f'Best R-squared Score: {best_score}')

# Visualizing evaluation Metric Score chart
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(train_x,train_y)
pred_y = rf_model.predict(test_x)
mse = mean_squared_error(test_y, pred_y)
r2 = r2_score(test_y, pred_y)
mae = mean_absolute_error(test_y, pred_y)
rmse = np.sqrt(mse)

print(f'Mean Squared Error for Random Forest Regressor is {mse}')
print(f'R-squared for Random Forest Regressor is {r2}')
print(f'Mean Absolute Error for Random Forest Regressor is {mae}')
print(f'Root Mean Squared Error for Random Forest Regressor is {rmse}')

from sklearn.tree import DecisionTreeRegressor
dt_model = DecisionTreeRegressor(random_state=42)

dt_model.fit(train_x, train_y)

pred_y = dt_model.predict(test_x)

mse = mean_squared_error(test_y, pred_y)
r2 = r2_score(test_y, pred_y)
mae = mean_absolute_error(test_y, pred_y)
rmse = mean_squared_error(test_y, pred_y, squared=False)


print(f'Mean Squared Error (MSE): {mse:.2f}')
print(f'R-squared (R2): {r2:.2f}')
print(f'Mean Absolute Error (MAE): {mae:.2f}')
print(f'Root Mean Squared Error (RMSE): {rmse:.2f}')
