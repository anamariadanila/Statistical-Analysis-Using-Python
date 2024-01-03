import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.stats.api as sms
import statsmodels.tsa.api as smt
from scipy.stats import pearsonr


# 2.1 Preliminary and transformation operations on variables

#importing the dataset
house_offers = pd.read_csv('house_offers.csv')


#show the dataset
#print(house_offers)

#verify the type of the variable house_offers
print(type(house_offers))


#create a categorical variable with at least 3 categories from a numeric variable
#in this case I chose the variable 'price' and I created the variable 'price_range' which contains the following categories: 'small', 'medium', 'large'
#the categories were created according to the minimum value, the first quartile, the third quartile and the maximum value of the variable 'price'
c = house_offers['price'].describe()
print(c)

house_offers['price_range'] = pd.cut(house_offers['price'], 
                                     bins=[c['min'] - 1 , c['25%'], c['75%'], c['max'] + 1],
                                     labels=['mic', 'mediu', 'mare'])


#selection of a sample of data from the dataset
#the variables after which the selection is made are 'price' and 'bathrooms_count'
number_of_rows = house_offers.count()[0]
print('Initial number of rows: ', number_of_rows)
house_offers = house_offers[(house_offers['price'] <= 1000000) & (house_offers['bathrooms_count'] >= 1)]
number_of_rows = house_offers.count()[0]
print('Number of rows after selection: ', number_of_rows)


#drop the variables that are not relevant for the analysis
#all variables that are in the dataset
initial_columns = list(house_offers)
print('Initial variables: ', initial_columns)

#select the columns that are not relevant for the analysis
columns_to_drop = ['id', 'location', 'location_area', 'seller_type', 'type', 'comfort', 'construction_year', 
                   'real_estate_type', 'height_regime', 'level', 'max_level', 'kitchens_count',
                   'garages_count', 'parking_lots_count', 'balconies_count']

#drop the columns that are not relevant for the analysis
house_offers = house_offers.drop(columns_to_drop, axis=1)

#check the columns that are left in the dataset
final_columns = list(house_offers)
print('Final variables: ', final_columns)

#verify the type of data for each variable in the dataset
print(house_offers.dtypes)

#change the type of the string variables to factor type variables
#the variable 'partitioning' has the type 'object' and it has to be changed
house_offers['partitioning'] = house_offers['partitioning'].astype('category')

#even if the variable 'price_range' is created by using the function 'cut', let's check its type
print(house_offers['price_range'].dtypes)

#check again the type of data for each variable in the dataset
print(house_offers.dtypes)

#check the categories of the variable 'partitioning'
print(house_offers['partitioning'].value_counts())

#check the categories of the variable 'price_range'
print(house_offers['price_range'].value_counts())

#check if there are NaN values in the dataset
print(house_offers.isna().sum())

#check if exists NULL values in the dataset
print (house_offers.isnull().sum())

#eliminate the NaN values from the dataset
house_offers = house_offers.dropna()

#check again if there are NaN values in the dataset
print(house_offers.isna().sum())

#check again if exists NULL values in the dataset
print (house_offers.isnull().sum())

#export the new dataset to a csv file
#house_offers.to_csv('house_offers_clean.csv', index=False)

# 2.2 Database description

print("The number of observations in the dataset is : ", len(house_offers), '\n')
print('The name of the variables in the dataset are: ', list(house_offers), '\n')
print('The first 5 rows of the dataset are: ', house_offers.head(), '\n')
print('Check if there are NaN values in the dataset: ', house_offers.isna().sum(), '\n')
print('Check if exists NULL values in the dataset: ', house_offers.isnull().sum(), '\n')
print('The factory type variables in the dataset are: ', house_offers.select_dtypes(include=['category']).columns, '\n')
print('The categories of the variable "partitioning" are: ', house_offers['partitioning'].value_counts(), '\n')
print('The categories of the variable "price_range" are: ', house_offers['price_range'].value_counts(), '\n')


# 3. Graphical and numerical analysis of the variables analysed

# 3.1 descriptive analysis of numeric and non-numeric variables

#create a new dataset with the numeric variables
house_offers_numeric = house_offers.select_dtypes(include=['int64', 'float64'])
#show the numeric dataset
print(house_offers_numeric)

#analyze the numeric variables
print('Analysis of the numeric variables: ', '\n')
print(house_offers_numeric.describe())
print('\n')
#median analysis
print(house_offers_numeric.median())
print('\n')
#skew analysis
print(house_offers_numeric.skew())
print('\n')
#kurtosis analysis
print(house_offers_numeric.kurtosis())
print('\n')



#create a new dataset with the non-numeric variables
house_offers_non_numeric = house_offers.select_dtypes(include=['category'])

# #show the non-numeric dataset
print(house_offers_non_numeric)

print('Analysis of the non-numeric variables: ', '\n')
partition_analysis = pd.crosstab(house_offers_non_numeric['partitioning'], columns = 'values')
print(partition_analysis, '\n')
print(partition_analysis/partition_analysis.sum(), '\n')

price_range_analysis = pd.crosstab(house_offers_non_numeric['price_range'], columns = 'values')
print(price_range_analysis)
print(price_range_analysis/price_range_analysis.sum())

#group analysis for numeric type variables
print('Analysis of the "price" variable grouped by the variable "partitioning": ')
print(house_offers.groupby('partitioning').describe())
print('\n')

print('Analysis of the "price" variable grouped by the variable "price_range": ')
print(house_offers.groupby('price_range').describe(), '\n')

print('Analysis of the "price" variable grouped by the variables "partitioning", "price_range": ', '\n')
print(house_offers.groupby(['partitioning', 'price_range']).describe(), '\n')


# 3.2 graphical analysis of numeric and non-numeric variables


#create the histograms for the numeric variables
# for var in house_offers_numeric:
#     title = 'Histogram of ' + str(var)
#     plt.hist(house_offers_numeric[var], color = 'orange', density=False)
#     plt.title(title)
#     plt.xlabel(var)
#     plt.ylabel('Frequency')
#     plt.legend(var)
#     plt.show()
    
#create the boxplots for the numeric variables
# for var in house_offers_numeric:
#     title = 'Boxplot of ' + str(var)
#     plt.boxplot(house_offers_numeric[var])
#     plt.title(title)
#     plt.ylabel(var)
#     plt.show()

#create the histograms for the non-numeric variables
# for var in house_offers_non_numeric:
#     title = 'Histogram of ' + str(var)
#     plt.hist(house_offers_non_numeric[var], color = 'orange', density=False)
#     plt.title(title)
#     plt.xlabel(var)
#     plt.ylabel('Frequency')
#     plt.legend(var)
#     plt.show()
    
# 3.3 identifying outliers and treating them 
def outliers_iqr(variable):
    df = house_offers[variable]
    q1 = df.quantile(0.25)
    q3 = df.quantile(0.75)
    iqr = q3 - q1
    outlier = df[((df < (q1 - 1.5 * iqr)) | (df > (q3 + 1.5 * iqr)))]
    print('For variable ', variable, ' the number of outliers is: ' + str(len(outlier)))
    print('\n')
    print('The outliers are: ')
    print(outlier)
    print('\n')
    return outlier

outliers_iqr('price')
outliers_iqr('bathrooms_count')
outliers_iqr('rooms_count')
outliers_iqr('built_surface')
outliers_iqr('useful_surface')


    
    