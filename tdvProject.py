import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.stats.api as sms
import statsmodels.tsa.api as smt
from scipy.stats import pearsonr


#Preliminary and transformation operations on variables

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

#export the new dataset to a csv file
house_offers.to_csv('house_offers_new.csv', index=False)
