import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.stats.api as sms
import statsmodels.tsa.api as smt
from scipy.stats import chi2_contingency, levene, ttest_ind, pearsonr, skew, kurtosis
from scipy import stats
from statsmodels.formula.api import ols


# 2.1 Preliminary and transformation operations on variables

#importing the dataset
house_offers = pd.read_csv('house_offers.csv')

#show the dataset
#print(house_offers)

#verify the type of the variable house_offers
print(type(house_offers))


#create a factory type variable with at least 3 categories from a numeric variable
#in this case I chose the variable 'price' and I created the variable 'price_range' which contains 
#the following categories: 'small', 'medium', 'large'
#the categories were created according to the minimum value, the first quartile, the third quartile 
# and the maximum value of the variable 'price'
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
print('Analysis of the numeric variables: ')
print('Description: ',house_offers_numeric.describe())
print('\n')
#median analysis
print("Median",house_offers_numeric.median())
print('\n')
#skew analysis
#TODO: CAHNGE THE SKEW FUNCTION
print('Skewness: ',house_offers_numeric.skew(), skew(house_offers_numeric))
print('\n')
#kurtosis analysis
#TODO: CAHNGE THE KURTOSIS FUNCTION
print("Kurtosis: ",house_offers_numeric.kurtosis(), kurtosis(house_offers_numeric))
print('\n')

#create a new dataset with the non-numeric variables
house_offers_non_numeric = house_offers.select_dtypes(include=['category'])

#show the non-numeric dataset
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
print(house_offers.groupby('partitioning')['price'].describe(), '\n')

print('Analysis of the "price" variable grouped by the variable "price_range": ')
print(house_offers.groupby('price_range')['price'].describe(), '\n')

print('Analysis of the "price" variable grouped by the variables "partitioning", "price_range": ', '\n')
print(house_offers.groupby(['partitioning', 'price_range'])['price'].describe(), '\n')


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
    
# #create the boxplots for the numeric variables
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


# 4.Statistical analysis of categorical variables

# 4.1 data tabulation

#contingency table
contingency_table = pd.crosstab(house_offers['partitioning'], house_offers['price_range'])
print('Contingency table: ',contingency_table, '\n')

#marginal frequency table
for var in house_offers_non_numeric:
    print("Marginal frequency table for the variable", var, ":", house_offers.groupby(var).size() / len(house_offers), '\n')

#conditional frequency table
conditional_frequency_partitioning  = house_offers.groupby('partitioning')['price_range'].value_counts(normalize=True)
print('Conditional frequency table for the variable "partitioning": ', conditional_frequency_partitioning, '\n')

conditional_frequency_price_range  = house_offers.groupby('price_range')['partitioning'].value_counts(normalize=True)
print('Conditional frequency table for the variable "price_range": ', conditional_frequency_price_range, '\n')


# 4.2 association analysis
association_analysis = pd.crosstab(house_offers['partitioning'], house_offers['price_range'], margins=True)
chi_square = chi2_contingency(association_analysis)
print ('Association analysis - Chi-square test: ', '\n', chi_square, '\n')


# 4.3 concordance analysis
for var in house_offers_non_numeric:
    print("Concordance analysis for variable: " ,var ,'\n',stats.chisquare(house_offers_non_numeric[var].value_counts()), '\n')

    
# 5. Estimation and testing of means

# 5.1 Estimation of means by confidence interval
# it will be used the DescrStatsW function from the statsmodels.stats.api library

for var in house_offers_numeric:
    confidence_interval = sms.DescrStatsW(house_offers_numeric[var]).tconfint_mean()
    print("Estimation of means by confidence interval for the variable", var, ":", '\n', confidence_interval, '\n')


# 5.2 Testing population means: testing a mean with a fixed value
# it will be used the ttest_1samp function from the scipy.stats library

ttest = stats.ttest_1samp(house_offers_numeric['price'], 130000)
print("Testing population means: testing a mean with a fixed value for the variable 'price':", '\n', ttest, '\n')

# 5.3 testing the difference between two means (independent samples or paired samples)
decomandat = house_offers[house_offers['partitioning'] == 'decomandat']
semidecomandat = house_offers[house_offers['partitioning'] == 'semidecomandat']
t_statistic, p_value = levene(decomandat['price'], semidecomandat['price'])
print('Testing the difference between two means (independent samples or paired samples) for the variable "price":', '\n')
print('t_statistic: ', t_statistic, '\n')
print('p_value: ', p_value, '\n')

#after the result that we got it is hard to say if p-value is greater than 0.05 or not
#so we will use the ttest_ind function from the scipy.stats library
#let's check how is p-value

if p_value > 0.05:
  test = ttest_ind(decomandat['price'], semidecomandat['price'])
  print('p-value is greater than 0.05', test, '\n')
else:
    test = ttest_ind(decomandat['price'], semidecomandat['price'], equal_var=False)
    print ('p-value is less than 0.05', test, '\n')

# 5.4 testing the difference between three means or more means
# for this we will use the ANOVA test
anova_test = ols('price ~ partitioning', data=house_offers).fit()
anova_table = sm.stats.anova_lm(anova_test, type=2)
print('Testing the difference between three means or more means for the variable "price": \n', anova_table)


# 6. Regression and correlation analysis

# 6.1 Correlation analysis
for var in house_offers_numeric:
    correlation_analysis = house_offers_numeric['price'].corr(house_offers_numeric[var])
    print("Correlation analysis for the variable", var, ":", '\n', pearsonr(house_offers_numeric[var], house_offers_numeric['price']), '\n')
    print("Correlation coefficient for ", var, "is: ", '\n', correlation_analysis, '\n')
    
    
# 6.2 regression analysis

# simple linear regression

# we need an independent variable and a dependent variable
# the independent variable will be 'built_surface' and the dependent variable will be 'price'
var_simple_reg = sm.add_constant(house_offers['built_surface'])
simple_linear_regression = sm.OLS(house_offers['price'], var_simple_reg).fit()
print("Summary for simple liniar regression: ",simple_linear_regression.summary(), '\n')
print('Parameters: ', simple_linear_regression.params, '\n')
print('Predictions: ', simple_linear_regression.predict(), '\n')
print('Standard errors: ', simple_linear_regression.bse, '\n')
print('Residuals: ', simple_linear_regression.resid, '\n')

# multiple linear regression 

# we need a dependent variable and at least 2 independent variables
# the dependent variable will be 'price' and the independent variables will be 'built_surface' and 'rooms_count'
var_multiple_reg = sm.add_constant(house_offers[['built_surface', 'rooms_count']])
multiple_linear_regression = sm.OLS(house_offers['price'], var_multiple_reg).fit()
print("Summary for multiple liniar regression: ",multiple_linear_regression.summary(), '\n')

# non-linear regression
# a polinomial regression will be used
# we need a dependent variable and an independent variable
# the dependent variable will be 'price' and the independent variable will be 'built_surface'

polinomial_regression = pd.DataFrame({'IndependentVar': house_offers['built_surface'], 
                                   'IndependetVar^2': house_offers['built_surface']**2})
polinomial_regression = sm.add_constant(polinomial_regression)

non_liniar_regression = sm.OLS(house_offers['price'], polinomial_regression).fit()
print("Summary for non-liniar regression: ",non_liniar_regression.summary(), '\n')


# 6.3 hypothesis testing
# for this step it will be easier to create a function that will be used for all the hypothesis testing

def hypothesis_testing( regression_type, var):
    
    #test if mean of error is 0
    test_errors = regression_type.resid
    test_errors_mean = stats.ttest_1samp(test_errors, 0)
    print('Test if mean of error is 0: ', test_errors_mean, '\n')
    
    #homoscedasticity test
    homoscedasticity_test = sm.OLS(abs(test_errors), var).fit()
    print('Homoscedasticity test: ', homoscedasticity_test.summary(), '\n')
    
    # #BP test
    # bp_test = sms.het_breuschpagan(regression_type.resid, regression_type.model.exog)
    # print('BP test: ', bp_test, '\n')
    
    # #GQ test
    # gq_test = sms.het_goldfeldquandt(regression_type.resid, regression_type.model.exog)
    # print('GQ test: ', gq_test, '\n')
    
    #error normality test
    error_normality_test = stats.normaltest(test_errors)
    print('Error normality test: ', error_normality_test, '\n')
    
    #autoconelation error test
    smt.graphics.plot_acf(test_errors, lags=40, alpha=0.05).savefig('autoconelation_error_test.png')
    
    
# hypothesis_testing(simple_linear_regression, var_simple_reg)
# hypothesis_testing(multiple_linear_regression, var_multiple_reg)
# hypothesis_testing(non_liniar_regression, polinomial_regression)


# 6.4 comparing at least 2 regression models and choosing the best-fitting model
comparing_regression_models = sm.stats.anova_lm(simple_linear_regression, multiple_linear_regression)
print('Comparing at least 2 regression models and choosing the best-fitting model: ', '\n', comparing_regression_models, '\n')