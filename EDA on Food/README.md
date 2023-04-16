# README

## Table of Contents
1. [A. An overview of the dataset](#overview)
2. [B. Data Cleaning](#data-cleaning)
3. [C. Preliminary Exploratory Data Analysis (EDA)](#preliminary-eda)
4. [D. EDA: Text data](#text-data)
5. [E. EDA: Time-series data](#time-series-data)
6. [F. EDA: Correlation analysis](#correlation-analysis)
7. [G. Advanced EDA](#advanced-eda)

<a name="overview"></a>
## A. An overview of the dataset
- Import the data as a pandas DataFrame into your notebook.
- Check the number of rows and columns. You should have 385,384 samples and 99 columns.
- Display a few entries from the DataFrame.
- Check the data type for each column. Create separate lists to hold the names of columns of the same data type.
- Check the data by the type of information they hold. Create 3 lists that hold _per_hundred, _per_portion and _unit columns. Put the remaining column names in a 4th list named other_cols.

<a name="data-cleaning"></a>
## B. Data Cleaning
1. Identify and remove duplicated products.
2. Analyze missing values and address them:
   a. Create a table that shows both the number and the percentage of missing values for all columns sorted from largest to smallest.
   b. Use missingno to help you visualize where the missing are in the whole data frame and when missing values overlap between columns or not.
   c. Create 4 lists that hold per_hundred, per_portion, _unit, and other columns. Create 4 line plots or bar charts that show the percentages of missing values in each list.
   d. Address missing values using different strategies.

<a name="preliminary-eda"></a>
## C. Preliminary Exploratory Data Analysis (EDA)
- Explore categorical variables using appropriate visualizations.
- Provide descriptive statistics and informative plots of the numerical variables.
- Check for errors and unrealistic values and address these problems.

<a name="text-data"></a>
## D. EDA: Text data
- Preprocess the text data in the ingredients_en column.
- Answer the following questions:
  1. Find the product with the longest ingredients list.
  2. Find the products with the shortest ingredients list.
  3. Which are the most frequent ingredients in products?

<a name="time-series-data"></a>
## E. EDA: Time-series data
- Analyze the total number of products added to the database using the created_at column.
- Investigate the total number of items created each month at each hour.
- Investigate the evolution of the total number of items over time.

<a name="correlation-analysis"></a>
## F. EDA: Correlation analysis
- Quantify the linear relationships between the energy_per_hundred and the per_hundred columns.
- Reveal the true nature of the relationship, linear or non-linear, between variables using visualizations.
- Test the independence of two categorical variables statistically.

<a name="advanced-eda"></a>
## G. Advanced EDA
- Do an in-depth analysis of the organic vs. non-organic products distribution in the Open Food database.
  1. What is the total number of samples by country?
  2. Count the number of organic and non-organic products in each country.
  3. Compare the distributions of the macronutrients between organic and non-organic products in each country.
