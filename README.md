# feature-engineering-ames-housing-dataset

My notebook focuses on the core concepts of Feature Engineering and Data Preprocessing using Python, Pandas, Matplotlib, and Seaborn. I systematically approach the data cleaning process in distinct phases:

1. Statistical Outlier Detection
I begin by creating a synthetic dataset of ages to demonstrate mathematical outlier removal. I calculate the 25th and 75th percentiles to find the Interquartile Range (IQR). Using the standard formula, I establish boundaries and successfully filter out the statistical outliers.

2. Visual Outlier Removal on Real Data
I load the Ames Housing dataset and evaluate the correlation of all numeric features against the target variable, SalePrice. By plotting scatter plots for highly correlated features like Overall Qual and Gr Liv Area, I visually identify extreme outliers, specifically houses with massive living areas but unusually low sale prices. I then extract the exact indices of these outliers and drop them from the dataframe to prevent them from skewing my future machine learning models.

3. Missing Data Evaluation and Imputation
I write a custom function to calculate and visualize the exact percentage of missing data for every column. I handle these missing values using a highly tailored approach: I drop columns that are missing almost all of their data, such as Pool QC, Misc Feature, Alley, and Fence.
For features where a missing value logically means the absence of that feature, like a house having no basement, no garage, or no fireplace, I replace numeric missing values with 0 and categorical missing values with the string 'None'. I drop a handful of specific rows where data is missing completely at random and cannot be safely inferred, such as Electrical and Garage Area. For Lot Frontage, I use an advanced imputation technique by grouping the data by Neighborhood and filling the missing values with the mean Lot Frontage of that specific neighborhood.

4. Categorical Encoding
I recognize that some numeric columns, like MS SubClass, are actually categorical codes. I convert these to strings, isolate all object-type columns, and use Pandas get_dummies to perform one-hot encoding while dropping the first column to avoid multicollinearity. Finally, I concatenate the numeric dataframe with the newly encoded categorical dataframe to create a machine-learning-ready dataset.

Ames Housing Dataset: Advanced Feature Engineering
This repository contains my comprehensive data preprocessing and feature engineering pipeline for the Ames Housing dataset. My project demonstrates how to clean, process, and prepare raw housing data for predictive machine learning models by properly handling outliers, missing data, and categorical variables. Raw data is rarely ready for immediate machine learning application. My project walks through the critical steps of feature engineering required to maximize model accuracy. By exploring the relationships between house features and their final sale price, I refined the dataset into a clean and strictly numerical format.\

Technologies Used:
1. Python 3
2. Pandas for data manipulation and cleaning
3. NumPy for statistical calculations
4. Matplotlib for data visualization
5. Seaborn for statistical plotting

Methodology and Pipeline:
1. Statistical and Visual Outlier Removal
My project begins by defining outliers using the Interquartile Range (IQR) method. Moving to the Ames Housing data, I generate scatter plots for features highly correlated with the target variable, SalePrice. I isolate and remove extreme outliers, specifically properties with massive above-ground living areas but unusually low sale prices, to prevent model distortion.

2. Missing Data Imputation
I use a custom function to evaluate the percentage of missing values across all features, which I visualize via bar charts. I handle missing data through strict logical rules rather than blind imputation. I drop columns with over 80 percent missing data entirely. For basement, garage, and masonry features, I replace missing values with 0 for numeric types or 'None' for categorical types to represent the physical absence of the feature.

3. Advanced Grouped Imputation
For the Lot Frontage feature, I impute missing values by grouping the dataset by Neighborhood. I replace the missing values with the mean Lot Frontage of the specific neighborhood the house belongs to, providing a highly accurate estimation based on local zoning trends.

4. Categorical Encoding and Dummy Variables
I convert numerical columns that actually represent categories, such as MS SubClass, to string types. I then transform all categorical text data into a machine-readable format using one-hot encoding. I utilize the drop_first parameter to drop one of the dummy columns per category, effectively preventing the dummy variable trap and avoiding multicollinearity.

5. Final Dataset Assembly
I concatenate the thoroughly cleaned numeric data with the newly generated one-hot encoded variables. My final output is a completely numeric, clean dataframe that is fully optimized for training regression models.
