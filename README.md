# Analysis-and-Prediction-of-Customer-Loss-in-Financial-Industry
Analysis and prediction of customer loss
## Data-Analysis-of-Customer-Loss-in-Financial-Industry
### Data-Cleaning
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

project=pd.read_csv('Modelling.csv')
print(project.shape,'\n')
print(project.info(),'\n')
print(project.describe(),'\n')

(project.astype(np.object)=='?').any()
#No missing values labeled as ?

column=list(project.columns)
for col in column:
    project.loc[project[col]=='?',col]=np.nan
project.dropna(axis=0, inplace=True)

print(project['CustomerId'].unique().shape)
#All values are unique. No repeated values.

def frequency_table(project, column):
    for col in column:
        print('For column '+col+'\n')
        print(project[col].value_counts())

column2=['Geography', 'Gender']
frequency_table(project, column2)

### Data-Visualization
#### Distribution-of-features-and-label
def plot_bars(project, column):
    for col in column:
        fig=plt.figure(figsize=(6,6))
        sns.set_style('whitegrid')
        sns.countplot(project[col])
        plt.title('Number of clients by '+col)
        plt.xlabel(col)
        plt.ylabel('Number of clients')
        plt.show()
        
column3=['Geography', 'Gender','NumOfProducts', 'HasCrCard', 'IsActiveMember', 'Exited']
plot_bars(project, column3)

def plot_histogram(project, column):
    for col in column:
        fig = plt.figure(figsize=(6,6))
        sns.set_style('whitegrid')
        sns.distplot(project[col], rug=False, hist=True)
        plt.title('Number of clients by '+col)
        plt.xlabel(col)
        plt.ylabel('Number of clients')
        plt.show()

column4=['CreditScore', 'Age', 'Tenure', 'Balance', 'EstimatedSalary']
plot_histogram(project, column4)

#### Numerical-variables-and-label
def plot_box(project, column, col_x='Exited'):
    for col in column:
        sns.set_style('whitegrid')
        sns.boxplot(col_x, col, data=project)
        plt.xlabel(col_x)
        plt.ylabel(col)
        plt.show()
        
num_column=['CreditScore', 'Age', 'Tenure', 'Balance', 'EstimatedSalary']
plot_box(project, num_column)

def plot_violin(project, column, col_x='Exited'):
    for col in column:
        sns.set_style('whitegrid')
        sns.violinplot(col_x, col, data=project)
        plt.xlabel(col_x)
        plt.ylabel(col)
        plt.show()
        
num_column=['CreditScore', 'Age', 'Tenure', 'Balance', 'EstimatedSalary']
plot_violin(project, num_column)

#Check the correlation between numerical variables
num_column=['CreditScore', 'Age', 'Tenure', 'Balance', 'EstimatedSalary']
sns.pairplot(project[num_column], palette='Set3', diag_kind='kde', size=2).map_upper(sns.kdeplot, cmap="Blues_d")

#### Categorical-variables-and-label
sns.countplot('Geography', hue='Exited',data=project)

sns.countplot('Gender', hue='Exited',data=project)

sns.countplot('NumOfProducts', hue='Exited',data=project)

sns.countplot('HasCrCard', hue='Exited',data=project)

sns.countplot('IsActiveMember', hue='Exited',data=project)
