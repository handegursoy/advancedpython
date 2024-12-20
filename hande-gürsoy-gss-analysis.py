import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

print("1. Data Loading and Initial Exploration\n")
gss = pd.read_csv('GSS7214.csv')

print(f"Dataset Shape: {gss.shape}")
print("\nFirst few rows of key variables:")
print(gss[['year', 'wrkstat', 'prestige', 'wrkslf', 'wrkgovt']].head())
print("\nMissing values in key variables:")
print(gss[['wrkstat', 'prestige', 'wrkslf', 'wrkgovt']].isnull().sum())

print("\n2. Descriptive Statistics\n")
work_vars = ['prestige', 'wrkstat', 'wrkslf', 'wrkgovt']
desc_stats = gss[work_vars].describe()
print(desc_stats)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.histplot(data=gss, x='prestige', kde=True)
plt.title('Distribution of Occupational Prestige')

plt.subplot(1, 2, 2)
sns.boxplot(data=gss, y='prestige')
plt.title('Box Plot of Prestige Scores')
plt.tight_layout()
plt.show()

print("\n3. Inferential Statistics\n")

self_employed = gss[gss['wrkslf'] == 1]['prestige'].dropna()
employees = gss[gss['wrkslf'] == 2]['prestige'].dropna()

t_stat, p_value = stats.ttest_ind(self_employed, employees)
print("T-test Results (Self-employed vs Employees):")
print(f"t-statistic: {t_stat:.4f}")
print(f"p-value: {p_value:.4f}")

#Confidence Intervals
ci_self = stats.t.interval(0.95, len(self_employed)-1, 
                          loc=np.mean(self_employed), 
                          scale=stats.sem(self_employed))
ci_emp = stats.t.interval(0.95, len(employees)-1, 
                         loc=np.mean(employees), 
                         scale=stats.sem(employees))

print("\n95% Confidence Intervals:")
print(f"Self-employed: {ci_self}")
print(f"Employees: {ci_emp}")

#Correlation Analysis
print("\n4. Correlation Analysis\n")

#Create correlation matrix
correlation_matrix = gss[work_vars].corr()
print("Correlation Matrix:")
print(correlation_matrix)

#Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix of Work Variables')
plt.show()

#Scatter plots
plt.figure(figsize=(12, 8))
sns.scatterplot(data=gss, x='wrkstat', y='prestige')
plt.title('Work Status vs Prestige')
plt.show()

#Regression Analysis
print("\n5. Regression Analysis\n")

#Prepare data for regression
X = gss[['wrkstat', 'wrkslf', 'wrkgovt']].dropna()
y = gss['prestige'].loc[X.index]

#Add constant for intercept
X = sm.add_constant(X)

#Fit OLS model
model = sm.OLS(y, X).fit()
print(model.summary())

#Model Diagnostics
print("\n6. Model Diagnostics\n")

#Get predictions and residuals
predictions = model.predict(X)
residuals = y - predictions

#Plot residuals vs fitted values
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(predictions, residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.title('Residuals vs Fitted Values')
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')

# Q-Q plot for normality
plt.subplot(1, 2, 2)
stats.probplot(residuals, dist="norm", plot=plt)
plt.title('Q-Q Plot of Residuals')
plt.tight_layout()
plt.show()

# Test for homoscedasticity
# Breusch-Pagan test
from statsmodels.stats.diagnostic import het_breuschpagan

_, p_value, _, _ = het_breuschpagan(residuals, X)
print("Breusch-Pagan test for heteroscedasticity:")
print(f"p-value: {p_value:.4f}")

vif_data = pd.DataFrame()
vif_data["Variable"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print("\nVariance Inflation Factors:")
print(vif_data)

# Analysis Summary
print("\nAnalysis Summary:")
print("""
The analysis of the GSS dataset revealed several key findings:

1. Data Exploration:
   - Examined work-related variables including prestige, work status, self-employment, and government work
   - Handled missing values and explored basic data structure

2. Descriptive Statistics:
   - Calculated comprehensive summary statistics
   - Visualized distributions through histograms and box plots

3. Inferential Statistics:
   - Conducted t-tests comparing prestige between self-employed and employees
   - Calculated confidence intervals for both groups

4. Correlation Analysis:
   - Created correlation matrix and heatmap
   - Visualized relationships through scatter plots

5. Regression Analysis:
   - Developed OLS model predicting prestige
   - Examined relationship between work variables

6. Model Diagnostics:
   - Checked assumptions through residual plots
   - Tested for heteroscedasticity and multicollinearity
   - Validated model reliability through various diagnostic tests
""")
