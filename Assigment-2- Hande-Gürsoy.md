# Advanced Gapminder Dataset Analysis

## Setup and Data Preparation
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load and prepare data
df = pd.read_csv('gapminder_data_graphs.csv')

# Enhanced missing value handling
def advanced_missing_value_report(dataframe):
    missing_data = dataframe.isnull().sum()
    missing_percentages = 100 * missing_data / len(dataframe)
    missing_table = pd.concat([missing_data, missing_percentages], axis=1, keys=['Missing Count', 'Missing Percentage'])
    return missing_table[missing_table['Missing Count'] > 0]

# Rename columns
df.rename(columns={
    'pop': 'Population', 
    'gdpPercap': 'GDP_per_Capita',
    'lifeExp': 'Life_Expectancy'
}, inplace=True)

# Advanced imputation strategy
df['GDP_per_Capita'].fillna(df.groupby('continent')['GDP_per_Capita'].transform('median'), inplace=True)
df['Life_Expectancy'].fillna(df.groupby('continent')['Life_Expectancy'].transform('median'), inplace=True)
```

## Advanced Data Classification
```python
def comprehensive_population_categorization(row):
    continent_thresholds = {
        'Africa': (5_000_000, 20_000_000),
        'Americas': (10_000_000, 50_000_000),
        'Asia': (20_000_000, 100_000_000),
        'Europe': (5_000_000, 20_000_000),
        'Oceania': (1_000_000, 5_000_000)
    }
    
    thresholds = continent_thresholds.get(row['continent'], (10_000_000, 50_000_000))
    
    if row['Population'] > thresholds[1]:
        return 'Very Large'
    elif row['Population'] > thresholds[0]:
        return 'Large'
    else:
        return 'Small'

df['Population_Category'] = df.apply(comprehensive_population_categorization, axis=1)
```

## Statistical Analysis
```python
# Comprehensive continent-level analysis
continent_summary = df.groupby('continent').agg({
    'GDP_per_Capita': ['mean', 'median', 'std'],
    'Life_Expectancy': ['mean', 'min', 'max'],
    'Population': ['sum', 'mean']
})
print("\nContinent-Level Summary Statistics:")
print(continent_summary)

# Correlation matrix
correlation_matrix = df[['GDP_per_Capita', 'Life_Expectancy', 'Population']].corr()
```

## Advanced Visualization
```python
# Multi-panel visualization
plt.figure(figsize=(15, 10))

# Life Expectancy by Continent
plt.subplot(2, 2, 1)
sns.boxplot(x='continent', y='Life_Expectancy', data=df)
plt.title('Life Expectancy Distribution by Continent')

# GDP per Capita Scatter
plt.subplot(2, 2, 2)
sns.scatterplot(x='GDP_per_Capita', y='Life_Expectancy', 
                hue='continent', data=df, alpha=0.6)
plt.title('GDP per Capita vs Life Expectancy')

# Population Category Distribution
plt.subplot(2, 2, 3)
df['Population_Category'].value_counts().plot(kind='bar')
plt.title('Population Category Distribution')

# Correlation Heatmap
plt.subplot(2, 2, 4)
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Heatmap')

plt.tight_layout()
plt.show()
```

## Regression Analysis
```python
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Prepare data for regression
X = df[['GDP_per_Capita']]
y = df['Life_Expectancy']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Fit linear regression
model = LinearRegression()
model.fit(X_scaled, y)

print("\nRegression Analysis:")
print(f"Coefficient: {model.coef_[0]:.4f}")
print(f"Intercept: {model.intercept_:.4f}")
print(f"R-squared: {model.score(X_scaled, y):.4f}")
```
