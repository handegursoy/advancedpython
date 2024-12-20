import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

def load_and_clean_data(filepath):
    """Load and perform initial data cleaning"""
    gss = pd.read_csv(filepath)
    
    # Convert work-related variables to categorical
    categorical_vars = ['wrkstat', 'wrkslf', 'wrkgovt']
    for var in categorical_vars:
        gss[var] = pd.Categorical(gss[var])
    
    return gss

def plot_prestige_distribution(gss):
    """Create enhanced prestige distribution plots"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Enhanced histogram
    sns.histplot(data=gss, x='prestige', kde=True, bins=30, ax=ax1)
    ax1.set_title('Distribution of Occupational Prestige')
    ax1.set_xlabel('Prestige Score')
    ax1.set_ylabel('Count')
    
    # Enhanced violin plot (instead of basic boxplot)
    sns.violinplot(data=gss, y='prestige', ax=ax2)
    ax2.set_title('Violin Plot of Prestige Scores')
    ax2.set_ylabel('Prestige Score')
    
    plt.tight_layout()
    return fig

def analyze_employment_groups(gss):
    """Perform detailed analysis of employment groups"""
    # Create employment categories
    employment_categories = {
        'self_employed': (gss['wrkslf'] == 1),
        'government': (gss['wrkgovt'] == 1),
        'private_sector': ((gss['wrkslf'] == 2) & (gss['wrkgovt'] == 2))
    }
    
    results = {}
    for category, mask in employment_categories.items():
        group_data = gss.loc[mask, 'prestige'].dropna()
        results[category] = {
            'mean': group_data.mean(),
            'median': group_data.median(),
            'std': group_data.std(),
            'ci': stats.t.interval(0.95, len(group_data)-1,
                                 loc=np.mean(group_data),
                                 scale=stats.sem(group_data))
        }
    
    return pd.DataFrame(results).round(2)

def create_correlation_heatmap(gss, work_vars):
    """Create an enhanced correlation heatmap"""
    correlation_matrix = gss[work_vars].corr()
    
    plt.figure(figsize=(10, 8))
    mask = np.triu(np.ones_like(correlation_matrix), k=1)
    sns.heatmap(correlation_matrix,
                mask=mask,
                annot=True,
                cmap='RdBu',
                center=0,
                square=True,
                fmt='.2f',
                linewidths=0.5)
    plt.title('Correlation Matrix of Work Variables')
    return plt.gcf()

def run_regression_analysis(gss):
    """Perform enhanced regression analysis with robust standard errors"""
    X = gss[['wrkstat', 'wrkslf', 'wrkgovt']].dropna()
    y = gss['prestige'].loc[X.index]
    
    X = sm.add_constant(X)
    
    # Fit model with robust standard errors
    model = sm.OLS(y, X).fit(cov_type='HC3')
    
    # Calculate VIF
    vif_data = pd.DataFrame({
        "Variable": X.columns,
        "VIF": [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    })
    
    return model, vif_data

def plot_regression_diagnostics(model, X, y):
    """Create comprehensive regression diagnostic plots"""
    predictions = model.predict(X)
    residuals = y - predictions
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Residuals vs Fitted
    sns.scatterplot(x=predictions, y=residuals, ax=ax1)
    ax1.axhline(y=0, color='r', linestyle='--')
    ax1.set_title('Residuals vs Fitted Values')
    ax1.set_xlabel('Fitted Values')
    ax1.set_ylabel('Residuals')
    
    # Q-Q plot
    stats.probplot(residuals, dist="norm", plot=ax2)
    ax2.set_title('Q-Q Plot of Residuals')
    
    # Scale-Location plot
    sns.scatterplot(x=predictions, y=np.sqrt(np.abs(residuals)), ax=ax3)
    ax3.set_title('Scale-Location Plot')
    ax3.set_xlabel('Fitted Values')
    ax3.set_ylabel('√|Standardized Residuals|')
    
    # Residuals histogram
    sns.histplot(residuals, kde=True, ax=ax4)
    ax4.set_title('Distribution of Residuals')
    ax4.set_xlabel('Residuals')
    
    plt.tight_layout()
    return fig

def main():
    # Load and prepare data
    gss = load_and_clean_data('GSS7214.csv')
    work_vars = ['prestige', 'wrkstat', 'wrkslf', 'wrkgovt']
    
    # Generate all plots and analyses
    prestige_dist = plot_prestige_distribution(gss)
    employment_stats = analyze_employment_groups(gss)
    correlation_plot = create_correlation_heatmap(gss, work_vars)
    model, vif_data = run_regression_analysis(gss)
    diagnostic_plots = plot_regression_diagnostics(model, 
                                                 sm.add_constant(gss[['wrkstat', 'wrkslf', 'wrkgovt']].dropna()),
                                                 gss['prestige'])
    
    # Print results
    print("\nEmployment Group Statistics:")
    print(employment_stats)
    print("\nRegression Summary:")
    print(model.summary())
    print("\nVariance Inflation Factors:")
    print(vif_data)
    
    plt.show()

if __name__ == "__main__":
    main()
