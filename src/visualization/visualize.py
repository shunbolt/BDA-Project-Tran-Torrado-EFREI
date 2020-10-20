import pandas as pd
from pandas.plotting import table 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df_train = pd.read_csv("../../data/raw/application_train.csv")

# Target distribution
df_train['TARGET'].astype(int).plot.hist()
plt.savefig('visuals/target_distribution.png')
plt.clf()

print("Generated target distribution")

# Table of missing values
def missing_values_table(df):
        # Total missing values
        mis_val = df.isnull().sum()
        
        # Percentage of missing values
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        
        # Make a table with the results
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        
        # Rename the columns
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        
        # Sort the table by percentage of missing descending
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
        
        # Return the dataframe with missing information
        return mis_val_table_ren_columns

missing_values = missing_values_table(df_train).head(20)
ax = plt.subplot(111, frame_on=False) 
ax.xaxis.set_visible(False)  
ax.yaxis.set_visible(False)  

table(ax, missing_values, loc = 'center')  
plt.savefig('visuals/missing_values.png',bbox_inches='tight')
plt.clf()
print("Generated missing values table")

# Check anomalies in DAYS_BIRTH and DAYS_EMPLOYED

df_train['DAYS_BIRTH'].plot.hist(title = 'Days of birth');
plt.savefig('visuals/days_birth_raw.png')
plt.clf()

(df_train['DAYS_BIRTH'] / -365 ).plot.hist(title = 'Days of birth');
plt.savefig('visuals/days_birth_fixed.png')
plt.clf()

print("Generated DAYS_BIRTH histograms")

df_train['DAYS_EMPLOYED'].plot.hist(title = 'Days Employment Histogram');
plt.savefig('visuals/days_employed_raw.png')
plt.clf()

df_train['DAYS_EMPLOYED_ANOM'] = df_train["DAYS_EMPLOYED"] == 365243
df_train['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace = True)

df_train['DAYS_EMPLOYED'].plot.hist(title = 'Days Employment Histogram');
plt.xlabel('Days Employment');
plt.savefig('visuals/days_employed_fixed.png')
plt.clf()

print("Generated DAYS_EMPLOYED histograms")

# Correlation lists
correlations = df_train.corr()['TARGET'].sort_values()

corr_positive = correlations.head(15)
corr_negative = correlations.tail(15)

ax = plt.subplot(111, frame_on=False) 
ax.xaxis.set_visible(False)  
ax.yaxis.set_visible(False) 

table(ax, corr_positive, loc = 'center')
plt.savefig('visuals/correlations_positive.png',bbox_inches='tight')
plt.clf()

ax = plt.subplot(111, frame_on=False) 
ax.xaxis.set_visible(False)  
ax.yaxis.set_visible(False) 

table(ax, corr_negative,loc = 'center')
plt.savefig('visuals/correlations_negative.png',bbox_inches='tight')
plt.clf()

print("Generated correlations")

# Age distribution 

plt.hist(df_train['DAYS_BIRTH'] / 365, edgecolor = 'k', bins = 25)
plt.title('Age of Client'); plt.xlabel('Age (years)'); plt.ylabel('Count');
plt.savefig('visuals/age_distribution.png')
plt.clf()

plt.style.use('fivethirtyeight')
plt.figure(figsize = (10, 8))

sns.kdeplot(df_train.loc[df_train['TARGET'] == 0, 'DAYS_BIRTH'] / 365, label = 'target == 0', legend = True)

sns.kdeplot(df_train.loc[df_train['TARGET'] == 1, 'DAYS_BIRTH'] / 365, label = 'target == 1', legend = True)

plt.xlabel('Age (years)'); plt.ylabel('Density'); plt.title('Distribution of Ages'); plt.legend()
plt.savefig('visuals/age_target_plot.png')
plt.clf()

# Plot age bins data

age_data = df_train[['TARGET', 'DAYS_BIRTH']]
age_data['YEARS_BIRTH'] = age_data['DAYS_BIRTH'] / 365

age_data['YEARS_BINNED'] = pd.cut(age_data['YEARS_BIRTH'], bins = np.linspace(20, 70, num = 11))

age_groups = age_data.groupby('YEARS_BINNED').mean()

plt.figure(figsize = (8, 8))

plt.bar(age_groups.index.astype(str), 100 * age_groups['TARGET'])

plt.xticks(rotation = 75); plt.xlabel('Age Group (years)'); plt.ylabel('Failure to Repay (%)')
plt.title('Failure to Repay by Age Group');
plt.savefig('visuals/age_groups_repayment_failure.png')
plt.clf()

print("Generated age related visualizations")

# Exterior sources

ext_data = df_train[['TARGET', 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH']]
ext_data_corrs = ext_data.corr()

plt.figure(figsize = (8, 6))

# Heatmap of correlations
sns.heatmap(ext_data_corrs, cmap = plt.cm.RdYlBu_r, vmin = -0.25, annot = True, vmax = 0.6)
plt.title('Correlation Heatmap');
plt.savefig('visuals/correlation_heatmap.png')
plt.clf()

# Plot targets to ext_source

plt.figure(figsize = (10, 12))

# iterate through the sources
for i, source in enumerate(['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']):
    
    # create a new subplot for each source
    plt.subplot(3, 1, i + 1)
    # plot repaid loans
    sns.kdeplot(df_train.loc[df_train['TARGET'] == 0, source], label = 'target == 0')
    # plot loans that were not repaid
    sns.kdeplot(df_train.loc[df_train['TARGET'] == 1, source], label = 'target == 1')
    
    # Label the plots
    plt.title('Distribution of %s by Target Value' % source)
    plt.xlabel('%s' % source); plt.ylabel('Density');
    plt.legend()
    plt.savefig('visuals/distribution_%s.png' % source)
    plt.clf()
    
plt.tight_layout(h_pad = 2.5)

print("Generated source revenues related visualization")
print("Visualization generation complete")

