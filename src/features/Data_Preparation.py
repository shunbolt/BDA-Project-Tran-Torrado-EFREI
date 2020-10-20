import pandas as pd
import numpy as np

df_train = pd.read_csv("../data/raw/application_train.csv")
df_test = pd.read_csv("../data/raw/application_test.csv")

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
        
        # Print some summary information
        #print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
        #   "There are " + str(mis_val_table_ren_columns.shape[0]) +
        #     " columns that have missing values.")
        
        # Return the dataframe with missing information
        return mis_val_table_ren_columns
    
    
# Missing values statistics
missing_values = missing_values_table(df_train)
missing_values = missing_values[missing_values["% of Total Values"] > 65]
to_drop = missing_values
to_drop = to_drop.transpose()
df_train = df_train.drop(columns = to_drop)



# Missing values statistics
missing_values = missing_values_table(df_test)
missing_values = missing_values[missing_values["% of Total Values"] > 65]
to_drop = missing_values
to_drop = to_drop.transpose()
df_test = df_test.drop(columns = to_drop)

df_train_encoded = pd.get_dummies(df_train)
df_test_encoded = pd.get_dummies(df_test)

#print("Training set shape :", df_train_encoded.shape)
#print("Testing set shape :", df_test_encoded.shape)

train_labels = df_train_encoded['TARGET']

df_train_encoded, df_test_encoded = df_train_encoded.align(df_test_encoded, join = 'inner', axis = 1)

# Add the target back in
df_train_encoded['TARGET'] = train_labels

#print('Training Features shape: ', df_train_encoded.shape)
#print('Testing Features shape: ', df_test_encoded.shape)


df_train = df_train_encoded
df_test = df_test_encoded


# Create an anomalous flag column
df_train['DAYS_EMPLOYED_ANOM'] = df_train["DAYS_EMPLOYED"] == 365243
df_test['DAYS_EMPLOYED_ANOM'] = df_test["DAYS_EMPLOYED"] == 365243

# Replace the anomalous values with nan
df_train['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace = True)
df_test['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace = True)

subset_df_train = df_train[["SK_ID_CURR", "DAYS_BIRTH", "EXT_SOURCE_3", "EXT_SOURCE_2", "EXT_SOURCE_1", "TARGET"]]
subset_df_test = df_test[["SK_ID_CURR", "DAYS_BIRTH", "EXT_SOURCE_3", "EXT_SOURCE_2", "EXT_SOURCE_1"]]

print(subset_df_train)
print(subset_df_test)

subset_df_train.to_csv(r'../data/processed/processed_application_train.csv', index=False)
subset_df_test.to_csv(r'../data/processed/processed_application_test.csv',  index=False)

