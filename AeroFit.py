##### Libraries #####
import numpy as np
import pandas as pd
import pingouin as pg
import seaborn as sns
from matplotlib import pyplot as plt
import warnings

##### Exploratory Data Analysis #####
df = pd.read_csv('datasets/datasets/aerofit_treadmill_data.csv')

pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 10)

# Data Types of Columns
def df_datatypes(df):
    df_desc = pd.DataFrame(df.dtypes.value_counts().reset_index())
    df_desc.columns = ['Data Type', 'Count']
    return df_desc.sort_values('Count', ascending=False)

df_datatypes(df)

# DataFrame info
def get_df_info(df):
    df_info = pd.DataFrame()
    df_info['Column'] = df.columns.tolist()
    df_info['Data Type'] = df.dtypes.tolist()
    df_info['Non-Null Count'] = df.notnull().sum().tolist()
    df_info['# Null'] = df.isna().sum().tolist()
    df_info['Total Count'] = df.shape[0]
    df_info['% Null'] = np.round(df_info['# Null'] / df_info['Total Count'],2) * 100
    df_info['% Null'] = [str(n) + '%' for n in df_info['% Null']]
    return df_info

get_df_info(df)

# Numerical Info
numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
get_df_info(df[numeric_columns])

# Categorical Info
categorical_columns = df.select_dtypes(exclude=np.number).columns.tolist()
get_df_info(df[categorical_columns])

# Categorical Description
def df_describe_categorical(df):
    categorical_df = pd.DataFrame()
    categorical_columns = df.select_dtypes(exclude=np.number).columns.tolist()
    categorical_df['Column'] = df[categorical_columns].columns.tolist()
    categorical_df['Data Type'] = df[categorical_columns].dtypes.tolist()
    categorical_df['Count'] = [df[col].count() for col in categorical_columns]
    categorical_df['Unique Values'] = [df[col].nunique() for col in categorical_columns]
    categorical_df['Top Category'] = [df[col].mode().tolist() for col in categorical_columns]
    categorical_df['Top Category Frequency'] = [df.loc[(df[col].isin(df[col].mode())), col].count() for col in categorical_columns]
    categorical_df['% Top Category'] = np.round(categorical_df['Top Category Frequency'] / categorical_df['Count'],2) * 100
    categorical_df['% Top Category'] = [str(n) + '%' for n in categorical_df['% Top Category'].round(2)]
    return categorical_df

df_describe_categorical(df)

# Numeric Description
def get_describe_numeric(df, precision=2):
    numeric_df = pd.DataFrame()
    numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
    numeric_df['Column'] = df[numeric_columns].columns.tolist()
    numeric_df['Data Type'] = df[numeric_columns].dtypes.tolist()
    numeric_df['Count'] = [df[col].count() for col in numeric_columns]
    numeric_df['Mean'] = [df[col].mean().round(precision) for col in numeric_columns]
    numeric_df['Std Dev'] = [df[col].std().round(precision) for col in numeric_columns]
    numeric_df['Min'] = [df[col].min().round(precision) for col in numeric_columns]
    numeric_df['25%'] = [np.nanpercentile(df[col], 25).round(precision) for col in numeric_columns]
    numeric_df['50%'] = [np.nanpercentile(df[col], 50).round(precision) for col in numeric_columns]
    numeric_df['75%'] = [np.nanpercentile(df[col], 75).round(precision) for col in numeric_columns]
    numeric_df['Max'] = [df[col].max().round(precision) for col in numeric_columns]
    return numeric_df

get_describe_numeric(df)

# numerical - correlogram
def graph_correlogram(df):
    sns.set_theme(style="white") 
    # Compute the correlation matrix (use round to change decimals)
    corr = df.corr(numeric_only=True).round(2)
    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))
    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))
    # Draw the heatmap with the mask and correct aspect ratio
    graph = sns.heatmap(
        corr, mask=mask, cmap='RdYlGn', vmax=.3, center=0, square=True,
        annot=True, linewidths=.5,
        cbar_kws={"shrink": .5}, annot_kws={"fontsize":15})
    return graph

graph_correlogram(df)

# correlation matrix with p-values
def df_correlation_matrix(df):
    numerical = df.select_dtypes(include=[np.number]).columns.tolist()
    print(f'Pearson Correlation Matrix with P-Values')
    print(f'[Coef in Btm Tri / p-Values in Up Tri]')
    print(f'*** for <0.001, ** for <0.01, * for <0.05')
    print(f'-----------------------------------------')
    return df[numerical].rcorr(method='pearson').round(3)

df_correlation_matrix(df)


# graph of all numeric data types
def graph_numeric_histograms(df):
    custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    sns.set_theme(style="ticks", rc=custom_params)
    numeric_columns = df.select_dtypes(include=np.number)
    for col in numeric_columns:
        sns.histplot(df[col], kde=True, color='black')
        plt.title(f'Histogram of {col.title()}')
        plt.show()
    
graph_numeric_histograms(df)

# anova df
def get_anova_df(target, df):
    # filter warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)
    numerical_columns = df.select_dtypes(include=np.number).columns.tolist()
    anova_df = pd.DataFrame()
    for num in numerical_columns:
        new_row = df.anova(dv=num, between=target, detailed=False).round(4)
        new_row['Target'] = num
        new_row = new_row.rename(columns={'Target':'Source', 'Source':'Target'})
        anova_df = pd.concat([anova_df, new_row], axis='rows')
        anova_df['S.S. Diff'] = ['Yes' if p <= 0.05 else 'No' for p in anova_df['p-unc']]
    anova_df = anova_df[['Target', 'Source', 'ddof1', 'ddof2', 'F', 'p-unc', 'np2', 'S.S. Diff']]
    return anova_df

get_anova_df('Product', df)


numerical_columns = df.select_dtypes(include=np.number).columns.tolist()

pw_df = pg.pairwise_tests(dv='Age', between='Product', data=df).round(3)
pw_df['S.S. Diff'] = ['Yes' if p <= 0.05 else 'No' for p in pw_df['p-unc']]
pw_df['Eff Size'] =
pw_df