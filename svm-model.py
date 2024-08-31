import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import accuracy_score
#Import necessary modules
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

data = pd.read_csv("/content/weather_classification_data.csv")
data

#verifica valores duplicados e ausentes
data.isnull().sum()
data.duplicated().sum()

#converte categorias em valores
le_Cloud_Cover = LabelEncoder()
le_Season = LabelEncoder()

le_Location  = LabelEncoder()
le_Weather_Type  = LabelEncoder()


data['Cloud_Cover1'] = le_Cloud_Cover.fit_transform(data['Cloud Cover'])
data['Season1'] = le_Season.fit_transform(data['Season'])
data['Location1'] = le_Location.fit_transform(data['Location'])
data['Weather_Type1'] = le_Weather_Type.fit_transform(data['Weather Type'])

#exclui colunas antigas
df = data.drop(['Cloud Cover','Season','Weather Type','Location','Cloud_Cover1','Season1','Location1','Humidity','Wind Speed','Atmospheric Pressure'],axis='columns')
df

df.info()

#matrix de correlação
correlation_matrix = df.corr()
plt.figure(figsize=(10, 10))
sns.heatmap(correlation_matrix,annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()


#identifica outliers
def detect_outliers(df):
    outliers = {}
    for col in df.columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers[col] = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
    return outliers

# Get outliers for each column
outliers = detect_outliers(df)
for col, vals in outliers.items():
    print(f"Outliers in {col}:")
    print(vals)
    print("\n")

#plot outliers
def plot_outliers(df):
    plt.figure(figsize=(12, 6))

    for col in df.columns:
        #Calculate Q1, Q3 and IQR
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Identify outliers
        is_outlier = (df[col] < lower_bound) | (df[col] > upper_bound)

        #  Draw Scatter Plot
        plt.subplot(1, len(df.columns), df.columns.get_loc(col) + 1)
        plt.scatter(df.index, df[col], c=is_outlier, cmap='coolwarm', label='Outliers')
        plt.axhline(y=lower_bound, color='r', linestyle='--', label='Lower Bound')
        plt.axhline(y=upper_bound, color='r', linestyle='--', label='Upper Bound')
        plt.title(col)
        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.legend()

    plt.tight_layout()
    plt.show()

# Draw outliers
plot_outliers(df)

#remove outliers
from scipy import stats

# Remove outliers using Z-Score
df_cleaned = df[(np.abs(stats.zscore(df)) < 3).all(axis=1)]

# Number of rows before removing outliers
print(f"Number of rows before: {len(df)}")

# Application to remove outliers
z_scores = np.abs(stats.zscore(df))
df_cleaned = df[(z_scores < 3).all(axis=1)]

#  Number of rows after removing outliers
print(f"Number of rows after: {len(df_cleaned)}")

df_cleaned.info()

X = df.drop('Weather_Type1',axis='columns')
Y = df.Weather_Type1
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2)

categorical_features = X.select_dtypes(include=['object']).columns.tolist()
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', SVC())])

param_grid = {
    'classifier__C': [0.1, 1, 10],
    'classifier__gamma': [1, 0.1, 0.01],
    'classifier__kernel': ['linear', 'rbf']
}
grid = GridSearchCV(pipeline, param_grid, refit=True, verbose=2)
grid.fit(X_train, Y_train)

y_pred = grid.predict(X_test)
print("Melhores parâmetros: ", grid.best_params_)
print("Acurácia: ", accuracy_score(Y_test, y_pred))
print(classification_report(Y_test, y_pred))
