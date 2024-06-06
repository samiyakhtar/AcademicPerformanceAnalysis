# Import Statements
import pandas as pd
import numpy as np
import plotly
from plotly import tools
import seaborn as sns 
import matplotlib.pyplot as plt

# Reading in CSV file
df = pd.read_csv("student-mat.csv")
df_backup = pd.read_csv("student-mat.csv")
df.head()

num_examples, num_features = df.shape

print(f"Number of examples: {num_examples}")
print(f"Number of features: {num_features}")
 
# Removing irrelevant and duplicate data
duplicates = df.duplicated()
num_duplicates = duplicates.sum()
print(f"Number of duplicate rows: {num_duplicates}")
duplicate_rows = df[duplicates]
print(duplicate_rows)

# Handle missing data 
total_missing = df.isnull().sum().sum()
print(f"Total number of missing values in the entire DataFrame: {total_missing}")
any_missing = df.isnull().any().any()
if any_missing:
    print("There are missing values in the DataFrame.")
else:
    print("There are no missing values in the DataFrame.")

# Filtering Outliers
outlier_mask = pd.Series([False] * len(df))

for column in df.select_dtypes(include=['float64', 'int64']).columns:
    lower_bound = df[column].quantile(0.01)
    upper_bound = df[column].quantile(0.99)
    
    outlier_mask |= (df[column] < lower_bound) | (df[column] > upper_bound)

filtered_df = df[~outlier_mask]

print(f"Original DataFrame size: {df.shape}")
print(f"Filtered DataFrame size: {filtered_df.shape}")
df = filtered_df

# Heatmap
plt.figure(figsize = (20, 7))
sns.heatmap(df.corr(numeric_only = True), cmap = 'crest', annot = True)

df.hist(bins = 50, figsize=(20,10), color = 'b')
plt.show()

# Histograms
import cufflinks as cf
from plotly.offline import download_plotlyjs, init_notebook_mode, iplot
from plotly.graph_objs import *
init_notebook_mode()
cf.set_config_file(offline=False, world_readable=True, theme='pearl')

x0 = df_backup["G1"]
x1 = df_backup["G2"]
x2 = df_backup["G3"]

First_Period = Histogram(
    x=x0,
    name="First Semester",
    marker= dict(
        color='#F79F81',
    )
)

Second_Period = Histogram(
    x=x1,
    name="Second Semester",
    marker= dict(
        color='#9FF781',
    )
)

Third_Period = Histogram(
    x=x2,
    name="Third Semester",
    marker= dict(
        color='#CED8F6',

    )
)

data = [First_Period, Second_Period, Third_Period]
layout = Layout(barmode='stack',
                  title="Distribution of Student's Grades",
                   font=dict(size=16),
                  xaxis=dict(
                  title="Grades"
                  ),
                  yaxis=dict(
                  title="Number of Students"))

fig = dict(data=data, layout=layout)
iplot(fig)

# Implementing PCA and StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

features_alcohol = ['Walc', 'Dalc']
features_education = ['Fedu', 'Medu']
scaler = StandardScaler()
df_scaled_alcohol = scaler.fit_transform(df[features_alcohol])
df_scaled_education = scaler.fit_transform(df[features_education])
pca_alcohol = PCA(n_components=1)
principal_component_alcohol = pca_alcohol.fit_transform(df_scaled_alcohol)
df['principal_component_alcohol'] = principal_component_alcohol
pca_education = PCA(n_components=1)
principal_component_education = pca_education.fit_transform(df_scaled_education)
df['principal_component_education'] = principal_component_education
df = df.drop(['Walc', 'Dalc', 'Medu', 'Fedu'], axis=1)

# Implementing train test split for RandomForestRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

X = df.drop(['G1', 'G2', 'G3'], axis=1)
y = df['G3']

X = pd.get_dummies(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

feature_importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)

print("Feature Importances:\n", feature_importances)


features_to_keep = [
    "absences", "failures", "principal_component_education", "goout", "principal_component_alcohol",
    "age", "famrel", "health", "freetime", "traveltime", "studytime", "Mjob", "schoolsup", "guardian", 
    "sex", "Fjob", "reason"
    
]

features_to_keep = list(set(features_to_keep))

# Select the specified columns from the DataFrame
df = df[features_to_keep]
print(f"Filtered DataFrame shape: {df.shape}")

df.loc[:, 'G3'] = df_backup.loc[df.index, 'G3']


df.to_csv('filtered_dataframe_with_grades.csv', index=False)
plt.figure(figsize = (20, 7))
sns.heatmap(df.corr(numeric_only = True), cmap = 'crest', annot = True)

columns_except_G3 = [col for col in df.columns if col != 'G3']
final_column_order = columns_except_G3 + ['G3']
df = df[final_column_order]
df.head()

# Creating new CSV with PCA filtered DF
df.to_csv('pca_filtered_dataframe_with_grades.csv', index=False)

# Decision Tree Regressor
import pandas as pd
df = pd.read_csv('pca_filtered_dataframe_with_grades.csv')
df.head()

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

categorical_columns = ['Mjob', 'schoolsup', 'Fjob', 'sex', 'reason', 'guardian']
for col in categorical_columns:
    df[col] = LabelEncoder().fit_transform(df[col])

X = df.drop('G3', axis=1)
y = df['G3']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
decision_tree_regressor = DecisionTreeRegressor(random_state=42)
decision_tree_regressor.fit(X_train, y_train)
y_pred = decision_tree_regressor.predict(X_test)

# Finding MSE RMSE and R2 Score from Decision Tree Regressor
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R-squared (R²) score: {r2}")

# Random Forest Regressor
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

categorical_columns = ['Mjob', 'schoolsup', 'Fjob', 'sex', 'reason', 'guardian']
for col in categorical_columns:
    df[col] = LabelEncoder().fit_transform(df[col])

X = df.drop('G3', axis=1)
y = df['G3']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
random_forest_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
random_forest_regressor.fit(X_train, y_train)
y_pred = random_forest_regressor.predict(X_test)

# Finding MSE RMSE and R2 Score from Random Forest Regressor
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R-squared (R²) score: {r2}")

# Implementing RandomizedSearchCV to find best params
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

param_distributions = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': [2, 5, 10, 15],
    'min_samples_leaf': [1, 2, 4, 6],
    'max_features': ['sqrt', 'log2']  # Corrected 'auto' to 'sqrt'
}


rf_regressor = RandomForestRegressor(random_state=42)
rf_random_search = RandomizedSearchCV(estimator=rf_regressor,
                                      param_distributions=param_distributions,
                                      n_iter=100,
                                      cv=5,
                                      verbose=2,
                                      random_state=42,
                                      n_jobs=-1)

X = df.drop('G3', axis=1)
y = df['G3']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf_random_search.fit(X_train, y_train)
best_params = rf_random_search.best_params_
best_model = rf_random_search.best_estimator_
y_pred = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Best parameters: {best_params}")
print(f"Mean Squared Error (MSE) on Test Set: {mse}")
print(f"Root Mean Squared Error (RMSE) on Test Set: {rmse}")
print(f"R-squared (R²) score on Test Set: {r2}")

# Implementing Logistic Regression Model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score

# Reading in the Cleaned DF Again for Logistic Regression as: DF_LogReg
import pandas as pd
DF_LogReg = pd.read_csv('pca_filtered_dataframe_with_grades.csv')
DF_LogReg.head()

# Pre-Processing Steps for Logistic Regression Model
# Finding Categorical Variables:
categorical = [var for var in DF_LogReg.columns if DF_LogReg[var].dtype=='O']

print('There are {} categorical variables\n'.format(len(categorical)))
print('The categorical variables are :', categorical)
DF_LogReg[categorical].head()

# Label Encoding the Categorical Columns
categorical_columns = ['Mjob', 'schoolsup', 'Fjob', 'sex', 'reason', 'guardian']
for col in categorical_columns:
    DF_LogReg[col] = LabelEncoder().fit_transform(DF_LogReg[col])


# Converting Target Variable ('G3') into Categorical Variable 
DF_LogReg['grade_status'] = DF_LogReg['G3'].apply(lambda x: '0' if x < 12 else '1')
DF_LogReg

DF_LogReg.drop('G3', axis =1 , inplace = True)

# Splitting X and Y Variables: Dropping Target ('grade_Status') and Labeling X and Y
# Features
X = DF_LogReg.drop('grade_status', axis = 1)

# Target variable
y = DF_LogReg['grade_status']

pip install yellowbrick

classes=DF_LogReg['grade_status'].unique()

# Importing ClassBalance visualizer from Yellowbrick 
from yellowbrick.target import ClassBalance

# Creating the class imbalance plot
visualizer = ClassBalance(labels=classes)

visualizer.fit(y)
visualizer.ax.set_xlabel("Grade Status")
visualizer.ax.set_ylabel("Student Counts")

# Train Test Split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 4)

# Scaling
from sklearn.metrics import recall_score, precision_score, roc_auc_score, roc_curve
from sklearn.preprocessing import MinMaxScaler

# Creating a scaler object
scaler = MinMaxScaler()

# Fitting and transforming the data
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns = X_train.columns)
X_test = pd.DataFrame(scaler.fit_transform(X_test), columns = X_test.columns)

# Displaying the Coefficients of each variable as well as the intercept 
log_reg = LogisticRegression() 
log_reg.fit(X_train, y_train) 

print('intercept ', log_reg.intercept_[0])
print('classes', log_reg.classes_)
pd.DataFrame({'coeff': log_reg.coef_[0]}, index=X_train.columns)

# Predicting + Retrieving Accuracy Metrics for Baseline Logistic Regression Model
y_pred = log_reg.predict(X_test) 
print('Accuracy of logistic regression classifier on train set: {:.2f}'.format(log_reg.score(X_train, y_train))) 
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(log_reg.score(X_test, y_test))) 
print('Recall of logistic regression classifier on test set: {:.2f}'.format(recall_score(y_test, y_pred,  average='macro'))) 
print('Precision of logistic regression classifier on test set: {:.2f}'.format(precision_score(y_test, y_pred,  average='macro'))) 

# Creating a Classification Report for Baseline Logistic Regression Model
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred, target_names=['0', '1']))

# Creating an ROC Plot for The Baseline Logistic Regression Model
import sklearn.metrics as metrics
probs = log_reg.predict_proba(X_test)
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(y_test, preds, pos_label = '1')
roc_auc = metrics.auc(fpr, tpr)

import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

# Classification Matrix Visualization for Baseline Logistic Regression 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics

cm = metrics.confusion_matrix(y_test, y_pred)

plt.figure(figsize=(9,9))
sns.heatmap(cm, annot=True, fmt=".0f", linewidths=.5, square = True, cmap = 'Blues_r');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Accuracy Score: {0}'.format(log_reg.score(X_test, y_test))
plt.title(all_sample_title, size = 15);


# Creating K-Fold Cross-Validation
from sklearn.model_selection import KFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)
lgr = LogisticRegression() 

# Performing K-Fold Cross-Validation
from sklearn.model_selection import cross_val_score

cv_scores_baseline = cross_val_score(lgr, X, y, cv=kf, scoring="accuracy")

# Printing CV scores
print("CV scores:", cv_scores_baseline)
print('Minimum score: ', round(cv_scores_baseline.min(), 4))
print('Maximum score: ', round(cv_scores_baseline.max(), 4))

# Averaging CV scores
import numpy as np
print("Average score:", np.round(cv_scores_baseline.mean(), 4))