# Cannot use KNN due to the amount of data it contains (49271 for df2013 and 13748 for df2015)
# Cannot use logistic regression as the dataset is heavily multi-categorical

# Unique data points in df2013
# Company_name: 10357
# Company_category_code: 43
# Company_country_code: 1
# Company_state_code: 50
# Company_region: 501
# Company_city: 1130
# Investor_name: 9815
# Investor_Category_Code: 33
# Investor_Country_Code: 71
# Investor_State_Code: 51
# investor_region: 562
# investor_city: 956
# funding_round_type: 9
# funded_at: 2769
# funded_quarter: 71
# funded_month: 188
# funded_year: 20
# raised_amount_usd: 1458

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import warnings
from sklearn import svm
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')
import statsmodels.api as sm

pd.set_option("display.width", 400)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)

df2013 = pd.read_csv("2013Investments.csv", encoding="unicode_escape")
df2015 = pd.read_csv("2015Investments.csv", encoding="unicode_escape")

df2013 = df2013[pd.notnull(df2013["raised_amount_usd"])]
df2015 = df2015[pd.notnull(df2015["raised_amount_usd"])]
df2013 = df2013.dropna(how="any", subset=["company_category_code", "company_city"])

del df2013["company_country_code"], df2013["company_permalink"], df2013["investor_permalink"], df2013[
    "investor_category_code"], df2013["investor_state_code"], df2013["investor_country_code"], df2013["investor_city"], \
    df2013["funded_at"], df2013["funded_month"], df2013["funded_quarter"], df2013["company_name"], df2013[
    "company_city"], df2013["company_state_code"]
df2013 = df2013.sample(frac=1)
df2013 = df2013.apply(LabelEncoder().fit_transform)
X = df2013[df2013.columns[1:]]
y = df2013["company_category_code"]
print(len(X))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
model = svm.SVC(C=1, kernel="linear", gamma=.001)
model.fit(X_train, y_train)
print(len(X_train))
y_pred = model.predict(X_test)
print("Precision: ", metrics.precision_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
