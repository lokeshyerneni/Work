import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

warnings.filterwarnings('ignore')
import statsmodels.api as sm

pd.set_option("display.width", 400)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)

df2013 = pd.read_csv("2013Investments.csv", encoding="unicode_escape")
df2015 = pd.read_csv("2015Investments.csv", encoding="unicode_escape")

df2013 = df2013[pd.notnull(df2013["raised_amount_usd"])]
df2015 = df2015[pd.notnull(df2015["raised_amount_usd"])]

df2015_company = df2015.groupby(["company_name"])["raised_amount_usd"].sum().sort_values(ascending=False)
df2015_investor = df2015.groupby(["investor_name"])["raised_amount_usd"].sum().sort_values(ascending=False)
df2015_locations = df2015.groupby("company_city")["raised_amount_usd"].sum().sort_values(ascending=False)
df2015_category = df2015.groupby(["company_category_list"])["raised_amount_usd"].sum().sort_values(ascending=False)
df2015_fundingType = df2015.groupby(["funding_round_type", "funding_round_code"])["raised_amount_usd"].sum().sort_values(ascending=False)

df2015_fundingType = df2015_fundingType.reset_index()
df2015_fundingType["fundingType"] = df2015_fundingType["funding_round_type"] + " " + df2015_fundingType["funding_round_code"]
del df2015_fundingType["funding_round_type"], df2015_fundingType["funding_round_code"]
df2015_fundingType = df2015_fundingType.groupby(["fundingType"])["raised_amount_usd"].sum().sort_values(ascending=False)


df2013_company = df2013.groupby(["company_name"])["raised_amount_usd"].sum().sort_values(ascending=False)
df2013_investor = df2013.groupby(["investor_name"])["raised_amount_usd"].sum().sort_values(ascending=False)
df2013_locations = df2013.groupby("company_city")["raised_amount_usd"].sum().sort_values(ascending=False)
df2013_category = df2013.groupby(["company_category_code"])["raised_amount_usd"].sum().sort_values(ascending=False)
df2013_fundingType = df2013.groupby(["funding_round_type"])["raised_amount_usd"].sum().sort_values(ascending=False)


figure = plt.figure()
figure.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)

plt.subplot(2, 1, 1)
plt.plot(df2015_fundingType)
plt.grid()
plt.title("Top 10 Funding Type that were utilized most amount in 2015")
plt.setp(plt.subplot(211).get_xticklabels(), rotation=30, horizontalalignment='right')

plt.subplot(2, 1, 2)
plt.plot(df2013_fundingType)
plt.grid()
plt.title("Top 10 Funding Type that were utilized most amount in 2013")
plt.setp(plt.subplot(212).get_xticklabels(), rotation=30, horizontalalignment='right')

figure.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
plt.ylabel("Amount raised")

plt.show()

# df2013 = df2013[pd.notnull(df2013["raised_amount_usd"])]
# df2015 = df2015.set_index(["company_name", "investor_name"])
# dfTotalAmountRaised = df2013.groupby(["funding_round_type"])["raised_amount_usd"].sum().sort_values(ascending=False)
# print(dfTotalAmountRaised)
# dfTotalAmountRaised.to_csv(r"C:\Users\lokes\OneDrive\Documents\Internship\Findings\typeOfFundingRaised2013.csv")

# df2015 = df2015.set_index(["company_name", "investor_name"])
#dfTotalAmountRaised = df2015.groupby(["funding_round_type"])["raised_amount_usd"].sum().sort_values(ascending=False)
#print(dfTotalAmountRaised)
# dfTotalAmountRaised.to_csv(r"C:\Users\lokes\OneDrive\Documents\Internship\Findings\typeOfFundingRaised2013.csv")
