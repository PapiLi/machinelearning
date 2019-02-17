import pandas as pd
import numpy as np

data=pd.read_csv('adult.data')
"""
data.reindex(columns=["age", "workclass", "fnlwgt", "education",  "education_num", "marital_status", "occupation", "relationship", "race","sex", "capital_gain",
                               "capital_loss", "hours_per_week","country", "income"])
"""
print(data.columns)

print(data.shape[0])

#去除多余属性
data.drop(['fnlwgt','marital_status','relationship','capital_gain', 'capital_loss','race','country'],axis=1, inplace=True)

print(data.columns)

#删除缺
#data.replace(' ?',np.nan)
#data.dropna()
data.drop(data[data["workclass"] == ' ?'].index.tolist(),axis=0, inplace=True)
data.drop(data[data["occupation"] == ' ?'].index.tolist(),axis=0, inplace=True)
print(data.shape[0])
#将离散值改为数值
d_workclass=pd.get_dummies(data["workclass"],prefix="workclass")
d_education=pd.get_dummies(data["education"],prefix="education")
d_occupation = pd.get_dummies(data["occupation"], prefix = "occupation")
#d_country = pd.get_dummies(data["country"], prefix = "country")
d_sex = pd.get_dummies(data["sex"], prefix = "sex")
data.drop(["workclass", "education", "occupation",  "sex"], axis=1, inplace=True)
data = pd.concat([data, d_workclass, d_education, d_occupation, d_sex], axis=1)

print(data.columns)

#标准化
def standardize(X):
    X_std = np.zeros(X.shape)
    mean = np.mean(X, axis = 0)
    # X.mean(axis = 0)
    std = np.std(X, axis = 0)
    X_std = (X - mean)/std

    return X_std

age_scale = standardize(np.array(data["age"]))
education_num_scale = standardize(np.array(data["education_num"]))
hours_per_week_scale = standardize(np.array(data["hours_per_week"]))
data["age_scale"] = age_scale
data["education_num_scale"] = education_num_scale
data["hours_per_week_scale"] = hours_per_week_scale

data.drop(["age", "education_num", "hours_per_week"], axis=1, inplace=True)


#类别修改
data.ix[data['income'] == ' <=50K', 'income'] = -1
data.ix[data['income'] == ' >50K', 'income'] = 1

print(data.shape[0])
print(data.shape[1])

data.to_csv('adult_for.data', index=False)