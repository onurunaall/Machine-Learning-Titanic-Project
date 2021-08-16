# -*- coding: utf-8 -*-

# Introduction
The sinking of Titanic is one of the most notorious shipwredcks in the history. In 1912, during her voyage, the titanic sank after colliding with an iceberg, killing 1502 out of 2224 passengers and crew. 

<font color = 'blue'>
Content:
    
1. [Load and Check Data](#1)
2. [Variable Description](#2)
    * [Univariate Variable Analysis](#3)    
        *    [Categorical Variable](#4)  
        *    [Numerical Variable](#5) 
3. [Basic Data Analysis](#6)  
4. [Outlier Detection](#7)
5. [Missing Value](#8) 
    * [Find Missing Value](#9)
    * [Fill Missing Value](#10)    
6. [Visualization](#11) 
    * [Correlation Between SibSp -- Parch -- Age -- Fare -- Survived](#12)   
    * [SibSp -- Survived](#13)    
    * [Parch -- Survived](#14) 
    * [Pclass -- Survived](#15)
    * [Age -- Survived](#16) 
    * [Pclass -- Survived -- Age](#17) 
    * [Embarked -- Sex -- Pclass -- Survived](#18) 
    * [Embarked -- Sex -- Fare -- Survived](#19)
    * [Fill Missing: Age Feature](#20) 
7. [Feature Engineering](#21)
    * [Name -- Title](#22)
    * [Family Size](#23)
    * [Embarked](#24)
    * [Ticket](#25)
    * [Pclass](#26)
    * [Sex](#27)
    * [Drop Passenger ID and Cabin](#28)
7. [Modeling](#29) 
    * [Train - Test Split](#30)
    * [Simple Logistic Regression](#31)
    * [Hyperparameter Tuning -- Grid Search -- Cross Validation](#32)
    * [Ensemble Modeling](#33)
    * [Prediction and Submission](#34)
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
plt.style.use("seaborn-whitegrid")

import seaborn as sns
from collections import Counter

import warnings
warnings.filterwarnings("ignore")

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

"""<a id = "1"></a><br>
# [Load and Check the Data]
"""

train_df = pd.read_csv("/kaggle/input/titanic/train.csv")
test_df = pd.read_csv("/kaggle/input/titanic/test.csv")
test_PassengerId = test_df["PassengerId"]

train_df.columns

train_df.head()

train_df.describe()

"""<a id = "2"></a><br>
# [Variable Description]
1. PassengerId: Unique d number to each passenger
2. Survived: Passenger survided(1) or died(0)
3. Pclass: Passenger class
4. Name: Name
5. Sex: Gender
6. Age: Age
7. SibSp: Number of siblings/spouses
8. Parch: Number of parents/children
9. Ticket: ticket number 
10. Fare: Amount of money paid for ticket
11. Cabin: Cabin category
12. Embarked: Port where passengers embarked, (C = cherbourg, Q = Queenstown, S=Southampton)
"""

train_df.info()

"""* float64(2): Fare and Age 
* int64(5): Pcalss, SibSp, Parch, PassengerId, Survived 
* object(5): Cabin, Embarked, Tickets, name and sex

<a id = "3"></a><br>
# Univariate Variable Analysis
* Categorical Variable: Survived, Sex, Pclass, Embarked, Cabin, Name, Ticket, SibSp, Parch
* Numerical Variable: Fare, Age, PassengerId

<a id = "4"></a><br>
## Categorical Variable:
"""

def bar_plot(variable):
    """
    input: variable ex: "Sex"
    output: bar plor & value count
    """
    # get feature
    var = train_df[variable]
    # count number of categorical variable(value)
    varValue = var.value_counts()
    
    # visualize
    plt.figure(figsize = (9,3))
    plt.bar(varValue.index, varValue)
    plt.xticks(varValue.index, varValue.index.values)
    plt.ylabel("Frequency")
    plt.title(variable)
    plt.show()
    print("{}: \n {}:".format(variable, varValue))

category1 = ["Survived", "Sex", "Pclass", "Embarked", "SibSp", "Parch"]
for c in category1:
    bar_plot(c)

category2 = ["Cabin", "Name", "Ticket"]
for c in category2:
    print("{} \n".format(train_df[c].value_counts()))

"""<a id = "5"></a><br>
## Numerical Variable:
"""

def plot_hist(variable):
    plt.figure(figsize = (9,3))
    plt.hist(train_df[variable], bins = 50)
    plt.xlabel(variable)
    plt.ylabel("Frequency")
    plt.title("{} distribution with hist".format(variable))
    plt.show

numericVar = ["Fare", "Age", "PassengerId"]
for n in numericVar:
    plot_hist(n)

"""<a id = "6"></a><br>
# Basic Data Analysis
* Pclass- Survived
* Sex - Survived
* SibSp - Survived
* Parch - Survived
"""

# Pclass vs Survived
train_df[["Pclass", "Survived"]].groupby(["Pclass"], as_index = False).mean().sort_values(by="Survived", ascending = False)

# Sex vs Survived
train_df[["Sex", "Survived"]].groupby(["Sex"], as_index = False).mean().sort_values(by="Survived", ascending = False)

# SibSp vs Survived
train_df[["SibSp", "Survived"]].groupby(["SibSp"], as_index = False).mean().sort_values(by="Survived", ascending = False)

# Parch vs Survived
train_df[["Parch", "Survived"]].groupby(["Parch"], as_index = False).mean().sort_values(by="Survived", ascending = False)

"""<a id = "7"></a><br>
# Outlier Detection
"""

def detect_outliers(df, features):
    outlier_indices = []
    for c in features:
        # 1st quartile
        Q1 = np.percentile(df[c], 25)
        # 3rd quartile
        Q3 = np.percentile(df[c], 75)
        # IQR
        IQR = Q3 - Q1
        # outlier Step
        outlier_step = IQR * 1.5
        # detect outlier and their indeces
        outlier_list_col = df[(df[c] < Q1 - outlier_step) | (df[c] > Q3 + outlier_step)].index
        # store indeces
        outlier_indices.extend(outlier_list_col)
        
    outlier_indices = Counter(outlier_indices) 
    # bir samplede ikiden fazla outlier varsa çıkart
    multiple_outliers = list(i for i, v in outlier_indices.items() if v > 2) 
    
    return multiple_outliers

train_df.loc[detect_outliers(train_df, ["Age", "SibSp", "Parch", "Fare"])]

# drop outliers
train_df = train_df.drop(detect_outliers(train_df, ["Age", "SibSp", "Parch", "Fare"]), axis = 0).reset_index(drop = True)

"""<a id = "8"></a><br>
# Missing Value
* Find Missing Value
* Fill Missing Value
"""

train_df_len = len(train_df)
train_df = pd.concat([train_df, test_df], axis = 0).reset_index(drop = True)

"""<a id = "9"></a><br>
## Find Missing Value
"""

train_df.columns[train_df.isnull().any()]

train_df.isnull().sum()

"""<a id = "10"></a><br>
## Fill Missing Value
* Embarked has 2 missing value
* Fare has only 1 missing value
"""

train_df[train_df["Embarked"].isnull()]

train_df.boxplot(column = "Fare", by = "Embarked")
plt.show()

train_df["Embarked"] = train_df["Embarked"].fillna("C")
train_df[train_df["Embarked"].isnull()]

train_df[train_df["Fare"].isnull()]

train_df["Fare"] = train_df["Fare"].fillna(np.mean(train_df[train_df["Pclass"] == 3]["Fare"]))
train_df[train_df["Fare"].isnull()]

"""<a id = "11"></a><br>
# Visualization

<a id = "12"></a><br>
 ## Correlation Between SibSp -- Parch -- Age -- Fare -- Survived
"""

list1 = ["SibSp","Parch","Age","Fare","Survived"]
sns.heatmap(train_df[list1].corr(), annot = True, fmt = ".2f")
plt.show()

"""Fare feature seems to have correlation with survived feature (0.26)

<a id = "13"></a><br>
 ## SibSp -- Survived
"""

g = sns.factorplot(x = "SibSp", y = "Survived", data = train_df, kind = "bar", size = 6)
g.set_ylabels("Survived Probability")
plt.show()

"""* Having a lot of SibSp have less cnhance to sruvive.
* If SibSp == 0 or 1 or 2, passenger has more chance to survive.
* We can consider a new feature describing these categories.

<a id = "14"></a><br>
 ## Parch -- Survived
"""

g = sns.factorplot(x = "Parch", y = "Survived", kind = "bar", data = train_df, size = 6)
g.set_ylabels("Survived Probabilty")
plt.show()

"""* SibSp and Parch can be used for new feature extraction  with threshold = 3
* Small families have more chance to survive.
* There is a standart deviation in survival of passenger with Parch = 3.

<a id = "15"></a><br>
 ## Pclass -- Survived
"""

g = sns.factorplot(x = "Pclass", y = "Survived", kind = "bar", data = train_df, size = 6)
g.set_ylabels("Survived Probabilty")
plt.show()

"""* Pclass stand for the class that passengers travel.
* Passenger that travels in 1st class have more chance to survive.

<a id = "16"></a><br>
 ## Age -- Survived
"""

g = sns.FacetGrid(train_df, col = "Survived")
g.map(sns.distplot, "Age", bins = 25)
plt.show()

"""* Children have more chance to survive accoridng to the 2nd graph age <= 10 has a high survival probabilty namely rate. 
* Also oldest passengers (60-80) survived at most.
* large number of 20 years old didn't survive in general, where as in general 35 year olds survived.
* most of the passengers are in 15-35 age range
* Gaussian distribution is obtained.
* use afe feaure in training
* use age distribution for missing value of age

<a id = "17"></a><br>
 ## Pclass -- Survived -- Age
"""

g = sns.FacetGrid(train_df, col = "Survived", row = "Pclass", size = 3)
g.map(plt.hist, "Age", bins = 25)
g.add_legend()
plt.show()

"""* Pclass is important feature for model training
* Pclass == 1 => survival rate is higher than death rate
* Pclass == 3 => survival rate is lower than death rate
* Pclass == 2 => survival rate is almost equal to death rate

<a id = "18"></a><br>
 ## Embarked -- Sex -- Pclass -- Survived
"""

g = sns.FacetGrid(train_df, row = "Embarked", size = 3)
g.map(sns.pointplot, "Pclass", "Survived", "Sex")
g.add_legend()
plt.show()

"""* Female passenger have more chance to survive, namely much better survival rate than males. This is supported by a further analysis above which is Sex vs Survived in Basic Analysis.
* Males have better survival rates in Pclass in C.
* Females have better survival rates in Pclass in S.
* Females have better survival rates in Pclass in Q.
* Embarked and Sex will be used ofr training.

<a id = "19"></a><br>
 ## Embarked -- Sex -- Fare -- Survived
"""

g = sns.FacetGrid(train_df, row = "Embarked", col = "Survived", size = 3)
g.map(sns.barplot, "Sex", "Fare")
g.add_legend()
plt.show()

"""* When fare goes up, survival rate also increases for passenger that deperated in S port. 
* When fare goes up, survival rate also increases for passenger that deperated in C port.
* In port Q, the difference is not strong as it is in S and C.
* Passengers who pay higher fare have better survival. Fare can be used as categorical for training.

<a id = "20"></a><br>
 ## Fill Missing: Age Feature
"""

train_df[train_df["Age"].isnull()]

sns.factorplot(x = "Sex", y = "Age", data = train_df, kind = "box" )
plt.show()

"""* Sex is not informative for age prediction. Because age distribution seems to be same with respect to sex."""

sns.factorplot(x = "Sex", y = "Age", hue= "Pclass", data = train_df, kind = "box")
plt.show()

"""* Pclass == 1 => age median is almost 40.
* Pclass == 2 => age median is almost 30.
* Pclass == 3 => age median is almost 25.
* So, the oldest passengers are tend to be in 1st class, whereas the youngest passengers tend to be 3rd class.
"""

sns.factorplot(x = "Parch", y = "Age", data = train_df, kind = "box")
sns.factorplot(x = "SibSp", y = "Age", data = train_df, kind = "box")
plt.show()

"""* If Parch is either 0 or 1, then passengers more likely to be around 20-30 year old. 
* If SibSp is either 0 or 1, then passengers more likely to be around 25-35 year old. 
"""

train_df["Sex"] = [1 if i == "male" else 0 for i in train_df["Sex"]]

sns.heatmap(train_df[["Age", "Sex", "SibSp", "Parch", "Pclass"]].corr(), annot = True)
plt.show()

"""* Age is not correlated with sex but it is correlated with Parch, SibSp and Pclass."""

index_nan_age = list(train_df["Age"][train_df["Age"].isnull()].index)
for i in index_nan_age:
    age_pred = train_df["Age"][((train_df["SibSp"] == train_df.iloc[i]["SibSp"]) & (train_df["Parch"] == train_df.iloc[i]["Parch"]) & (train_df["Pclass"] == train_df.iloc[i]["Pclass"]))].median()
    age_median = train_df["Age"].median()
    if not np.isnan(age_pred):
        train_df["Age"].iloc[i] = age_pred
    else:
        train_df["Age"].iloc[i] = age_median

train_df[train_df["Age"].isnull()]

"""<a id = "21"></a><br>
 # Feature Engineering

<a id = "22"></a><br>
 ## Name -- Title
"""

train_df["Name"].head(10)

name = train_df["Name"]
train_df["Title"] = [i.split(".")[0].split(",")[-1].strip() for i in name]

train_df["Title"].head()

sns.countplot(x = "Title", data = train_df)
plt.xticks(rotation = 60)
plt.show()

# convert to categorical 
train_df["Title"] = train_df["Title"].replace(["Lady", "the Countess", "Capt", "Don", "Dr", "Major", "Rev", "Sir", "Jonkheer", "Dona"], "other")
train_df["Title"] = [0 if i == "Master" else 1 if i == "Miss" or i == "Ms" or i == "Mlle" or i == "Mrs" else 2 if i == "Mr" else 3 for i in train_df["Title"] ]

sns.countplot(x = "Title", data = train_df)
plt.xticks(rotation = 60)
plt.show()

g = sns.factorplot(x = "Title", y = "Survived", data = train_df, kind = "bar")
g.set_xticklabels(["Master", "Mrs", "Mr", "Other"])
g.set_ylabels("Survival Probability")

train_df.drop(labels = ["Name"], axis = 1, inplace = True)

train_df.head()

train_df = pd.get_dummies(train_df, columns = ["Title"])
train_df.head()

""" <a id = "23"></a><br>
 ## Family Size
"""

train_df.head()

# The reason for adding 1 is even passenger has no realtive he/she is still a family
train_df["Fsize"] = train_df["SibSp"] + train_df["Parch"] + 1

g = sns.factorplot(x = "Fsize", y = "Survived", data = train_df, kind = "bar")
g.set_ylabels("Survival")
plt.show()

"""* Family size 4'e kdara artış gösteririken 5 olunca aniden düşüş yaşıyor."""

train_df["family_size"] = [1 if i < 5 else 0 for i in train_df["Fsize"]]

train_df.head(10)

sns.countplot( x = "family_size", data = train_df)
plt.show()

g = sns.factorplot(x = "family_size", y = "Survived", data = train_df, kind = "bar")
g.set_ylabels("Survival")
plt.show()

"""* Big families have a lower chance to survive.
* Small families have more chance to survive than large families.
"""

train_df = pd.get_dummies(train_df, columns = ["family_size"])

""" <a id = "24"></a><br>
 ## Embarked
"""

train_df["Embarked"].head(10)

sns.countplot(x = "Embarked", data = train_df)
plt.show()

train_df = pd.get_dummies(data = train_df, columns = ["Embarked"])

train_df.head()

""" <a id = "25"></a><br>
 ## Ticket
"""

train_df["Ticket"].head(10)

tickets = []
for i in list(train_df.Ticket):
    if not i.isdigit():
        tickets.append(i.replace(".", "").replace("/","").strip().split(" ")[0])
    else:
        tickets.append("x")
train_df["Ticket"] = tickets

train_df["Ticket"].head(10)

train_df = pd.get_dummies(train_df, columns = ["Ticket"], prefix = "T")
train_df.head(10)

""" <a id = "26"></a><br>
 ## Pclass
"""

sns.countplot(x = "Pclass", data = train_df)
plt.show()

train_df["Pclass"] = train_df["Pclass"].astype("category")
train_df = pd.get_dummies(train_df, columns = ["Pclass"])
train_df.head()

""" <a id = "27"></a><br>
 ## Sex
"""

train_df["Sex"] = train_df["Sex"].astype("category")
train_df = pd.get_dummies(train_df, columns = ["Sex"])
train_df.head()

""" <a id = "28"></a><br>
 ## Drop Passenger ID and Cabin
"""

train_df.drop(labels = ["PassengerId", "Cabin"], axis = 1, inplace = True)

train_df.columns

""" <a id = "29"></a><br>
 # Modeling
"""

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

""" <a id = "30"></a><br>
 ## Train - Test Split
"""

train_df_len

test = train_df[train_df_len:]
test.drop(labels = ["Survived"], axis = 1, inplace = True)

train = train_df[:train_df_len]
X_train = train.drop(labels = ["Survived"], axis = 1)
y_train = train["Survived"]
# now X_test and y_test means X_validation and y_validation
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size = 0.33, random_state = 42)
print("X_train", len(X_train))
print("X_test", len(X_test))
print("y_train", len(y_train))
print("y_test", len(y_test))
print("test", len(test))

""" <a id = "31"></a><br>
 ## Simple Logistic Regression
"""

logreg = LogisticRegression()
logreg.fit(X_train, y_train)
acc_log_train = round(logreg.score(X_train, y_train) * 100, 2)
acc_log_test = round(logreg.score(X_test, y_test) * 100, 2)
print("Training Accuracy: % {}".format(acc_log_train))
print("Testing Accuracy: % {}".format(acc_log_test))

""" <a id = "32"></a><br>
 ## Hyperparameter Tuning -- Grid Search -- Cross Validation
 * We will compare 5 ML classifier and evaluate mean accuracy of each of them by stratified cross validation
 * Decision Tree
 * SVM
 * Random Forest
 * KNN
 * Logistic Regression
"""

random_state = 42
classifier = [DecisionTreeClassifier(random_state = random_state),
             SVC(random_state = random_state),
             RandomForestClassifier(random_state = random_state),
             LogisticRegression(random_state = random_state),
             KNeighborsClassifier()]

dt_param_grid = {"min_samples_split" : range(10,500,20),
                "max_depth": range(1,20,2)}

svc_param_grid = {"kernel" : ["rbf"],
                 "gamma": [0.001, 0.01, 0.1, 1],
                 "C": [1,10,50,100,200,300,1000]}

rf_param_grid = {"max_features": [1,3,10],
                "min_samples_split":[2,3,10],
                "min_samples_leaf":[1,3,10],
                "bootstrap":[False],
                "n_estimators":[100,300],
                "criterion":["gini"]}

logreg_param_grid = {"C":np.logspace(-3,3,7),
                    "penalty": ["l1","l2"]}

knn_param_grid = {"n_neighbors": np.linspace(1,19,10, dtype = int).tolist(),
                 "weights": ["uniform","distance"],
                 "metric":["euclidean","manhattan"]}
classifier_param = [dt_param_grid,
                   svc_param_grid,
                   rf_param_grid,
                   logreg_param_grid,
                   knn_param_grid]

cv_result = []
best_estimators = []
for i in range(len(classifier)):
    clf = GridSearchCV(classifier[i], param_grid=classifier_param[i], cv = StratifiedKFold(n_splits = 10), scoring = "accuracy", n_jobs = -1,verbose = 1)
    clf.fit(X_train,y_train)
    cv_result.append(clf.best_score_)
    best_estimators.append(clf.best_estimator_)
    print(cv_result[i])

cv_results = pd.DataFrame({"Cross Validation Means":cv_result, "ML Models":["DecisionTreeClassifier",
                                                                            "SVM",
                                                                            "RandomForestClassifier",
                                                                            "LogisticRegression",
                                                                            "KNeighborsClassifier"]})

g = sns.barplot("Cross Validation Means", "ML Models", data = cv_results)
g.set_xlabel("Mean Accuracy")
g.set_title("Cross Validation Scores")
plt.show()

""" <a id = "33"></a><br>
## Ensemble Modeling
"""

votingC = VotingClassifier(estimators = [("dt",best_estimators[0]),
                                        ("rfc",best_estimators[2]),
                                        ("lr",best_estimators[3])],
                                        voting = "soft", n_jobs = -1)
votingC = votingC.fit(X_train, y_train)
print(accuracy_score(votingC.predict(X_test),y_test))

"""<a id = "34"></a><br>
## Prediction and Submission
"""

test_survived = pd.Series(votingC.predict(test), name = "Survived").astype(int)
results = pd.concat([test_PassengerId, test_survived],axis = 1)
results.to_csv("titanic.csv", index = False)

test_survived
