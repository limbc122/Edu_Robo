import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
import graphviz
import os

seed = 7
model = DecisionTreeClassifier(max_depth=5, random_state=seed)

train = pd.read_csv("./data/train.csv", index_col="PassengerId")
test = pd.read_csv("./data/test.csv", index_col="PassengerId")

# pd.pivot_table(data=train, index='Sex', values='Survived')
# sns.countplot(data=train, x='Sex', hue='Survived')

# low_fare = train[train['Fare'] < 100]
# survived = low_fare[low_fare['Survived'] == 1]
# dead = low_fare[low_fare['Survived'] == 0]

# sns.distplot(survived['Fare'], hist=False, label='S')
# sns.distplot(dead['Fare'], hist=False, label='D')

test.loc[pd.isnull(test['Fare']), 'Fare'] = train['Fare'].mean()
test.loc[pd.isnull(test['Age']), 'Age'] = train['Age'].mean()

train.loc[pd.isnull(train['Fare']), 'Fare'] = train['Fare'].mean()
train.loc[pd.isnull(train['Age']), 'Age'] = train['Age'].mean()

train.loc[train['Sex'] == 'male', 'Sex_revised'] = 0
train.loc[train['Sex'] != 'male', 'Sex_revised'] = 1

test.loc[test['Sex'] == 'male', 'Sex_revised'] = 0
test.loc[test['Sex'] != 'male', 'Sex_revised'] = 1

train['Embarked_C'] = train['Embarked'] == 'C'
train['Embarked_S'] = train['Embarked'] == 'S'
train['Embarked_Q'] = train['Embarked'] == 'Q'

test['Embarked_C'] = test['Embarked'] == 'C'
test['Embarked_S'] = test['Embarked'] == 'S'
test['Embarked_Q'] = test['Embarked'] == 'Q'

train['Cabin_Tp'] = pd.isnull(train['Cabin'])
test['Cabin_Tp'] = pd.isnull(test['Cabin'])

train['Single'] = train['SibSp'] + train['Parch']
test['Single'] = test['SibSp'] + test['Parch']

feature_names = ['Sex_revised', 'Pclass', 'Fare', 'Embarked_C', 'Embarked_S', 'Embarked_Q', 'Cabin_Tp', 'Single', 'Age']

X_train = train[feature_names]
X_test = test[feature_names]

y_train = train['Survived']

model.fit(X_train, y_train)

prediction = model.predict(X_test)

export_graphviz(model, feature_names=feature_names, class_names=['Dead','Survived'], out_file='decision_tree.dot')

with open('decision_tree.dot') as f:
    dot_graph = f.read()

# graphviz.Source(dot_graph)

figure, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3)
figure.set_size_inches(18,4)
sns.countplot(data=train, x='Embarked', hue='Survived', ax=ax1)
sns.countplot(data=train, x='Sex', hue='Survived', ax=ax2)
sns.countplot(data=train, x='Pclass', hue='Survived', ax=ax3)

# submit = pd.read_csv('./data/gender_submission.csv', index_col='PassengerId')
# submit['Survived'] = prediction
# submit.to_csv('./data/submit.csv')