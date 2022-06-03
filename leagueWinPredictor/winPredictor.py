import pandas as pd
import numpy as np
import sklearn
from pandas import DataFrame
from sklearn import linear_model
from sklearn.linear_model import RidgeClassifier

df = pd.read_csv('MatchTimelinesFirst15.csv')

predict = "blue_win"

df = df.drop('Unnamed: 0', axis=1)
df = df.drop('redDragonKills', axis=1)
df = df.drop('blueDragonKills', axis=1)

# newDf = DataFrame(df, columns=['blue_win', 'blueGold', 'blueAvgLevel'])
# print(df.describe())

new_blueWin_df = {
    'blueWin': df[predict].apply(lambda x: 1 if x == 1 else -1)
}

x = np.array(df.drop([predict], axis=1))
y = np.array(new_blueWin_df['blueWin'])

for _ in range(100):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

    #print('{0}, {1}'.format(type(x_train), x_train))

    ridge = linear_model.RidgeClassifier()
    # trains model
    ridge.fit(x_train, y_train)

    acc = ridge.score(x_test, y_test)

    print('Acc {0}'.format(acc))
