import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
import random
import numpy as np
import math
import matplotlib.pyplot as plt
import time

def stratify_sampling(df):
    X = df.iloc[:, df.columns != 'class']
    y = df[['class']]

    acc_DT = []
    acc_RF = []
    time_DT = []
    time_RF = []

    for i in range(100):
        Train_x, Test_x, Train_y, Test_y = train_test_split(X, y, stratify=y, test_size=0.4, random_state=random.randint(0, 100000))

        start_time = time.time()
        model = DecisionTreeClassifier()
        model.fit(Train_x, Train_y)
        predictions = model.predict(Test_x)
        acc_DT.append(metrics.accuracy_score(Test_y, predictions))
        time_DT.append(time.time() - start_time)
        print("Do chinh xac  Decision Tree:", metrics.accuracy_score(Test_y, predictions))

        start_time = time.time()
        rf_model = RandomForestClassifier(n_estimators=100, max_features=int(math.sqrt(X.shape[1])) + 1)
        rf_model.fit(Train_x, Train_y.values.ravel())
        pred_y = rf_model.predict(Test_x)
        acc_RF.append(metrics.accuracy_score(Test_y, pred_y))
        time_RF.append(time.time() - start_time)
        print("Do chinh xac Random Forest:", metrics.accuracy_score(Test_y, pred_y))

    print("Do chinh xac trung binh Decision Tree:", sum(acc_DT) / len(acc_DT))
    print("Do chinh xac trung binh Random Forest:", sum(acc_RF) / len(acc_RF))
    print("Thoi gian trung binh Decision Tree:", sum(time_DT) / len(time_DT))
    print("Thoi gian trung binh Random Forest:", sum(time_RF) / len(time_RF))

    results = [acc_DT, acc_RF]
    names = ('Decision Tree', 'Random Forest')

    fig = plt.figure()
    fig.suptitle('Algorithm Comparison')
    plt.boxplot(results, labels=names)
    plt.ylabel('Accuracy')

    plt.show()


def main():
    df = pd.read_csv('C:\\Users\\user\\Downloads\\Prostate.csv')
    df.columns.values[0] = "class"
    stratify_sampling(df)


if __name__ == "__main__":
    main()
