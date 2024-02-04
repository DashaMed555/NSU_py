import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression


def get_accuracy(prediction, fact):
    res = (prediction == fact).astype('int')
    right_num = sum(res)
    total_num = len(res)
    accuracy = right_num / total_num
    return accuracy


def main():
    # 1
    data = pd.read_csv("titanic_prepared.csv")
    label = data["label"]
    data = data.drop("label", axis=1)
    data_train, data_test, label_train, label_test = train_test_split(data, label, test_size=0.1)

    # 2
    model_rf = RandomForestClassifier()
    model_xgb = XGBClassifier()
    model_lr = LogisticRegression(solver="liblinear")

    model_rf.fit(data_train, label_train)
    model_xgb.fit(data_train, label_train)
    model_lr.fit(data_train, label_train)

    # 3
    prediction_rf = model_rf.predict(data_test)
    prediction_xgb = model_xgb.predict(data_test)
    prediction_lr = model_lr.predict(data_test)

    accuracy_rf = get_accuracy(prediction_rf, label_test)
    accuracy_xgb = get_accuracy(prediction_xgb, label_test)
    accuracy_lr = get_accuracy(prediction_lr, label_test)

    # 4
    print(f"accuracy_rf = {accuracy_rf}")
    print(f"accuracy_xgb = {accuracy_xgb}")
    print(f"accuracy_lr = {accuracy_lr}", '\n')

    # 5
    importances = model_rf.feature_importances_
    features = data.columns
    indices = np.argsort(importances)[::-1]
    important = features[indices][:2]

    plt.title('Важность признаков')
    plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    plt.yticks(range(len(indices)), features[indices])
    plt.xlabel('Относительная важность')
    plt.show(block=False)
    if plt.waitforbuttonpress():
        plt.close()

    model_rf.fit(data_train[important], label_train)
    model_xgb.fit(data_train[important], label_train)
    model_lr.fit(data_train[important], label_train)

    prediction_rf = model_rf.predict(data_test[important])
    prediction_xgb = model_xgb.predict(data_test[important])
    prediction_lr = model_lr.predict(data_test[important])

    accuracy_rf = get_accuracy(prediction_rf, label_test)
    accuracy_xgb = get_accuracy(prediction_xgb, label_test)
    accuracy_lr = get_accuracy(prediction_lr, label_test)

    print(f"accuracy_rf on important features = {accuracy_rf}")
    print(f"accuracy_xgb on important features = {accuracy_xgb}")
    print(f"accuracy_lr on important features = {accuracy_lr}")


main()
