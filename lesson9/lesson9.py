import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier


def task1():
    data = pd.read_csv("train.csv")
    label = data["Survived"]
    data.drop(["Survived", "PassengerId", "Name", "Ticket", "Cabin"], axis=1, inplace=True)

    df_sex = pd.get_dummies(data["Sex"]).astype('int')
    df_pclass = pd.get_dummies(data["Pclass"], prefix="Class").astype('int')
    df_emb = pd.get_dummies(data["Embarked"], prefix="Emb").astype('int')
    data.drop(["Sex", "Pclass", "Embarked"], axis=1, inplace=True)
    data = pd.concat([data, df_sex, df_pclass, df_emb], axis=1)
    data.fillna(data.median(), inplace=True)

    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    data_train, data_test, label_train, label_test = train_test_split(data_scaled, label, test_size=0.2)
    data_train, data_val, label_train, label_val = train_test_split(data_train, label_train, test_size=0.2)

    # Random Forest
    rf_accuracy = pd.DataFrame(columns=["n_estimators", "max_depth", "criterion", "accuracy"])
    for n_estimators in range(1, 21):
        for max_depth in range(1, 6):
            for criterion in ["entropy", "gini", "log_loss"]:
                model_rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, criterion=criterion)
                model_rf.fit(data_val, label_val)
                prediction = model_rf.predict(data_test)
                accuracy = get_accuracy(prediction, label_test)
                rf_accuracy = pd.concat([rf_accuracy if not rf_accuracy.empty else None,
                                         pd.DataFrame({"n_estimators": [n_estimators],
                                                       "max_depth": [max_depth],
                                                       "criterion": [criterion],
                                                       "accuracy": [accuracy]})], ignore_index=True)
    plt.title("rf_accuracy")
    plt.plot(rf_accuracy["accuracy"])
    plt.xlabel("model_num")
    plt.ylabel("%")
    plt.show(block=False)
    if plt.waitforbuttonpress():
        plt.close()
    hyp_par_rf = rf_accuracy.iloc[rf_accuracy["accuracy"].argmax()]
    print("max rf_accuracy: ", hyp_par_rf["accuracy"])

    # XGBoost
    xgb_accuracy = pd.DataFrame(columns=["n_estimators", "max_depth", "accuracy"])
    for n_estimators in range(1, 21):
        for max_depth in range(1, 6):
            model_xgb = XGBClassifier(n_estimators=n_estimators, max_depth=max_depth)
            model_xgb.fit(data_val, label_val)
            prediction = model_xgb.predict(data_test)
            accuracy = get_accuracy(prediction, label_test)
            xgb_accuracy = pd.concat([xgb_accuracy if not xgb_accuracy.empty else None,
                                      pd.DataFrame({"n_estimators": [n_estimators],
                                                    "max_depth": [max_depth],
                                                    "accuracy": [accuracy]})], ignore_index=True)
    plt.title("xgb_accuracy")
    plt.plot(xgb_accuracy["accuracy"])
    plt.xlabel("model_num")
    plt.ylabel("%")
    plt.show(block=False)
    if plt.waitforbuttonpress():
        plt.close()
    hyp_par_xgb = xgb_accuracy.iloc[xgb_accuracy["accuracy"].argmax()]
    print("max xgb_accuracy: ", hyp_par_xgb["accuracy"])

    # Logistic Regression
    lr_accuracy = pd.DataFrame(columns=["C", "solver", "accuracy"])
    for C in [0.0001, 0.001, 0.01, 0.1]:
        for solver in ["lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"]:
            model_lr = LogisticRegression(C=C, solver=solver)
            model_lr.fit(data_val, label_val)
            prediction = model_lr.predict(data_test)
            accuracy = get_accuracy(prediction, label_test)
            lr_accuracy = pd.concat([lr_accuracy if not lr_accuracy.empty else None,
                                     pd.DataFrame({"C": C,
                                                   "solver": [solver],
                                                   "accuracy": [accuracy]})], ignore_index=True)
    plt.title("lr_accuracy")
    plt.plot(lr_accuracy["accuracy"])
    plt.xlabel("model_num")
    plt.ylabel("%")
    plt.show(block=False)
    if plt.waitforbuttonpress():
        plt.close()
    hyp_par_lr = lr_accuracy.iloc[lr_accuracy["accuracy"].argmax()]
    print("max lr_accuracy: ", hyp_par_lr["accuracy"])

    # K Neighbors
    knn_accuracy = pd.DataFrame(columns=["n_neighbors", "accuracy"])
    for n_neighbors in range(1, 11):
        model_knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        model_knn.fit(data_val, label_val)
        prediction = model_knn.predict(data_test)
        accuracy = get_accuracy(prediction, label_test)
        knn_accuracy = pd.concat([knn_accuracy if not knn_accuracy.empty else None,
                                  pd.DataFrame({"n_neighbors": [n_neighbors],
                                                "accuracy": [accuracy]})], ignore_index=True)
    plt.title("knn_accuracy")
    plt.plot(knn_accuracy["accuracy"])
    plt.xlabel("model_num")
    plt.ylabel("%")
    plt.show(block=False)
    if plt.waitforbuttonpress():
        plt.close()
    hyp_par_knn = knn_accuracy.iloc[knn_accuracy["accuracy"].argmax()]
    print("max knn_accuracy: ", hyp_par_knn["accuracy"], '\n')

    model_rf = RandomForestClassifier(n_estimators=hyp_par_rf["n_estimators"], max_depth=hyp_par_rf["max_depth"],
                                      criterion=hyp_par_rf["criterion"])
    model_rf.fit(data_train, label_train)
    prediction = model_rf.predict(data_test)
    accuracy = get_accuracy(prediction, label_test)
    print("Result rf_accuracy: ", accuracy)

    model_xgb = XGBClassifier(n_estimators=int(hyp_par_xgb["n_estimators"]), max_depth=int(hyp_par_xgb["max_depth"]))
    model_xgb.fit(data_train, label_train)
    prediction = model_xgb.predict(data_test)
    accuracy = get_accuracy(prediction, label_test)
    print("Result xgb_accuracy: ", accuracy)

    model_lr = LogisticRegression(C=hyp_par_lr["C"], solver=hyp_par_lr["solver"])
    model_lr.fit(data_train, label_train)
    prediction = model_lr.predict(data_test)
    accuracy = get_accuracy(prediction, label_test)
    print("Result lr_accuracy: ", accuracy)

    model_knn = KNeighborsClassifier(n_neighbors=int(hyp_par_knn["n_neighbors"]))
    model_knn.fit(data_train, label_train)
    prediction = model_knn.predict(data_test)
    accuracy = get_accuracy(prediction, label_test)
    print("Result knn_accuracy: ", accuracy, '\n')
    return data, label, model_rf, model_xgb, model_lr, model_knn


def get_accuracy(prediction, fact):
    res = (prediction == fact).astype('int')
    right_num = sum(res)
    total_num = len(res)
    accuracy = right_num / total_num
    return accuracy


def task2(data, label, model_rf, model_xgb, model_lr, model_knn):
    importances = model_rf.feature_importances_
    features = data.columns
    indices = np.argsort(importances)
    plt.title('Важность признаков')
    plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    plt.yticks(range(len(indices)), features[indices])
    plt.xlabel('Относительная важность')
    plt.show(block=False)
    if plt.waitforbuttonpress():
        plt.close()

    two = features[indices][::-1][:2]
    four = features[indices][::-1][:4]
    eight = features[indices][::-1][:8]

    for count in [two, four, eight]:
        scaler = MinMaxScaler()
        data_scaled = scaler.fit_transform(data[count])
        data_train, data_test, label_train, label_test = train_test_split(data_scaled, label, test_size=0.2)

        model_rf.fit(data_train, label_train)
        prediction = model_rf.predict(data_test)
        accuracy = get_accuracy(prediction, label_test)
        print(f"Result rf_accuracy on {len(count)} "f"important features: ", accuracy)

        model_xgb.fit(data_train, label_train)
        prediction = model_xgb.predict(data_test)
        accuracy = get_accuracy(prediction, label_test)
        print(f"Result xgb_accuracy on {len(count)} important features: ", accuracy)

        model_lr.fit(data_train, label_train)
        prediction = model_lr.predict(data_test)
        accuracy = get_accuracy(prediction, label_test)
        print(f"Result lr_accuracy on {len(count)} important features: ", accuracy)

        model_knn.fit(data_train, label_train)
        prediction = model_knn.predict(data_test)
        accuracy = get_accuracy(prediction, label_test)
        print(f"Result knn_accuracy on {len(count)} important features: ", accuracy, '\n')


def main():
    data, label, model_rf, model_xgb, model_lr, model_knn = task1()
    task2(data, label, model_rf, model_xgb, model_lr, model_knn)


main()
