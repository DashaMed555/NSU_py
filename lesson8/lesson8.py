import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import seaborn as sns


def task1():

    table = pd.read_csv("wells_info_with_prod.csv", index_col=0)
    table = table[["CompletionDate", "StateName", "Prod1Year"]]

    date_vector = CountVectorizer(token_pattern="\d{4}-\d\d-\d\d")
    date_vector.fit_transform(table["CompletionDate"])

    category_vector = CountVectorizer(lowercase=False, token_pattern=".+")
    category_vector.fit_transform(table["StateName"])

    table["StateNameConverted"] = table["StateName"].replace(category_vector.vocabulary_)
    table["CompletionDateConverted"] = table["CompletionDate"].replace(date_vector.vocabulary_)

    return table


def task2(table):
    data = table.drop("Prod1Year", axis=1)
    target = table["Prod1Year"]
    data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=0.2, random_state=1)
    return data_train, data_test, target_train, target_test


def task3(train):
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train.drop(["StateName", "CompletionDate"], axis=1))
    return scaler, train_scaled


def task4(test, scaler):
    test_scaled = scaler.transform(test.drop(["StateName", "CompletionDate"], axis=1))
    return test_scaled


def main():
    table = task1()
    data_train, data_test, target_train, target_test = task2(table)
    train = data_train
    train["Prod1Year"] = target_train
    scaler, train_scaled = task3(train)
    test = data_test
    test["Prod1Year"] = target_test
    test_scaled = task4(test, scaler)
    new_df = pd.DataFrame(train_scaled)
    sns.pairplot(new_df)
    plt.show()


main()
