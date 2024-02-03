import pandas as pd
import numpy as np


def task1():
    print("Task 1\n")
    data = np.random.random((10, 5))
    print("Data:\n", data)
    df = pd.DataFrame(data)
    print("\nData Frame:\n\n", df)
    df_0_3 = df[df > 0.3]
    df_avg = df_0_3.mean(axis=1)
    print("\nAverage values:\n", df_avg, '\n')


def task2():
    print("Task 2\n")
    df = pd.read_csv("wells_info.csv", index_col=0)[["FirstProductionDate", "CompletionDate"]]
    prod = pd.to_datetime(df["FirstProductionDate"])
    comp = pd.to_datetime(df["CompletionDate"])
    df["NumOfDays"] = comp - prod
    df["NumOfMonths"] = df["NumOfDays"].dt.days // 30
    print(df, '\n')


def task3():
    print("Task 3\n")
    df = pd.read_csv("wells_info_na.csv")
    df_num = df.select_dtypes(["int64", "float64"])
    df_obj = df.select_dtypes(["object"])
    df.fillna(df_num.median(), inplace=True)
    df.fillna(df_obj.mode()[:1].squeeze(), inplace=True)
    df.to_csv("task3_res.csv")
    print(df, '\n')


def main():
    task1()
    task2()
    task3()


main()
