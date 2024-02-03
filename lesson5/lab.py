import numpy as np
import pandas as pd


def main():
    sessions = pd.read_csv("cinema_sessions.csv", sep=' ', index_col=0)
    labels = pd.read_csv("titanic_with_labels.csv", sep=' ', index_col=0)
    step_one(labels)
    step_two(labels)
    step_three(labels)
    step_four(labels)
    step_five(labels)
    step_six(labels, sessions)
    labels.to_csv('titanic_res.csv')


def step_one(labels):
    labels['sex'].replace(['-', 'Не указан', 'не указан'], np.NaN, inplace=True)
    labels['sex'].replace(['Ж', 'ж'], 0, inplace=True)
    labels['sex'].replace(['M', 'м', 'Мужчина'], 1, inplace=True)


def step_two(labels):
    max_row = labels['row_number'].max()
    labels['row_number'].replace(np.NaN, max_row, inplace=True)


def step_three(labels):
    labels['liters_drunk'].where((0 <= labels['liters_drunk']) & (labels['liters_drunk'] <= 5), inplace=True)
    labels['liters_drunk'].replace(np.NaN, labels['liters_drunk'].mean(), inplace=True)


def step_four(labels):
    age = labels['age']
    children = age[age < 18]
    adults = age[(18 <= age) & (age <= 50)]
    elderly = age[age > 50]
    labels.drop('age', axis=1, inplace=True)
    labels['age_children'] = children
    labels['age_adults'] = adults
    labels['age_elderly'] = elderly


def step_five(labels):
    labels['drink'].replace(['Cola', 'Fanta', 'Water'], 0, inplace=True)
    labels['drink'].replace(['Beerbeer', 'Bugbeer', 'Strong beer', 'Наше пиво'], 1, inplace=True)


def step_six(labels, sessions):
    merged = pd.merge(labels, sessions, on='check_number')
    start = merged['session_start']
    f = '%H:%M:%S.%f'
    start = pd.to_datetime(start, format=f)
    morning = start < pd.to_datetime('12:00:00.000', format=f)
    day = (pd.to_datetime('12:00:00.000', format=f) <= start) & (start <= pd.to_datetime('18:00:00.000', format=f))
    evening = start > pd.to_datetime('18:00:00.000', format=f)
    labels['morning'] = morning
    labels['day'] = day
    labels['evening'] = evening


main()
