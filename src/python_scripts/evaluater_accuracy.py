from pandas import read_excel
import os

def evaluate_accuracy(path):
    df = read_excel(path)
    cnt = true_predict = 0
    for idx in df.index:
        cnt += 1
        if df['Оценка ТМ'][idx] == df['Оценка_прог'][idx]:
            true_predict += 1
    return true_predict / cnt

