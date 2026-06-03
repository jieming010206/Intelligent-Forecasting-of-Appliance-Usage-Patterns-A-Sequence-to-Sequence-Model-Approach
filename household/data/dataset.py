import pandas as pd
import numpy as np

def get_edit_distance(s1, s2):
    d = [[x for x in range(len(s1) + 1)] for _ in range(len(s2) + 1)]
    for y in range(1, len(s2) + 1):
        d[y][0] = y  # Corrected this line
    for x in range(1, len(s1) + 1):
        for y in range(1, len(s2) + 1):
            if s1[x - 1] == s2[y - 1]:
                d[y][x] = d[y - 1][x - 1]
            else:
                substute = d[y - 1][x - 1] + 1
                add = d[y][x - 1] + 1
                delete = d[y - 1][x] + 1
                d[y][x] = min(add, substute, delete)
    return d[-1][-1]

def get_lcs(s1, s2):
    m, n = len(s1), len(s2)
    lcs_len = [[0] * (n + 1) for _ in range(2)]
    current, previous = 0, 1
    
    for i in range(1, m + 1):
        current, previous = previous, current
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                lcs_len[current][j] = lcs_len[previous][j - 1] + 1
            else:
                lcs_len[current][j] = max(lcs_len[previous][j], lcs_len[current][j - 1])
    return lcs_len[current][n]



for id in range(8):
    data = pd.read_csv(f"data/house_ids_{id}.csv")
    years = data["year"].unique()
    days = []
    for year in years:
        days.append(np.unique(data[data["year"] == year]["day"]))
    max_edit = []
    timestamp = []
    appliance = []
    
    for year in range(2):
        for i in range(len(days[year])):
            for j in range(i + 1, min(i + 3, len(days[year]))):
                data_i = data[(data["year"] == years[year]) & (data["day"] == days[year][i])]
                data_j = data[(data["year"] == years[year]) & (data["day"] == days[year][j])]
                max_edit.append(max(len(data_i), len(data_j)))
                timestamp.append(get_edit_distance(np.array(data_i.iloc[:,-2]), np.array(data_j.iloc[:,-2])))
                appliance.append(get_edit_distance(np.array(data_i.iloc[:,-1]), np.array(data_j.iloc[:,-1])))

        
    print("====================================")
    print(f"household_{id}")
    print(f"length: {len(data)}")
    # subset_length = len(original[original["house_id"] == houses[id]]) // 144
    # print(f"length: {subset_length}")
    '''
    print("example sequnece")
    li = []
    for i in range(len(days[0])):
        li.append(np.array(data[data["day"] == days[i]]))
        print(f"day {i}")
        print(f"time: {li[i][:,-2]}")
        print(f"Appliance: {li[i][:,-1]}")
    '''

    print(f"max edit distance:  mean: {round(np.mean(max_edit))}, mode: {np.argmax(np.bincount(max_edit))}")
    print(f"timestamp   mean:{round(np.mean(timestamp))}, mode: {np.argmax(np.bincount(timestamp))}")
    print(f"appliance:  mean:{round(np.mean(appliance))}, mode: {np.argmax(np.bincount(appliance))}")
    print("\n")
