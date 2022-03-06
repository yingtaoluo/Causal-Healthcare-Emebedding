import numpy as np
import csv
import torch
import pdb

group_code, group_icds = [], []
with open('./data/prref.csv', "r", encoding="utf-8") as f:
    reader = csv.reader(f)
    rows = [row for row in reader]
    sheet = np.array([x for x in np.array(rows[3:])])[:, :2]

for item in sheet:
    if eval(item[1]).strip(' ') not in group_code:
        group_code.append(eval(item[1]).strip(' '))
        group_icds.append([])
        group_icds[-1].append(eval(item[0]).strip(' '))
    else:
        group_icds[-1].append(eval(item[0]).strip(' '))
    if eval(item[0]).strip(' ') == '68':
        pdb.set_trace()

pdb.set_trace()


