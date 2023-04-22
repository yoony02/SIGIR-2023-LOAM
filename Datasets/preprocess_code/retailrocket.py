'''
RetailRocket Preprocessing Code (without and with window)
'''
import argparse
import os
import operator
import pickle

import csv
import datetime
from collections import Counter
import numpy as np
import time


parser = argparse.ArgumentParser()
parser.add_argument('--save-dir', default='retailrocket', help='save directory name')
opt = parser.parse_args()
print(opt)


print("-- Starting @ %ss" % datetime.datetime.now())

if not os.path.exists(f'../{opt.save_dir}'):
    os.makedirs(f'../{opt.save_dir}')

with open("../raw/events.csv", "r") as f:
    lines = f.readlines()
    user_dict = {}
    items_freq = {}

    for line in lines[1:]:
        data = line.strip('\n').split(',')
        timestamp = int(int(data[0]) / 1000)
        user = data[1]
        item = data[3]
        new_line = [timestamp, user, item]

        if user in user_dict.keys():
            user_dict[user] += [new_line]
        else:
            user_dict[user] = [new_line]
        
        if item in items_freq:
            items_freq[item] += 1
        else:
            items_freq[item] = 1
    
if len(items_freq) > 40000:
    items_freq = dict(sorted(items_freq.items(), key=lambda d: d[1], reverse=True)[:40000])

rr_data = open(f"../{opt.save_dir}/retailrocket_data.csv", "w")
writer = csv.writer(rr_data)
writer.writerow(["SessionId","TimeStamp","ItemId"])
sessid = 0
for user in user_dict.keys():
    user_dict[user] = sorted(user_dict[user], key=lambda x: x[1])
    curdate = None
    
    for record in user_dict[user]:
        item_id = record[2]
        # if len(item_id) >= 2:
        if len(item_id) >= 2 and item_id in items_freq:
            sess_date = record[0]
            if curdate and sess_date-curdate > 28800:
                sessid += 1
            curdate = sess_date
            new_record = [str(sessid), str(record[0]), record[2]]
            writer.writerow(new_record)  
    sessid += 1  

print("-- Allocating Session ids @ %ss" % datetime.datetime.now())

with open(f'../{opt.save_dir}/retailrocket_data.csv', "r") as f:
    reader = csv.DictReader(f, delimiter=',')
    sess_clicks = {}
    sess_date = {}
    ctr = 0
    curid = -1
    curdate = None
    for data in reader:
        sessid = int(data['SessionId'])
        if curdate and curid != sessid:
            date = ''
            date = curdate
            sess_date[curid] = date
        curid = sessid
        item = int(data['ItemId']), float(data['TimeStamp'])
        curdate = ''
        curdate = float(data['TimeStamp'])

        if sessid in sess_clicks:
            sess_clicks[sessid] += [item]
        else:
            sess_clicks[sessid] = [item]
        ctr += 1
    date = curdate

    for i in list(sess_clicks):
        sorted_clicks = sorted(sess_clicks[i], key=operator.itemgetter(1))
        sess_clicks[i] = [c[0] for c in sorted_clicks]
    sess_date[curid] = date
print("-- Reading data @ %ss" % datetime.datetime.now())


# Filter out length 1 sessions
for s in list(sess_clicks):
    if len(sess_clicks[s]) < 2:
        del sess_clicks[s]
        del sess_date[s]

# Count number of times each item appears
iid_counts = {}
for s in sess_clicks:
    seq = sess_clicks[s]
    for iid in seq:
        if iid in iid_counts:
            iid_counts[iid] += 1
        else:
            iid_counts[iid] = 1

sorted_counts = sorted(iid_counts.items(), key=operator.itemgetter(1))

length = len(sess_clicks)
for s in list(sess_clicks):
    curseq = sess_clicks[s]
    filseq = list(filter(lambda i: iid_counts[i] >= 5, curseq))
    
    if len(filseq) < 2:
        del sess_clicks[s]
        del sess_date[s]
    else:
        sess_clicks[s] = filseq

import pdb
pdb.set_trace()

# Split out test set based on dates
dates = list(sess_date.items())
maxdate = dates[0][1]

for _, date in dates:
    if maxdate < date:
        maxdate = date

# the last of 100 seconds for test
splitdate = maxdate - 86400 * 7

print('Splitting date', splitdate)      # Yoochoose: ('Split date', 1411930799.0)
tra_sess = filter(lambda x: x[1] < splitdate, dates)
tes_sess = filter(lambda x: x[1] > splitdate, dates)

# Sort sessions by date
tra_sess = sorted(tra_sess, key=operator.itemgetter(1))     # [(session_id, timestamp), (), ]
tes_sess = sorted(tes_sess, key=operator.itemgetter(1))     # [(session_id, timestamp), (), ]
print(f'# train sessions: {len(tra_sess)}')    # 186670    # 7966257
print(f'# test sessions: {len(tes_sess)}')    # 15979     # 15324
print(f'train sessions example: {tra_sess[:3]}')
print(f'test sessions example: {tes_sess[:3]}')
    
print("-- Splitting train set and test set @ %ss" % datetime.datetime.now())

# Choosing item count >=5 gives approximately the same number of items as reported in paper
item_dict = {}
# Convert training sessions to sequences and renumber items to start from 1
def obtian_tra():
    train_ids = []
    train_seqs = []
    train_dates = []
    item_cnt = 1
    for s, date in tra_sess:
        seq = sess_clicks[s]
        outseq = []
        for i in seq:
            if i in item_dict:
                outseq += [item_dict[i]]
            else:
                outseq += [item_cnt]
                item_dict[i] = item_cnt
                item_cnt += 1
        if len(outseq) < 2:  # Doesn't occur
            continue
        train_ids += [s]
        train_dates += [date]
        train_seqs += [outseq]

    print(f"item count: {item_cnt}")
    pickle.dump(item_cnt, open(f'../{opt.save_dir}/n_node.txt', 'wb'))
    return train_ids, train_dates, train_seqs

# Convert test sessions to sequences, ignoring items that do not appear in training set
def obtian_tes():
    test_ids = []
    test_seqs = []
    test_dates = []
    for s, date in tes_sess:
        seq = sess_clicks[s]
        outseq = []
        for i in seq:
            if i in item_dict:
                outseq += [item_dict[i]]
        if len(outseq) < 2:
            continue
        test_ids += [s]
        test_dates += [date]
        test_seqs += [outseq]
    return test_ids, test_dates, test_seqs

tra_ids, tra_dates, tra_seqs = obtian_tra()
tes_ids, tes_dates, tes_seqs = obtian_tes()

def process_seqs(iseqs, idates):
    out_seqs = []
    out_dates = []
    labs = []
    ids = []
    for id, seq, date in zip(range(len(iseqs)), iseqs, idates):
        for i in range(1, len(seq)):
            tar = seq[-i]
            labs += [tar]
            out_seqs += [seq[:-i]]
            out_dates += [date]
            ids += [id]
    return out_seqs, out_dates, labs, ids

tr_seqs, tr_dates, tr_labs, tr_ids = process_seqs(tra_seqs, tra_dates)
te_seqs, te_dates, te_labs, te_ids = process_seqs(tes_seqs, tes_dates)
tra = (tr_seqs, tr_labs)
tes = (te_seqs, te_labs)

print(f"final # train sessions: {len(tr_seqs)}")
print(f"final # test sessions: {len(te_seqs)}")
print(f"final examples train sessions: {tr_seqs[:3], tr_dates[:3], tr_labs[:3]}")
print(f"final exmaples test sessions: {te_seqs[:3], te_dates[:3], te_labs[:3]}")

all = 0
for seq in tra_seqs:
    all += len(seq)
for seq in tes_seqs:
    all += len(seq)
print('sessions average length: ', all * 1.0/(len(tra_seqs) + len(tes_seqs)))


pickle.dump(tra, open(f'../{opt.save_dir}/train.txt', 'wb'))
pickle.dump(tes, open(f'../{opt.save_dir}/test.txt', 'wb'))
pickle.dump(tra_seqs, open(f'../{opt.save_dir}/all_train_seq.txt', 'wb'))

# head tail split 
lab_cnt_dict = Counter(tr_labs)

lab_dict_sorted = sorted(lab_cnt_dict.items(), reverse=True, key=lambda item: item[1])
lab_dict_keys = [item[0] for item in lab_dict_sorted]
lab_dict_values = [item[1] for item in lab_dict_sorted]

split_point = int(np.floor(len(lab_dict_keys) * 0.2))
point_cnt_value = lab_dict_values[split_point]
split_idx = [i for i, cnt in enumerate(lab_dict_values) if cnt == (point_cnt_value-1)][0]

ht_dict = dict()
ht_dict['head'] = lab_dict_keys[:split_idx]
ht_dict['tail'] = lab_dict_keys[split_idx:]

print(f'# head items : {len(ht_dict["head"])}, # tail items : {len(ht_dict["tail"])}')
pickle.dump(ht_dict, open(f'../{opt.save_dir}/ht_dict.pickle', 'wb'))
