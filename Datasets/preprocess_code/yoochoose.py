import argparse
import os
import operator
import pickle
import time

import csv
import datetime
import numpy as np
from collections import Counter


parser = argparse.ArgumentParser()
parser.add_argument('--save-dir', default='yoochoose', help='save directory name')
opt = parser.parse_args()
print(opt)

os.makedirs(f'../{opt.save_dir}_4', exist_ok=True)
os.makedirs(f'../{opt.save_dir}_64', exist_ok=True)

# with open('../raw/yoochoose-clicks.dat', "r") as f, open('../raw/yoochoose-clicks-withHeader.dat', 'w') as fn:
#     fn.write('sessionId,timestamp,itemId,category'+'\n')
#     for line in f:
#         fn.write(line)

dataset = '../raw/yoochoose-clicks.dat'


print("-- Starting @ %ss" % datetime.datetime.now())

with open(dataset, "r") as f:
    reader = csv.reader(f, delimiter=',')
    sess_clicks = {}
    sess_date = {}
    ctr = 0
    curid = -1
    curdate = None
    for data in reader:
        sessid = data[0]
        if curdate and not curid == sessid:
            date = ''
            date = time.mktime(time.strptime(curdate[:19], '%Y-%m-%dT%H:%M:%S'))
            sess_date[curid] = date
        curid = sessid
        item = data[2]
        curdate = ''
        curdate = data[1]

        if sessid in sess_clicks:
            sess_clicks[sessid] += [item]
        else:
            sess_clicks[sessid] = [item]
        ctr += 1
    date = time.mktime(time.strptime(curdate[:19], '%Y-%m-%dT%H:%M:%S'))
    # add
    for i in list(sess_clicks):
        sorted_clicks = sorted(sess_clicks[i], key=operator.itemgetter(1))
        sess_clicks[i] = [c for c in sorted_clicks]
    sess_date[curid] = date
print("-- Reading data @ %ss" % datetime.datetime.now())

# Filter out length shorter than 2
for s in list(sess_clicks):
    if len(sess_clicks[s]) < 2 :
        del sess_clicks[s]
        del sess_date[s]

# Counter number of times each appears
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

    if len(filseq) < 2 :
        del sess_clicks[s]
        del sess_date[s]
    else:
        sess_clicks[s] = filseq

dates = list(sess_date.items())
maxdate = dates[0][1]

for _, date in dates:
    if maxdate < date:
        maxdate = date

# 7 days for test
splitdate = 0
splitdate = maxdate - 86400 * 1

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


item_dict = {}
def obtain_tra():
    # Convert training sessions to sequences and renumber items to start from 1
    train_ids = []
    train_seqs = []
    train_dates = []
    item_ctr = 1
    for s, date in tra_sess:
        seq = sess_clicks[s]
        outseq = []
        for i in seq:
            if i in item_dict:
                outseq += [item_dict[i]]
            else:
                outseq += [item_ctr]
                item_dict[i] = item_ctr
                item_ctr += 1
        if len(outseq) < 2: 
            continue
        train_ids += [s]
        train_dates += [date]
        train_seqs += [outseq]
    print(f"all item count: {item_ctr}") 
    pickle.dump(item_ctr, open(f'../yoochoose_4/n_node.txt', 'wb'))
    pickle.dump(item_ctr, open(f'../yoochoose_64/n_node.txt', 'wb'))
    return train_ids, train_dates, train_seqs


# Convert test sessions to sequences, ignoring items that do not appear in training set
def obtain_tes():
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

tra_ids, tra_dates, tra_seqs = obtain_tra()
tes_ids, tes_dates, tes_seqs = obtain_tes()


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

import pdb
pdb.set_trace()

pickle.dump(tes, open(f'../{opt.save_dir}_4/test.txt', 'wb'))
pickle.dump(tes, open(f'../{opt.save_dir}_64/test.txt', 'wb'))

split4, split64 = int(len(tr_seqs) / 4), int(len(tr_seqs) / 64)
print(len(tr_seqs[-split4:]))
print(len(tr_seqs[-split64:]))

tra4, tra64 = (tr_seqs[-split4:], tr_labs[-split4:]), (tr_seqs[-split64:], tr_labs[-split64:])
seq4, seq64 = tra_seqs[tr_ids[-split4]:], tra_seqs[tr_ids[-split64]:]

pickle.dump(tra4, open(f'../{opt.save_dir}_4/train.txt', 'wb'))
pickle.dump(seq4, open(f'../{opt.save_dir}_4/all_train_seq.txt', 'wb'))

pickle.dump(tra64, open(f'../{opt.save_dir}_64/train.txt', 'wb'))
pickle.dump(seq64, open(f'../{opt.save_dir}_64/all_train_seq.txt', 'wb'))




seq4_all, seq64_all = 0, 0
for seq in seq4:
    seq4_all += len(seq)
for seq in tes_seqs:
    seq4_all += len(seq)
print('yoochoose 1/4 avg. length: ', seq4_all * 1.0/(len(seq4) + len(tes_seqs)))

for seq in seq64:
    seq64_all += len(seq)
for seq in tes_seqs:
    seq64_all += len(seq)
print('yoochoose 1/64 avg. length: ', seq64_all * 1.0/(len(seq64) + len(tes_seqs)))


# # item count 
# item_dict4, item_dict64 = [], []
# for i in seq4:
#     if i in item_dict4:
#         pass
#     else:
#         item_dict4.append(i)
# print("yoochoose 1/4 item cnt", len(item_dict4))

# for i in seq64:
#     if i in item_dict64:
#         pass
#     else:
#         item_dict64.append(i)
# print("yoochoose 1/64 item cnt", len(item_dict64))





# head tail split 
lab_cnt_dict4 = Counter(tr_labs[-split4:])

lab_dict_sorted4 = sorted(lab_cnt_dict4.items(), reverse=True, key=lambda item: item[1])
lab_dict_keys4 = [item[0] for item in lab_dict_sorted4]
lab_dict_values4 = [item[1] for item in lab_dict_sorted4]

split_point4 = int(np.floor(len(lab_dict_keys4) * 0.2))
point_cnt_value4 = lab_dict_values4[split_point4]
split_idx4 = [i for i, cnt in enumerate(lab_dict_values4) if cnt == (point_cnt_value4-1)][0]

ht_dict4 = dict()
ht_dict4['head'] = lab_dict_keys4[:split_idx4]
ht_dict4['tail'] = lab_dict_keys4[split_idx4:]

print(f'# head items : {len(ht_dict4["head"])}, # tail items : {len(ht_dict4["tail"])}')
pickle.dump(ht_dict4, open(f'../yoochoose_4/ht_dict.pickle', 'wb'))


################## split 64
lab_cnt_dict64 = Counter(tr_labs[-split64:])

lab_dict_sorted64 = sorted(lab_cnt_dict64.items(), reverse=True, key=lambda item: item[1])
lab_dict_keys64 = [item[0] for item in lab_dict_sorted64]
lab_dict_values64 = [item[1] for item in lab_dict_sorted64]

split_point64 = int(np.floor(len(lab_dict_keys64) * 0.2))
point_cnt_value64 = lab_dict_values64[split_point64]
split_idx64 = [i for i, cnt in enumerate(lab_dict_values64) if cnt == (point_cnt_value64-1)][0]

ht_dict64 = dict()
ht_dict64['head'] = lab_dict_keys64[:split_idx64]
ht_dict64['tail'] = lab_dict_keys64[split_idx64:]

print(f'# head items : {len(ht_dict64["head"])}, # tail items : {len(ht_dict64["tail"])}')
pickle.dump(ht_dict64, open(f'../yoochoose_64/ht_dict.pickle', 'wb'))