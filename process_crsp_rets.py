# MSFT IBM AAPL INTC NVDA from 2000 to 2024
import csv
import math

import numpy as np

# transform csv to train 70 val 10 test 20 split and (number observations, window size, number sensors, observed value (dim 1))
skip_permnos = ('10145', '10779', '10874', '12052', '12490', '13901', '13987', '14795', '15318', '15720', '16678', '17073', '17778', '18075', '18438', '19166', '19271', '19828', '20415', '20512', '20598', '21020', '21178', '24053', '24766', '25081', '25304', '26201', '26650', '27617', '28302', '28804', '29102', '29209', '29744', '29938', '30509', '30796', '31480', '32678', '32791', '32803', '32870', '32942', '33452', '33770', '33823', '33849', '34673', '34948', '35263', '36281', '36898', '37381', '38033', '38156', '38746', '39538', '39731', '41188', '42358', '42439', '42614', '43123', '43334', '44230', '44943', '45225', '46068', '46340', '46463', '46834', '46923', '47002', '48072', '48347', '49138', '49322', '49330', '49429', '49488', '50243', '50876', '50948', '51530', '51925', '52250', '52396', '52425', '52898', '52936', '52978', '53110', '53225', '53604', '54114', '54199', '54244', '55678', '55862', '55984', '56822', '56856', '56945', '57007', '57913', '58318', '58334', '58413', '58421', '58771', '58836', '58975', '59045', '59176', '59256', '59328', '59408', '59459', '59504', '59600', '59627', '60038', '60097', '60468', '60687', '61487', '61496', '61508', '61567', '62308', '62341', '62367', '62498', '62958', '63132', '63546', '63765', '64742', '65665', '65700', '65947', '66384', '68145', '70578', '75039', '75064', '76655', '77973', '78749', '79039', '79573', '79864', '80857')
with open("data_unprocessed/stocks_19800101_20230101.csv", 'r') as csvfile:
    csvreader = csv.reader(csvfile, delimiter=",")
    csvreader.__next__()  # hop over header
    csvreader.__next__()  # hop over header
    curr_permno = -1
    np_rets = []
    rets = []
    error_count = 0
    error_permnos = dict()
    for date, permno_str, prc_str, ret_str in csvreader:
        #print(f'{int(permno_str)} {float(ret_str)}')
        if permno_str in skip_permnos:
            continue
        if int(permno_str) != curr_permno:
            if curr_permno != -1:
                # create the np array
                np_rets.append(np.array(rets))
                rets = []
            curr_permno = int(permno_str)
        rets.append(float(ret_str))
        if float(ret_str) < -1:  # we check for -66, -77, -99 errors
            error_count = error_count + 1
            # print("invalid " + ret_str + " " + date + " " + permno_str + " " + prc_str)
            if permno_str not in error_permnos:
                error_permnos[permno_str] = set()
                error_permnos[permno_str].add(ret_str)
            else:
                error_permnos[permno_str].add(ret_str)


    np_rets.append(np.array(rets))  # to catch the end of file
    np_rets = np.vstack(np_rets).transpose()
    print("error count")
    print(error_count)
    print(tuple(error_permnos.keys()))


    print(len(error_permnos))

    print(np_rets.shape)
    print(np.isnan(np.sum(np_rets)))
    np_rets = np.log(np_rets + 1)

    # now we split the data

    window = 12
    x = []
    y = []
    for i in range(np_rets.shape[0] - window - window):
        x.append(np_rets[i: i + window])
        y.append(np_rets[i+window: i+2*window])
    x = np.stack(x)
    x = np.expand_dims(x, axis=3)
    y = np.stack(y)
    y = np.expand_dims(y, axis=3)

    print(x.shape)

    print(y.shape)

    # now we have to sample for the train valid test data

    # for n=6273, we have train 4391, valid 627, test 1255 0.7, 0.1, 0.2
    n = x.shape[0]
    num_train = math.ceil(0.7 * n)
    num_val = math.floor(0.1 * n)
    num_test = math.floor(0.2 * n)
    indices = np.random.permutation(x.shape[0])
    training_idx, valid_idx, test_idx = indices[:num_train], indices[num_train: num_train+num_val], indices[num_train+num_val:]
    training_x, valid_x, test_x = x[training_idx, :], x[valid_idx, :], x[test_idx, :]
    training_y, valid_y, test_y = y[training_idx, :], y[valid_idx, :], y[test_idx, :]

    # now we save as .npz format

    np.savez("data/CRSP_large/train.npz", x=training_x, y=training_y)
    np.savez("data/CRSP_large/val.npz", x=valid_x, y=valid_y)
    np.savez("data/CRSP_large/test.npz", x=test_x, y=test_y)

