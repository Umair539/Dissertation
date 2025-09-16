import pandas as pd
import numpy as np
import keras
import matplotlib.pyplot as plt

features = ['speed','temperature','density','bz_gsm','bx_gsm','by_gsm','pressure','smoothed_ssn']#,'theta_gsm',]#bt

x = pd.read_csv('xtest.csv')
y = pd.read_csv('test/dst_labels.csv')
stats = pd.read_csv('stats.csv')
a,b = (stats.iloc[-1])
print(a,b)

M = stats['Median'][:-2].values
I = stats['Iqr'][:-2].values
print(M)
print(len(M))
print(len(x[features].columns))
x[features] = (x[features] - M)/I

#exit()
#a
x1 = x[x['period'] == 'test_a'][['timedelta'] + features]
x1.reset_index(inplace=True, drop=True)
x1['timedelta'] = pd.to_timedelta(x1['timedelta'])

x1 = x1.set_index('timedelta').resample('h').agg(['mean','std'])
x1.columns = ['_'.join(col) for col in x1]
x1.drop('smoothed_ssn_std',axis=1,inplace=True)
print(len(x1))

y1 = y[y['period'] == 'test_a']['dst']
y1 = (y1 - a)/b

#b
x2 = x[x['period'] == 'test_b'][['timedelta'] + features]
x2.reset_index(inplace=True, drop=True)
x2['timedelta'] = pd.to_timedelta(x2['timedelta'])

x2 = x2.set_index('timedelta').resample('h').agg(['mean','std'])
x2.columns = ['_'.join(col) for col in x2]
x2.drop('smoothed_ssn_std',axis=1,inplace=True)
print(len(x2))

y2 = y[y['period'] == 'test_b']['dst']
y2 = (y2 - a)/b

#c
x3 = x[x['period'] == 'test_c'][['timedelta'] + features]
x3.reset_index(inplace=True, drop=True)
x3['timedelta'] = pd.to_timedelta(x3['timedelta'])

x3 = x3.set_index('timedelta').resample('h').agg(['mean','std'])
x3.columns = ['_'.join(col) for col in x3]
x3.drop('smoothed_ssn_std',axis=1,inplace=True)
print(len(x3))

y3 = y[y['period'] == 'test_c']['dst']
y3 = (y3 - a)/b

length = len(x1.columns) #length for model.fit
win = 24*7

#validation split
#period a
xt1 = x1.iloc[:] #train
yt1 = y1.iloc[:]

#period b
xt2 = x2.iloc[:] #train
yt2 = y2.iloc[:]

#period c
xt3 = x3.iloc[:] #train
yt3 = y3.iloc[:]

#convert from datframe to array
xt1 = xt1.values
yt1 = yt1.values
xt2 = xt2.values
yt2 = yt2.values
xt3 = xt3.values
yt3 = yt3.values
results = []
for t in [1,5]:
    #sliding windows
    batch = 32
    test_1 = keras.utils.timeseries_dataset_from_array(
        data = xt1,
        targets = yt1[win+t:],
        sequence_length = win,
        sampling_rate = 1,
        sequence_stride = 1,
        batch_size = batch,
        )

    test_2 = keras.utils.timeseries_dataset_from_array(
        data = xt2,
        targets = yt2[win+t:],
        sequence_length = win,
        sampling_rate = 1,
        sequence_stride = 1,
        batch_size = batch,
        )

    test_3 = keras.utils.timeseries_dataset_from_array(
            data = xt3,
            targets = yt3[win+t:],
            sequence_length = win,
            sampling_rate = 1,
            sequence_stride = 1,
            batch_size = batch,
        )
    
    test = test_1.concatenate(test_2).concatenate(test_3)
    yv = np.concatenate([yt1[win+t:],yt2[win+t:],yt3[win+t:]]) * b + a

    model = keras.models.load_model('final_Linear_' + str(t) +'.keras')
    model.summary()

    predictions1 = model.predict(test).flatten()
    predictions1 = predictions1 * b + a

    rmse1 = np.sqrt(np.mean((yv - predictions1) ** 2))
    print(model.evaluate(test))
    print(f"RMSE: {rmse1:.2f}")

    # Compute correlation coefficient using numpy
    corr_matrix = np.corrcoef(yv, predictions1)
    corr_coef = corr_matrix[0, 1]
    print(corr_coef)
    # Plot
    fig, ax = plt.subplots(dpi=200)
    ax.scatter(yv, predictions1, s=2)
    ax.set_ylabel('Predicted Dst (nT)')
    ax.set_xlabel('Observed Dst (nT)')

    min_val = min(min(yv), min(predictions1))
    max_val = max(max(yv), max(predictions1))
    pad = (max_val - min_val) * 0.05
    ax.set_xlim(min_val - pad, max_val + pad)
    ax.set_ylim(min_val - pad, max_val + pad)
    ax.set_aspect('equal', adjustable='box')

    # Identity line
    ax.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='y = x')
    ax.set_title('Linear Model Predictions for t+'+str(t))
    ax.legend()
    #fig.savefig('linear')
    plt.show()
