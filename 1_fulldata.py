import pandas as pd
import numpy as np
import keras
from keras import layers
from keras import Model
import matplotlib.pyplot as plt

features = ['speed','temperature','density','bz_gsm','bx_gsm','by_gsm','pressure','smoothed_ssn']#,'theta_gsm',]#bt
#features = ['speed','temperature','density','bz_gsm','pressure','smoothed_ssn']
x = pd.read_csv('x.csv')
y = pd.read_csv('train/dst_labels.csv')
stats = pd.read_csv('stats.csv')
a,b = (stats.iloc[-1])
print(a,b)
#a
x1 = x[x['period'] == 'train_a'][['timedelta'] + features]
x1.reset_index(inplace=True, drop=True)
x1['timedelta'] = pd.to_timedelta(x1['timedelta'])
x1 = x1.set_index('timedelta').resample('h').agg(['mean','std'])
x1.columns = ['_'.join(col) for col in x1]
x1.drop('smoothed_ssn_std',axis=1,inplace=True)

y1 = y[y['period'] == 'train_a']['dst']
y1 = (y1 - a)/b

#b
x2 = x[x['period'] == 'train_b'][['timedelta'] + features]
x2.reset_index(inplace=True, drop=True)
x2['timedelta'] = pd.to_timedelta(x2['timedelta'])
x2 = x2.set_index('timedelta').resample('h').agg(['mean','std'])
x2.columns = ['_'.join(col) for col in x2]
x2.drop('smoothed_ssn_std',axis=1,inplace=True)

y2 = y[y['period'] == 'train_b']['dst']
y2 = (y2 - a)/b

#c
x3 = x[x['period'] == 'train_c'][['timedelta'] + features]
x3.reset_index(inplace=True, drop=True)
x3['timedelta'] = pd.to_timedelta(x3['timedelta'])
x3 = x3.set_index('timedelta').resample('h').agg(['mean','std'])
x3.columns = ['_'.join(col) for col in x3]
x3.drop('smoothed_ssn_std',axis=1,inplace=True)

y3 = y[y['period'] == 'train_c']['dst']
y3 = (y3 - a)/b

length = len(x1.columns) #length for model.fit
buffer = len(x)
win = 24*7

#validation split
#period a
v_len = int(round( len(x1) * 0.2 )) #val size is 20%
ind1 = len(x1)-v_len #index sperating validation

xt1 = x1.iloc[:ind1] #train
yt1 = y1.iloc[:ind1]
xv1 = x1.iloc[ind1:] #validate
yv1 = y1.iloc[ind1:]

#period b
v_len = int(round( len(x2) * 0.2 )) #val size is 20%
ind1 = len(x2)-v_len #index sperating validation

xt2 = x2.iloc[:ind1] #train
yt2 = y2.iloc[:ind1]
xv2 = x2.iloc[ind1:] #validate
yv2 = y2.iloc[ind1:]

#period c
v_len = int(round( len(x3) * 0.2 )) #val size is 20%
ind1 = len(x3)-v_len #index sperating validation

xt3 = x3.iloc[:ind1] #train
yt3 = y3.iloc[:ind1]
xv3 = x3.iloc[ind1:] #validate
yv3 = y3.iloc[ind1:]

#convert from datframe to array
xt1 = xt1.values
yt1 = yt1.values
xt2 = xt2.values
yt2 = yt2.values
xt3 = xt3.values
yt3 = yt3.values

xv1 = xv1.values
yv1 = yv1.values
xv2 = xv2.values
yv2 = yv2.values
xv3 = xv3.values
yv3 = yv3.values
for t in [1,5]:
    
    #sliding windows
    batch = 32
    train_1 = keras.utils.timeseries_dataset_from_array(
        data = xt1,
        targets = yt1[win+t:],
        sequence_length = win,
        sampling_rate = 1,
        sequence_stride = 1,
        batch_size = batch,
        )

    train_2 = keras.utils.timeseries_dataset_from_array(
        data = xt2,
        targets = yt2[win+t:],
        sequence_length = win,
        sampling_rate = 1,
        sequence_stride = 1,
        batch_size = batch,
        )
    
    train_3 = keras.utils.timeseries_dataset_from_array(
            data = xt3,
            targets = yt3[win+t:],
            sequence_length = win,
            sampling_rate = 1,
            sequence_stride = 1,
            batch_size = batch,
        )

    train_1.batch(batch, drop_remainder = True)
    train_2.batch(batch, drop_remainder = True)
    train_3.batch(batch, drop_remainder = True)

    val_1 = keras.utils.timeseries_dataset_from_array(
        data = xv1,
        targets = yv1[win+t:],
        sequence_length = win,
        sampling_rate = 1,
        sequence_stride = 1,
        batch_size = batch,
        )

    val_2 = keras.utils.timeseries_dataset_from_array(
        data = xv2,
        targets = yv2[win+t:],
        sequence_length = win,
        sampling_rate = 1,
        sequence_stride = 1,
        batch_size = batch,
        )
    
    val_3 = keras.utils.timeseries_dataset_from_array(
            data = xv3,
            targets = yv3[win+t:],
            sequence_length = win,
            sampling_rate = 1,
            sequence_stride = 1,
            batch_size = batch,
        )

    val_1.batch(batch, drop_remainder = True)
    val_2.batch(batch, drop_remainder = True)
    val_3.batch(batch, drop_remainder = True)

    train_data = train_1.concatenate(train_2).concatenate(train_3)
    train_data = train_data.shuffle(buffer_size = buffer)
    val_data = val_1.concatenate(val_2).concatenate(val_3)

    yv = np.concatenate([yv1[win+t:],yv2[win+t:],yv3[win+t:]]) * b + a
    
    # Load and evaluate model
    #model
    i = 1
    if i == 1:
        inputs = layers.Input(shape=(win, length))
        x = layers.Reshape((win * length,))(inputs)
        outputs = layers.Dense(1, activation='linear')(x)

        model = Model(inputs=inputs, outputs=outputs)
        model.summary()

        model.compile(
            optimizer = keras.optimizers.Adam(learning_rate=0.0001),
            #loss = keras.losses.Huber(5),  #6,7,8,9
            loss = 'mse',
            metrics = [keras.metrics.RootMeanSquaredError()]
        )

        #run model
        early_stop = keras.callbacks.EarlyStopping(
            monitor = "val_loss",
            patience = 3,
            restore_best_weights = True
        )

        history = model.fit(
        train_data,
        epochs = 40,
        validation_data = val_data,
        callbacks = [early_stop],
        verbose = 1
        )

    elif i==2:
        F = 20
        U = 30
        K2 = 7
        K3 = 10
        inputs = layers.Input(shape=(win, length))

        x2 = layers.Conv1D(F, kernel_size=K2, strides=2, activation='relu')(inputs)
        x2 = layers.Conv1D(F, kernel_size=K2, strides=2, activation='relu')(x2)
        x2 = layers.Conv1D(2*F, kernel_size=2*K2, strides=4, activation='relu')(x2)
        x2 = layers.GlobalAveragePooling1D()(x2)

        x3 = layers.Conv1D(F, kernel_size=K3, strides=2, activation='relu')(inputs)
        x3 = layers.Conv1D(F, kernel_size=K3, strides=2, activation='relu')(x3)
        x3 = layers.Conv1D(2*F, kernel_size=2*K3, strides=4, activation='relu')(x3)
        x3 = layers.GlobalAveragePooling1D()(x3)

        x = layers.Concatenate()([x2, x3])

        x = layers.Dense(U, activation='relu')(x)
        outputs = layers.Dense(1, activation='linear')(x)

        model = Model(inputs=inputs, outputs=outputs)

        model.summary()

        model.compile(
            optimizer = keras.optimizers.Adam(learning_rate=0.0007),
            #loss = keras.losses.Huber(5),  #6,7,8,9
            loss = 'mse',
            metrics = [keras.metrics.RootMeanSquaredError()]
        )

        #run model
        early_stop = keras.callbacks.EarlyStopping(
            monitor = "val_loss",
            patience = 3,
            restore_best_weights = True
        )

        history = model.fit(
        train_data,
        epochs = 40,
        validation_data = val_data,
        callbacks = [early_stop],
        verbose = 1
        )

    predictions1 = model.predict(val_data).flatten()
    predictions1 = predictions1 * b + a

    rmse1 = np.sqrt(np.mean((yv - predictions1) ** 2))
    print(model.evaluate(val_data))
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
    ax.set_title('Linear Model Performance on Validation Data')
    ax.legend()
    #fig.savefig('linear')
    model.save('final_Linear_' + str(t) + '.keras')
    plt.show()