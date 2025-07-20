import pandas as pd

features = ['speed','temperature','density','bz_gsm','bx_gsm','by_gsm','pressure','smoothed_ssn','theta_gsm']

#load
x = pd.read_csv('train/solar_wind.csv') #solar wind
s = pd.read_csv('train/sunspots.csv')   #sunspots
y = pd.read_csv('train/dst_labels.csv') #dst

#convert timedelta
x['timedelta'] = pd.to_timedelta(x['timedelta'])
s['timedelta'] = pd.to_timedelta(s['timedelta'])

s.sort_values(['period', 'timedelta'], inplace=True)

#pressure column
x['pressure'] = x['density'] * x['speed']**2

#merge
x['days'] = x['timedelta'].dt.days
s['days'] = s['timedelta'].dt.days
x = pd.merge(x, s[['period', 'days', 'smoothed_ssn']], 'left', ['period', 'days'])
x.drop(columns='days', inplace=True)

#x = pd.merge(x, y, 'left', ['period', 'timedelta'])
x.sort_values(['period', 'timedelta'], inplace=True)
x.reset_index(inplace=True)

#drop unimportant columns
x = x[['period','timedelta'] + features]

#fill blanks
train_short = [c for c in features if c != "smoothed_ssn"]
for p in ["train_a", "train_b", "train_c"]:
    curr_period = x["period"] == p

    x.loc[curr_period, "smoothed_ssn"] = (
        x.loc[curr_period, "smoothed_ssn"].ffill().bfill()
    )
    roll = (
        x[train_short].rolling(window=20, min_periods=5).mean().interpolate("linear", axis=0)
    )
    x.loc[curr_period, train_short] = x.loc[curr_period, train_short].fillna(roll)
    x.loc[curr_period, train_short] = (
        x.loc[curr_period, train_short].ffill().bfill()
    )
   
#normalise
median = x[features].median()
uq = x[features].quantile(0.75)
lq = x[features].quantile(0.25)
iqr = uq - lq
x[features] = (x[features] - median)/iqr

stats = pd.DataFrame({'Median': median, 'Iqr':iqr})

median = y['dst'].median()
uq = y['dst'].quantile(0.75)
lq = y['dst'].quantile(0.25)
iqr = uq - lq
y = (y['dst'] - median)/iqr

stats.loc[len(stats)] = {'Median':median, 'Iqr':iqr}
stats.to_csv('stats.csv', index=False)

x.to_csv('x.csv', index=False)
y.to_csv('y.csv', index=False)