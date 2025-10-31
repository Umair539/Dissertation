import pandas as pd

features = [
    "speed",
    "temperature",
    "density",
    "bz_gsm",
    "bx_gsm",
    "by_gsm",
    "pressure",
    "smoothed_ssn",
]

# load
x = pd.read_csv("test/solar_wind.csv")  # solar wind
s = pd.read_csv("test/sunspots.csv")  # sunspots

# convert timedelta
x["timedelta"] = pd.to_timedelta(x["timedelta"])
s["timedelta"] = pd.to_timedelta(s["timedelta"])

s.sort_values(["period", "timedelta"], inplace=True)

# pressure column
x["pressure"] = x["density"] * x["speed"] ** 2

# merge
x["days"] = x["timedelta"].dt.days
s["days"] = s["timedelta"].dt.days
x = pd.merge(x, s[["period", "days", "smoothed_ssn"]], "left", ["period", "days"])
x.drop(columns="days", inplace=True)

x.sort_values(["period", "timedelta"], inplace=True)
x.reset_index(inplace=True)

# drop unimportant columns
x = x[["period", "timedelta"] + features]

# fill blanks
train_short = [c for c in features if c != "smoothed_ssn"]
for p in ["test_a", "test_b", "test_c"]:
    curr_period = x["period"] == p

    x.loc[curr_period, "smoothed_ssn"] = (
        x.loc[curr_period, "smoothed_ssn"].ffill().bfill()
    )
    roll = (
        x[train_short]
        .rolling(window=20, min_periods=5)
        .mean()
        .interpolate("linear", axis=0)
    )
    x.loc[curr_period, train_short] = x.loc[curr_period, train_short].fillna(roll)
    x.loc[curr_period, train_short] = x.loc[curr_period, train_short].ffill().bfill()

x.to_csv("xtest.csv", index=False)
