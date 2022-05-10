import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as pyplot

df = pd.read_csv("MatchTimelinesFirst15.csv")

# Print csv data som describtive data med count, mean, std, min, interval, max.
print(df.describe())

# drop de columns som ikke er relavante.
df = df.drop('Unnamed: 0', axis=1)
df = df.drop('redDragonKills', axis=1)
df = df.drop('blueDragonKills', axis=1)
# Print de første 5 rows.
print(df.head())

fig_size = 15

# Boxplot med lambda expression fra Lærke
boxplot_data = {
    'victory': df['blue_win'].apply(lambda x: 'BlueWin' if x == 1 else 'RedWin'),
    'gold_diff': abs(df['blueGold'] - df['redGold'])
}

# Dataframe med victory og gold difference som er defineret i boxplot_data herover.
df_boxplot = pd.DataFrame(boxplot_data, columns=['victory', 'gold_diff'])
print(df_boxplot.head())

print(df_boxplot.shape)

# Sætter størrelsen af grafen med matplotlib figure.
pyplot.subplots(figsize=(fig_size, fig_size))

# opretter et boksplot med victory og gold_diff som akser.
sns.boxplot(x='victory', y='gold_diff', data=df_boxplot, order=['BlueWin', 'RedWin'], width=0.95)
pyplot.show()

print(df_boxplot['victory'].value_counts())

# Histogram ud fra victories og level differences.
hist_data = {
    'victory': df['blue_win'].apply(lambda x: 'BlueWin' if x == 1 else 'RedWin'),
    'level_diff': abs(df['blueAvgLevel'] - df['redAvgLevel'])
    }
df_histogram = pd.DataFrame(hist_data, columns=['victory', 'level_diff'])
print(df_histogram.head())

# Sætter størrelsen af grafen med matplotlib figure.
pyplot.subplots(figsize=(fig_size, fig_size))
sns.histplot(data=df_histogram, x='level_diff', multiple='dodge', hue='victory', shrink=5)
pyplot.show()

# Heatmap
df_heatmap_blue = pd.DataFrame(df.filter(regex='blue', axis=1))
df_heatmap_blue = df_heatmap_blue.drop('blueTowersDestroyed', axis=1)
df_heatmap_blue = pd.concat([df_heatmap_blue , df['redTowersDestroyed']], axis=1)

print(df_heatmap_blue)

corr = df_heatmap_blue.corr(method="spearman")
# Laver en mask med numPy, som udtrukker hvilke data bliver vist. zeros_like lave en matrix af 0'er
mask = np.zeros_like(corr, dtype=bool)
mask[np.triu_indices_from(mask)] = True
# Sætter størrelsen af grafen med matplotlib figure.
pyplot.subplots(figsize=(fig_size, fig_size))

sns.heatmap(corr, vmin=-1, vmax=1, mask=mask, annot=True, fmt=".2f")
pyplot.show()
