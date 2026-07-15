import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('people.csv')
# print(df.head())

correlation = df[["Age", "Height"]].corr()

print(correlation)

plt.figure(figsize=(6, 4))

sns.heatmap(correlation,
            annot=True,
            cmap="coolwarm",
            vmin=-1,
            vmax=1)

plt.title("Correlation Matrix")

plt.show()