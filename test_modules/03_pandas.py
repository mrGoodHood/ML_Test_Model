import pandas as pd
import numpy as np

data = {
    "Name": ["Andrey", "Ivan", "Anna"],
    "Age": [25, 30, 22],
    "Height": [180, 175, 190]
}

# df = pd.DataFrame(data)
#
# print(df["Height"] > 179)

data_with_missing = {
    "Name": ["Andrey", "Ivan", "Anna", "Maria"],
    "Age": [25, 30, np.nan, 28],
    "Height": [180, np.nan, 190, 168]
}

# df_missing = pd.DataFrame(data_with_missing)
#
# print(df_missing["Age"].fillna(df_missing["Age"].mean()))

df = pd.read_csv('people.csv')
print(df.info())

