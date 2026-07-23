import numpy as np
import pandas as pd

scores = np.array([72, 68, 75, 90, 81, 77, 84, 95, 73, 85])


mean_score = np.mean(scores)
median_score = np.median(scores)
score_range = np.max(scores) - np.min(scores)
variance = np.var(scores)
std_deviation = np.std(scores)
deviations = scores - mean_score
squared_deviations = deviations ** 2
manual_variance = np.mean(squared_deviations)


# print("Среднее значение:", mean_score)
# print("Медиана:", median_score)
# print("Размах:", score_range)
print("Дисперсия:", variance)
print("Стандартное отклонение:", std_deviation)

print(manual_variance)