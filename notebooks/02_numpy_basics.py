import numpy as np

students_scores = np.array([
    [78, 85, 90, 72],
    [65, 70, 68, 75],
    [92, 88, 95, 91],
    [55, 60, 58, 63],
    [80, 82, 79, 85]
])

average_by_students = np.mean(students_scores, axis=1)
average_by_subjects = np.mean(students_scores, axis =0)
best_score_by_student = np.max(students_scores, axis=1)
best_student_index = np.argmax(average_by_students)
scores_above_80 = students_scores[students_scores > 80]
bonuses = np.array([2, 3, 1, 4])
final_scores = students_scores + bonuses
final_scores = np.clip(final_scores, 0, 100)

normalized_scores = (
    (final_scores - final_scores.min())
    / (final_scores.max() - final_scores.min())
)

print(normalized_scores)
