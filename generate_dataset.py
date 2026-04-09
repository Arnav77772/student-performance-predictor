import pandas as pd
import numpy as np

np.random.seed(42)
n_students = 1000

# Features banana
study_hours = np.random.normal(6, 2, n_students).clip(1, 12)
attendance = np.random.normal(75, 15, n_students).clip(30, 100)
internal_marks = np.random.normal(65, 15, n_students).clip(20, 100)
previous_marks = np.random.normal(60, 20, n_students).clip(20, 100)

# Exam score calculate karna (realistic formula)
exam_score = (
    0.3 * study_hours * 8 +
    0.25 * attendance * 0.8 +
    0.25 * internal_marks +
    0.20 * previous_marks +
    np.random.normal(0, 5, n_students)
).clip(0, 100)

# Pass/Fail column
pass_fail = (exam_score >= 40).astype(int)

# DataFrame banana
df = pd.DataFrame({
    'study_hours': study_hours.round(1),
    'attendance_percent': attendance.round(1),
    'internal_marks': internal_marks.round(1),
    'previous_marks': previous_marks.round(1),
    'exam_score': exam_score.round(1),
    'pass_fail': pass_fail
})

# CSV save karna
df.to_csv('student_data.csv', index=False)
print(f"Dataset ready! {n_students} students ka data save hua ✅")
print(df.head())
