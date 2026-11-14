import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

X = np.array([[25, 115], [32, 120], [54, 155], [63, 170]])
y = np.array([0, 0, 1, 1])

model = LogisticRegression().fit(X, y)
joblib.dump(model, "model.pkl")
print("SUKSES: model.pkl berhasil dibuat.")