# knn_model.py

import pickle
from prep_data import load_data
from sklearn.neighbors import KNeighborsClassifier

# Load the dataset
X_train, X_test, y_train, y_test = load_data()

# Train the KNN model
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

# Save the trained model to a file
with open("knnmodel.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved as knnmodel.pkl")
