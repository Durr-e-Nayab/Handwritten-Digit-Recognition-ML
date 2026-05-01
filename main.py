from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns # Table ko khoobsurat banane ke liye

# 1. Load Data
digits = datasets.load_digits()
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

# 2. Split Data (50% training, 50% testing)
X_train, X_test, y_train, y_test = train_test_split(
    data, digits.target, test_size=0.5, shuffle=False)

# 3. Model Training
clf = svm.SVC(gamma=0.001)
clf.fit(X_train, y_train)

# 4. Predictions
predicted = clf.predict(X_test)

# --- EXTENSIONS START HERE ---

# A. Print Detailed Report in Terminal
print(f"Classification report for classifier {clf}:\n"
      f"{metrics.classification_report(y_test, predicted)}\n")

# B. Plot Confusion Matrix (Professional Table)
disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
disp.figure_.suptitle("Confusion Matrix")
print(f"Confusion matrix:\n{disp.confusion_matrix}")

# C. Show Predictions with Images
_, axes = plt.subplots(nrows=1, ncols=8, figsize=(15, 4))
for ax, image, prediction, actual in zip(axes, digits.images[n_samples // 2:], predicted, y_test):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title(f"P: {prediction}\n(A: {actual})")

plt.show()