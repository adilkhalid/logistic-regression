import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix

from dataloader.breast_cancer_loader import BreastCancerDataLoader
from logistic_regression import LogisticRegression

# Step 1: Load only 2 features for 2D plotting
loader = BreastCancerDataLoader(
    feature_names=["mean radius", "mean texture"],  # pick 2 features
    normalize=True
)
split = loader.load_and_split_data()

# Step 2: Train the model
model = LogisticRegression(learning_rate=0.01, n_iters=1000)
model.fit(split.x_train, split.y_train)

# Step 3: Evaluate
y_pred = model.predict(split.x_test)
acc = accuracy_score(split.y_test, y_pred)
print(f"Accuracy: {acc:.4f}")

# Step 4: Plot decision boundary
x_min, x_max = split.x_train[:, 0].min() - 1, split.x_train[:, 0].max() + 1
y_min, y_max = split.x_train[:, 1].min() - 1, split.x_train[:, 1].max() + 1

xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                     np.linspace(y_min, y_max, 200))

grid_points = np.c_[xx.ravel(), yy.ravel()]
Z = model.predict(grid_points)
Z = Z.reshape(xx.shape)

plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.6)
plt.scatter(split.x_train[:, 0], split.x_train[:, 1], c=split.y_train, cmap=plt.cm.Spectral, edgecolors='k')
plt.xlabel("mean radius (normalized)")
plt.ylabel("mean texture (normalized)")
plt.title("Decision Boundary of Logistic Regression on Breast Cancer (2 features)")
plt.grid(True)
plt.tight_layout()
plt.savefig("decision_boundary.png")
plt.show()
