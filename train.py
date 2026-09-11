from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import joblib

data = load_iris()
X, y = data.data, data.target

model = RandomForestClassifier()
model.fit(X, y)

joblib.dump(model, 'model.pkl')
print("Model successfully saved as model.pkl!")

