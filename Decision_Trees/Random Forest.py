from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#Load sample data
data = load_iris()
X = data.data
y = data.target

#Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#Initialize Random Forest model with feature randomness
model = RandomForestClassifier(n_estimators = 100, max_features = 'sqrt', random_state = 42)
model.fit(X_train,y_train)

#Predict and evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: ", accuracy)