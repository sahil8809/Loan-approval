import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load data
data = pd.read_csv(r"C:\Users\mdsah\OneDrive\Desktop\PYTHON BASICS\Machine Learning\projects\loan_features_dataset.csv")
# print (data)

# # Fill missing values
data = data.fillna(0)

# # Label Encoder
encoder = LabelEncoder()

gender_encoder = LabelEncoder()
education_encoder = LabelEncoder()
loan_encoder = LabelEncoder()

data['Gender'] = gender_encoder.fit_transform(data['Gender'])
data['Education'] = education_encoder.fit_transform(data['Education'])
data['Loan_Status'] = loan_encoder.fit_transform(data['Loan_Status'])


# # Features and Label
x = data[['Gender', 'Education', 'Income', 'LoanAmount']]
y = data['Loan_Status']

# # Split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42,shuffle=True,stratify=y)

# # Model
from sklearn.tree import DecisionTreeClassifier


model = DecisionTreeClassifier()

# # Train
model.fit(X_train, y_train)

# # Predict
y_pred = model.predict(X_test)

# # Accuracy
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test,y_pred))

joblib.dump(model, "loan_model.pkl")

joblib.dump(gender_encoder, "gender_encoder.pkl")
joblib.dump(education_encoder, "education_encoder.pkl")
joblib.dump(loan_encoder, "loan_encoder.pkl")
print("model & label encoder successfully saved")