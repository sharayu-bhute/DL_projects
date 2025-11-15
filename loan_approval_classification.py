import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
import matplotlib.pyplot
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow as tf
tf.random.set_seed(2)
from tensorflow import keras

df=pd.read_csv("loan_data.csv")

label=LabelEncoder()
for col in df.columns:
    if df[col].dtype=="object":
        df[col]=label.fit_transform(df[col])

X=df.drop(columns="loan_status",axis=1)
y=df['loan_status']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=2)

scaler=StandardScaler()
X_train_std=scaler.fit_transform(X_train)
X_test_std=scaler.transform(X_test)


model=keras.Sequential([
    keras.layers.Flatten(input_shape=(13,)),
    keras.layers.Dense(9,activation='relu'),
    keras.layers.Dense(1,activation='sigmoid')
])

model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy'])
model.fit(X_train_std,y_train,epochs=10)

loss,accuracy=model.evaluate(X_test_std,y_test)
print(f'Test accuracy: {accuracy*100:.2f}%')

def input_val(prompt,default,cast_type=str):
    user_val= input(f"{prompt},(default={default}:)")
    if user_val.strip=="":
        return default
    return cast_type(user_val)
person_age = input_val("Enter age", 30.0, float)

person_gender = input_val("Enter your gender (male/female)", "female")
person_gender = person_gender.strip().lower()

person_education = input_val(
    "Enter your level of education (Master/HighSchool/Bachelor/Associate)",
    "HighSchool"
)
person_education = person_education.strip()

person_income = input_val("Enter your income", 300000.0, float)

person_emp_exp = input_val(
    "Enter Employee Experience years [0, 1, 2, 3, 4, 5, 7]:",
    0,
    int
)

person_home_ownership = input_val(
    "Enter your type of house ['RENT', 'OWN', 'MORTGAGE', 'OTHER']:",
    "OTHER"
)
person_home_ownership = person_home_ownership.strip().upper()

loan_amnt = input_val("Enter the amount of loan you want:", 100000.00, float)

loan_intent = input_val(
    "Enter your intention for loan ['PERSONAL', 'EDUCATION', 'MEDICAL', 'VENTURE', 'HOMEIMPROVEMENT', 'DEBTCONSOLIDATION']",
    "EDUCATION"
)
loan_intent = loan_intent.strip().upper()

loan_int_rate = input_val("Enter the rate of interest:", 16.00, float)

loan_percent_income = input_val("Enter the loan percent income:", 0.49, float)

cb_person_cred_hist_length = input_val(
    "Enter credit history length in years [2.0, 3.0, 4.0]:",
    3.0,
    float
)

credit_score = input_val("Enter your credit score:", 680, float)

previous_loan_defaults_on_file = input_val(
    "Any previous loan default? (Yes/No):",
    "No"
)
previous_loan_defaults_on_file = previous_loan_defaults_on_file.strip().capitalize()

user_input_dict = {
    "person_age": [person_age],
    "person_gender": [person_gender],
    "person_education": [person_education],
    "person_income": [person_income],
    "person_emp_exp": [person_emp_exp],
    "person_home_ownership": [person_home_ownership],
    "loan_amnt": [loan_amnt],
    "loan_intent": [loan_intent],
    "loan_int_rate": [loan_int_rate],
    "loan_percent_income": [loan_percent_income],
    "cb_person_cred_hist_length": [cb_person_cred_hist_length],
    "credit_score": [credit_score],
    "previous_loan_defaults_on_file": [previous_loan_defaults_on_file]
}

input_df = pd.DataFrame(user_input_dict)

for col in ["person_gender", "person_education", "person_home_ownership",
            "loan_intent", "previous_loan_defaults_on_file"]:
    input_df[col] = label.fit_transform(input_df[col])

input_std = scaler.transform(input_df)

prediction = model.predict(input_std)[0][0]

if prediction > 0.5:
    print("The loan is APPROVED.")
else:
    print("The loan is NOT APPROVED.")

