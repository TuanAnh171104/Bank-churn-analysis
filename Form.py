import streamlit as st
import pickle
import numpy as np
import pandas as pd
from Process import process

with open("model.pkl", "rb") as model_file:
    model = pickle.load(model_file)


FEATURE_COLUMNS = ['Customer_Age', 'Gender', 'Dependent_count', 'Months_on_book',
                    'Total_Relationship_Count', 'Months_Inactive_12_mon', 'Contacts_Count_12_mon',
                    'Credit_Limit', 'Total_Revolving_Bal', 'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1',
                    'Total_Trans_Amt', 'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
                    'Education_Level_Doctorate', 'Education_Level_Graduate', 'Education_Level_High School',
                    'Education_Level_Post-Graduate', 'Education_Level_Uneducated',
                    'Marital_Status_Married', 'Marital_Status_Single',
                    'Income_Category_$40K - $60K', 'Income_Category_$60K - $80K',
                    'Income_Category_$80K - $120K', 'Income_Category_Less than $40K',
                    'Card_Category_Gold', 'Card_Category_Platinum', 'Card_Category_Silver']


st.title("Dự đoán Khách hàng Rời bỏ")

option = st.radio("Chọn phương thức nhập dữ liệu:", ("Nhập thủ công", "Tải file CSV"))

if option == "Nhập thủ công":
    input_features = {}
    for feature in FEATURE_COLUMNS:
        input_features[feature] = st.number_input(feature, min_value=0.0, step=0.1)
    
    if st.button("Dự đoán"):
        input_data = np.array([[input_features[f] for f in FEATURE_COLUMNS]])
        prediction = model.predict(input_data)[0]
        label = "Rời bỏ" if prediction == 1 else "Không rời bỏ"
        st.success(f"Khách hàng: {label}")

else:
    uploaded_file = st.file_uploader("Tải lên file CSV chứa dữ liệu khách hàng", type=["csv"])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df_process = process(df)
        df["Churn_Prediction"] = model.predict(df_process)
        df["Churn_Prediction"] = df["Churn_Prediction"].apply(lambda x: "Rời bỏ" if x == 1 else "Không rời bỏ")

        st.write("Dự đoán:")
        st.dataframe(df)
        
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(label="Tải xuống kết quả", data=csv, file_name="churn_predictions.csv", mime="text/csv")
