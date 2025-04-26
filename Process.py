import pandas as pd

def process(data):
    data.dropna(inplace=True)
    data.drop_duplicates(inplace=True)
    data = data[data['Education_Level'] != 'Unknown']
    data = data[data['Marital_Status'] != 'Unknown']
    data = data[data['Income_Category'] != 'Unknown']
    data = data.drop('CLIENTNUM', axis=1)
    data['Gender'] = data['Gender'].map({'M': 1, 'F': 0})
    
    data = pd.get_dummies(data, drop_first=True)

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

    data = data.reindex(columns=FEATURE_COLUMNS, fill_value=0)
    
    return data