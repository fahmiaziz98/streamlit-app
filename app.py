import streamlit as st
import numpy as np
import pandas as pd
import json
import joblib

st.write(""" # Predicted Rent House/Apartement/Room for the next 10 years """)
st.cache(allow_output_mutation=True)

bhk = st.slider(label='BHK', min_value=1, max_value=6, step=1)
area_type = st.selectbox('Area Type',('Super Area', 'Carpet Area'))
city = st.selectbox('City', ('Kolkata', 'Mumbai', 'Bangalore', 'Delhi', 'Chennai', 'Hyderabad'))
furn_s = st.selectbox('Furnishing Status', ('Unfurnished', 'Semi-Furnished', 'Furnished'))
tenant = st.selectbox('Tenant Preferred', ('Bachelors/Family', 'Bachelors', 'Family'))
bath = st.slider(label='Bathroom', min_value=1, max_value=7, step=1)
point_c = st.selectbox('Point of Contact', ('Contact Owner', 'Contact Agent'))
rent = st.slider(label='Rental Floor', min_value=-2, max_value=22, step=1)
total_f = st.slider(label='Total Number of Floor', min_value=0, max_value=30, step=1)
fixed_s = st.slider(label="Fixed Size Squere", min_value=10, max_value=3100, step=10)


def preprocessing():
    
    columns = [
        'BHK', 'Area Type', 'City', 'Furnishing Status', 
        'Tenant Preferred','Bathroom', 'Point of Contact',
        'Rental Floor', 'Total Number of Floor','Fixed Size'
    ]
    new_data = [[bhk, area_type, city, furn_s, tenant, bath, point_c, rent, total_f, fixed_s]]
    new_data = pd.DataFrame(new_data, columns=columns)
    return new_data


def predict(new_data):
    
    model = joblib.load('final_model.pkl')
    
    return model.predict(new_data)



def main():

    year = 2022
    increase = 0.1
    new_data = preprocessing()
    
    
    if st.button(label = 'Predict'):
        st.success("""
            Disclaimer ! This prediction is not entirely true, 
            because the rental price changes from time to time according 
            to the owner's policy""")
        pred =predict(new_data)
        for i in range(1, 11):
            up = (pred * increase) + pred
            increase = increase + 0.1
            years = year + i
            st.success(f"{years} : {np.round(up, 2)} Rupee")

st.cache(allow_output_mutation=True)


if __name__ == '__main__':
    main()

