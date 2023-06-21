import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle


def eda():
    st.sidebar.header("Visualizations")
    
    st.header("Upload your CSV data")
    data_file = st.file_uploader("Upload CSV", type=["csv"])
    
    if data_file is not None:
        data = pd.read_csv(data_file)
        st.write("Data Overview:")
        st.write(data.head())
        st.write(data.describe().T)
        
        plot_options = ["Bar plot", "Scatter plot", "Histogram", "Box plot"]
        selected_plot = st.sidebar.selectbox("Choose a plot type", plot_options)
        
        if selected_plot == "Bar plot":
            x_axis = st.sidebar.selectbox("Select x-axis", data.columns)
            y_axis = st.sidebar.selectbox("Select y-axis", data.columns)
            st.write("Bar plot:")
            fig, ax = plt.subplots()
            sns.barplot(x=data[x_axis], y=data[y_axis], ax=ax)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")  
            st.pyplot(fig)
            
        elif selected_plot == "Scatter plot":
            x_axis = st.sidebar.selectbox("Select x-axis", data.columns)
            y_axis = st.sidebar.selectbox("Select y-axis", data.columns)
            st.write("Scatter plot:")
            fig, ax = plt.subplots()
            sns.scatterplot(x=data[x_axis], y=data[y_axis], ax=ax)
            st.pyplot(fig)
            
        elif selected_plot == "Histogram":
            column = st.sidebar.selectbox("Select a column", data.columns)
            bins = st.sidebar.slider("Number of bins", 5, 100, 20)
            st.write("Histogram:")
            fig, ax = plt.subplots()
            sns.histplot(data[column], bins=bins, ax=ax)
            st.pyplot(fig)

        elif selected_plot == "Box plot":
            column = st.sidebar.selectbox("Select a column", data.columns)
            st.write("Box plot:")
            fig, ax = plt.subplots()
            sns.boxplot(data[column], ax=ax)
            st.pyplot(fig)
            

def input_data():
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
    
    columns = [
        'BHK', 'Area Type', 'City', 'Furnishing Status', 
        'Tenant Preferred','Bathroom', 'Point of Contact',
        'Rental Floor', 'Total Number of Floor','Fixed Size'
    ]
    new_data = [[bhk, area_type, city, furn_s, tenant, bath, point_c, rent, total_f, fixed_s]]
    new_data = pd.DataFrame(new_data, columns=columns)
    return new_data


def predict():
    st.write("""Predicted Rent House/Apartement/Room""")
    with open("final_model_v2.pkl", "rb") as f:
        model = pickle.load(f)
    
    new_data = input_data()
    if st.button(label='Predict'):
        charges_pred = model.predict(new_data)
        st.success(f'Apartment Price Estimate (INR): {np.round(charges_pred[0], 2)}')

        
pages = {"EDA":eda, "Predict":predict}


def main():
    st.title("ApartemenPriceEstimator && Exploratory Data Analyst")
    selected_page = st.sidebar.selectbox("Choose a page", options=list(pages.keys()))

    pages[selected_page]()


st.cache(allow_output_mutation=True)
if __name__ == "__main__":
    main()   
        