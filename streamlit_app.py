import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')

import os
# st.write("Current working directory:", os.getcwd())
# st.write("Files in current directory:", os.listdir('.'))

# Page configuration
# st.set_page_config(
#     page_title="CPM Predictor",
#     page_icon="💰",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# Title and description
st.title("💰 CPM Prediction App")
st.markdown("---")
st.markdown("### Predict Cost Per Mille (CPM) for your advertising campaigns")

# Load model and preprocessors
@st.cache_resource
def load_model():
    try:
        model_path = os.path.join(os.path.dirname(__file__), 'best_model.pkl')
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        return model_data
    except FileNotFoundError:
        st.error("Model file 'best_model.pkl' not found. Please ensure the model is trained and saved.")
        return None

# Load the model
model_data = load_model()

if model_data is not None:
    model = model_data['model']
    scaler = model_data['scaler']
    label_encoders = model_data['label_encoders']
    feature_columns = model_data['feature_columns']
    model_name = model_data['model_name']
    
    st.success(f"✅ Model loaded successfully: {model_name}")
    
    # Create two columns for input
    col1, col2 = st.columns(2)
    
    # Define categorical options based on your data
    categorical_options = {
        'Type': ['TrueView', 'Video'],
        'Subtype': ['Reach', 'Simple', 'Non Skippable'],
        'Budget_Type': ['TrueView Budget', 'Unlimited'],
        'Pacing': ['Daily'],
        'Pacing_Rate': ['Even'],
        'Frequency_Period': ['Days'],
        'Bid_Strategy_Type': ['None', 'Maximize']
    }
    
    with col1:
        st.subheader("📊 Campaign Configuration")
        
        # Categorical inputs
        campaign_type = st.selectbox(
            "Campaign Type",
            options=categorical_options['Type'],
            help="Select the type of campaign"
        )
        
        subtype = st.selectbox(
            "Campaign Subtype",
            options=categorical_options['Subtype'],
            help="Select the subtype of campaign"
        )
        
        budget_type = st.selectbox(
            "Budget Type",
            options=categorical_options['Budget_Type'],
            help="Select the budget type"
        )
        
        pacing = st.selectbox(
            "Pacing",
            options=categorical_options['Pacing'],
            help="Select pacing strategy"
        )
    
    with col2:
        st.subheader("⚙️ Campaign Settings")
        
        pacing_rate = st.selectbox(
            "Pacing Rate",
            options=categorical_options['Pacing_Rate'],
            help="Select pacing rate"
        )
        
        frequency_period = st.selectbox(
            "Frequency Period",
            options=categorical_options['Frequency_Period'],
            help="Select frequency period"
        )
        
        bid_strategy = st.selectbox(
            "Bid Strategy Type",
            options=categorical_options['Bid_Strategy_Type'],
            help="Select bid strategy (None if not applicable)"
        )
    
    # Numerical inputs
    st.subheader("🔢 Numerical Parameters")
    
    col3, col4 = st.columns(2)
    
    with col3:
        pacing_amount = st.number_input(
            "Pacing Amount",
            min_value=0.0,
            max_value=1000.0,
            value=50.0,
            step=1.0,
            help="Enter the pacing amount for the campaign"
        )
        
    with col4:
        frequency_exposures = st.number_input(
            "Frequency Exposures",
            min_value=0.0,
            max_value=20.0,
            value=0.0,
            step=1.0,
            help="Enter the number of frequency exposures (0 if no frequency cap)"
        )
    
    # Prediction function
    def predict_cpm(input_data):
        try:
            # Create DataFrame from input
            df_input = pd.DataFrame([input_data])
            
            # Handle missing values
            df_input['Bid_Strategy_Type'] = df_input['Bid_Strategy_Type'].replace('None', np.nan)
            df_input['Bid_Strategy_Type'] = df_input['Bid_Strategy_Type'].fillna('None')
            
            # Encode categorical variables
            for col, le in label_encoders.items():
                if col in df_input.columns:
                    try:
                        df_input[f'{col}_encoded'] = le.transform(df_input[col])
                    except ValueError:
                        # Handle unseen categories
                        df_input[f'{col}_encoded'] = 0
            
            # Create additional features
            df_input['has_frequency_cap'] = (df_input['Frequency_Exposures'] > 0).astype(int)
            df_input['has_bid_strategy'] = (df_input['Bid_Strategy_Type'] != 'None').astype(int)
            df_input['pacing_per_exposure'] = df_input['Pacing_Amount'] / (df_input['Frequency_Exposures'] + 1)
            
            # Select features for prediction
            X_input = df_input[feature_columns]
            
            # Make prediction
            if 'Regression' in model_name:
                X_input_scaled = scaler.transform(X_input)
                prediction = model.predict(X_input_scaled)[0]
            else:
                prediction = model.predict(X_input)[0]
            
            return prediction
            
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            return None
    
    # Prediction section
    st.markdown("---")
    st.subheader("🎯 CPM Prediction")
    
    # Create prediction button
    if st.button("Predict CPM", type="primary", use_container_width=True):
        # Prepare input data
        input_data = {
            'Type': campaign_type,
            'Subtype': subtype,
            'Budget_Type': budget_type,
            'Pacing': pacing,
            'Pacing_Rate': pacing_rate,
            'Pacing_Amount': pacing_amount,
            'Frequency_Exposures': frequency_exposures,
            'Frequency_Period': frequency_period,
            'Bid_Strategy_Type': bid_strategy
        }
        
        # Make prediction
        predicted_cpm = predict_cpm(input_data)
        
        if predicted_cpm is not None:
            # Display results
            col5, col6, col7 = st.columns(3)
            
            with col5:
                st.metric(
                    label="Predicted CPM",
                    value=f"${predicted_cpm:.2f}",
                    help="Predicted Cost Per Mille for your campaign"
                )
            
            with col6:
                # Calculate estimated cost for 1000 impressions
                estimated_cost_1k = predicted_cpm
                st.metric(
                    label="Cost per 1K Impressions",
                    value=f"${estimated_cost_1k:.2f}",
                    help="Cost for 1,000 impressions"
                )
            
            with col7:
                # Calculate estimated cost for 10k impressions
                estimated_cost_10k = predicted_cpm * 10
                st.metric(
                    label="Cost per 10K Impressions",
                    value=f"${estimated_cost_10k:.2f}",
                    help="Cost for 10,000 impressions"
                )
            
            # Show input summary
            st.markdown("---")
            st.subheader("📋 Campaign Summary")
            
            summary_data = {
                "Parameter": list(input_data.keys()),
                "Value": list(input_data.values())
            }
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True, hide_index=True)
            
            # Interpretation
            st.markdown("---")
            st.subheader("💡 Interpretation")
            
            if predicted_cpm < 10:
                st.success("🟢 **Low CPM**: This campaign configuration suggests a cost-effective advertising opportunity.")
            elif predicted_cpm < 25:
                st.warning("🟡 **Medium CPM**: This campaign has moderate costs. Consider optimizing targeting or bid strategy.")
            else:
                st.error("🔴 **High CPM**: This campaign configuration may be expensive. Review settings to optimize costs.")

else:
    st.error("❌ Unable to load the prediction model. Please check if the model file exists.")

# Sidebar with additional information
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    This app predicts the **Cost Per Mille (CPM)** for advertising campaigns based on various campaign parameters.
    
    **How to use:**
    1. Select your campaign configuration options
    2. Enter numerical parameters
    3. Click 'Predict CPM' to get your prediction
    
    **CPM** represents the cost for 1,000 ad impressions.
    """)
    
    st.header("📊 Model Info")
    if model_data:
        st.markdown(f"""
        - **Model**: {model_name}
        - **Features**: {len(feature_columns)}
        - **Status**: ✅ Ready
        """)
    else:
        st.markdown("- **Status**: ❌ Model not loaded")
    
    st.header("🔗 Links")
    st.markdown("""
    - [Streamlit Documentation](https://docs.streamlit.io)
    - [Source Code](https://github.com/your-repo)
    """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>Made with ❤️ using Streamlit</div>",
    unsafe_allow_html=True
)