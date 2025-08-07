import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')
import os

# Page configuration
st.set_page_config(
    page_title="CPM Predictor",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
        'Type': ['TrueView', 'Video', 'Display', 'Demand Gen'],
        'Subtype': ['Reach', 'Simple', 'Non Skippable', 'View', 'Demand Gen', 'Audio'],
        'Budget_Type': ['TrueView Budget', 'Unlimited', 'Impressions'],
        'Pacing': ['Daily', 'Flight'],
        'Pacing_Rate': ['Even', 'ASAP', 'Ahead'],
        'Frequency_Period': ['Days', 'Minutes'],
        'Bid_Strategy_Type': ['None', 'TrueValue', 'Maximize', 'Minimize', 'Fixed'],
        'Optimized_Targeting': ['None', 'False', 'True'],
        # START: ADDED LINES
        'Bid_Strategy_Unit': ['None', 'Unknown', 'AV_VIEWED', 'CPC'] 
        # END: ADDED LINES
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
        
        # START: ADDED LINES
        bid_strategy_unit = st.selectbox(
            "Bid Strategy Unit",
            options=categorical_options['Bid_Strategy_Unit'],
            help="Select the unit for the bid strategy"
        )
        # END: ADDED LINES

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
        
        optimized_targeting = st.selectbox(
            "Optimized Targeting",
            options=categorical_options['Optimized_Targeting'],
            help="Select optimized targeting option"
        )

        # START: ADDED LINES
        bid_strategy_value = st.number_input(
            "Bid Strategy Value",
            value=0.0,
            step=0.01,
            format="%.2f",
            help="Enter the value for the bid strategy (0 if not applicable)"
        )
        # END: ADDED LINES
    
    # Numerical inputs
    st.subheader("🔢 Numerical Parameters")
    
    col3, col4 = st.columns(2)
    
    with col3:
        pacing_amount = st.number_input(
            "Pacing Amount",
            value=50.0,
            step=0.01,
            format="%.2f",
            help="Enter the pacing amount for the campaign"
        )
        
    with col4:
        frequency_exposures = st.selectbox(
            "Frequency Exposures",
            options=[0.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            index=0,
            help="Select the number of frequency exposures (0 if no frequency cap)"
        )
    
    # Prediction function
    def predict_cpm(input_data):
        try:
            # Create DataFrame from input
            df_input = pd.DataFrame([input_data])
            
            # START: MODIFIED LINES
            # Handle missing values and ensure correct data types
            for col in ['Bid_Strategy_Type', 'Optimized_Targeting', 'Bid_Strategy_Unit']:
                df_input[col] = df_input[col].fillna('None').astype(str)
            # END: MODIFIED LINES
            df_input['Bid_Strategy_Value'] = df_input['Bid_Strategy_Value'].fillna(0.0)

            # Re-create all engineered features used in the training script
            df_input['has_frequency_cap'] = (df_input['Frequency_Exposures'] > 0).astype(int)
            df_input['has_bid_strategy'] = (df_input['Bid_Strategy_Type'] != 'None').astype(int)
            df_input['pacing_per_exposure'] = df_input['Pacing_Amount'] / (df_input['Frequency_Exposures'] + 1)
            
            # START: ADDED LINES
            df_input['has_optimized_targeting'] = (df_input['Optimized_Targeting'] != 'None').astype(int)
            df_input['has_bid_strategy_unit'] = (df_input['Bid_Strategy_Unit'] != 'None').astype(int)
            df_input['has_bid_strategy_value'] = (df_input['Bid_Strategy_Value'] > 0).astype(int)
            
            df_input['log_pacing_amount'] = np.log1p(df_input['Pacing_Amount'])
            df_input['sqrt_pacing_amount'] = np.sqrt(df_input['Pacing_Amount'])
            df_input['frequency_squared'] = df_input['Frequency_Exposures'] ** 2
            
            # Interaction features
            df_input['type_subtype_interaction'] = 0
            df_input['budget_pacing_interaction'] = df_input['Pacing_Amount'] * 0
            df_input['frequency_pacing_interaction'] = df_input['Frequency_Exposures'] * df_input['Pacing_Amount']
            df_input['bid_strategy_value_interaction'] = df_input['Bid_Strategy_Value'] * df_input['has_bid_strategy']
            df_input['targeting_pacing_interaction'] = df_input['has_optimized_targeting'] * df_input['Pacing_Amount']
            
            # Binning features (simplified for single prediction)
            pacing_amount = df_input['Pacing_Amount'].iloc[0]
            if pacing_amount <= 10:
                pacing_bin_encoded = 0
            elif pacing_amount <= 50:
                pacing_bin_encoded = 1
            elif pacing_amount <= 100:
                pacing_bin_encoded = 2
            elif pacing_amount <= 200:
                pacing_bin_encoded = 3
            else:
                pacing_bin_encoded = 4
            df_input['pacing_amount_bin_encoded'] = pacing_bin_encoded

            bid_value = df_input['Bid_Strategy_Value'].iloc[0]
            if bid_value == 0:
                bid_value_bin_encoded = 0
            else:
                bid_value_bin_encoded = 1
            df_input['bid_value_bin_encoded'] = bid_value_bin_encoded
            # END: ADDED LINES

            # Encode categorical variables
            for col, le in label_encoders.items():
                if f'{col}_encoded' in feature_columns:
                    try:
                        df_input[f'{col}_encoded'] = le.transform(df_input[col].astype(str))
                    except ValueError:
                        df_input[f'{col}_encoded'] = 0

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
            'Bid_Strategy_Type': bid_strategy,
            'Optimized_Targeting': optimized_targeting,
            # START: ADDED LINES
            'Bid_Strategy_Unit': bid_strategy_unit,
            'Bid_Strategy_Value': bid_strategy_value
            # END: ADDED LINES
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