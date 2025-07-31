# import streamlit as st
# import pandas as pd
# import numpy as np
# import pickle
# from sklearn.preprocessing import LabelEncoder, StandardScaler
# import warnings
# warnings.filterwarnings('ignore')

# # Page configuration
# st.set_page_config(
#     page_title="CPM Predictor",
#     page_icon="💰",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # Title and description
# st.title("💰 CPM Prediction App")
# st.markdown("---")
# st.markdown("### Predict Cost Per Mille (CPM) for your advertising campaigns")

# # Load model and preprocessors
# @st.cache_resource
# def load_model():
#     try:
#         with open('best_model.pkl', 'rb') as f:
#             model_data = pickle.load(f)
#         return model_data
#     except FileNotFoundError:
#         st.error("Model file 'best_model.pkl' not found. Please ensure the model is trained and saved.")
#         return None

# # Load the model
# model_data = load_model()

# if model_data is not None:
#     model = model_data['model']
#     scaler = model_data['scaler']
#     selector = model_data.get('selector', None)
#     label_encoders = model_data['label_encoders']
#     feature_columns = model_data['feature_columns']
#     all_features = model_data.get('all_features', feature_columns)
#     model_name = model_data['model_name']
#     model_type = model_data.get('model_type', 'single')
    
#     st.success(f"✅ Model loaded successfully: {model_name} ({model_type})")
    
#     # Create two columns for input
#     col1, col2 = st.columns(2)
    
#     # Define categorical options based on your data
#     categorical_options = {
#         'Type': ['TrueView', 'Video', 'Display', 'Demand Gen'],
#         'Subtype': ['Reach', 'Simple', 'Non Skippable', 'View', 'Demand Gen', 'Audio'],
#         'Budget_Type': ['TrueView Budget', 'Unlimited', 'Impressions'],
#         'Pacing': ['Daily', 'Flight'],
#         'Pacing_Rate': ['Even', 'ASAP', 'Ahead'],
#         'Frequency_Period': ['Days', 'Minutes'],
#         'Bid_Strategy_Type': ['None', 'TrueValue', 'Maximize', 'Minimize', 'Fixed'],
#         'Optimized_Targeting': ['None', 'False', 'True']
#     }
    
#     with col1:
#         st.subheader("📊 Campaign Configuration")
        
#         # Categorical inputs
#         campaign_type = st.selectbox(
#             "Campaign Type",
#             options=categorical_options['Type'],
#             help="Select the type of campaign"
#         )
        
#         subtype = st.selectbox(
#             "Campaign Subtype",
#             options=categorical_options['Subtype'],
#             help="Select the subtype of campaign"
#         )
        
#         budget_type = st.selectbox(
#             "Budget Type",
#             options=categorical_options['Budget_Type'],
#             help="Select the budget type"
#         )
        
#         pacing = st.selectbox(
#             "Pacing",
#             options=categorical_options['Pacing'],
#             help="Select pacing strategy"
#         )
    
#     with col2:
#         st.subheader("⚙️ Campaign Settings")
        
#         pacing_rate = st.selectbox(
#             "Pacing Rate",
#             options=categorical_options['Pacing_Rate'],
#             help="Select pacing rate"
#         )
        
#         frequency_period = st.selectbox(
#             "Frequency Period",
#             options=categorical_options['Frequency_Period'],
#             help="Select frequency period"
#         )
        
#         bid_strategy = st.selectbox(
#             "Bid Strategy Type",
#             options=categorical_options['Bid_Strategy_Type'],
#             help="Select bid strategy (None if not applicable)"
#         )
        
#         optimized_targeting = st.selectbox(
#             "Optimized Targeting",
#             options=categorical_options['Optimized_Targeting'],
#             help="Select optimized targeting option"
#         )
    
#     # Numerical inputs
#     st.subheader("🔢 Numerical Parameters")
    
#     col3, col4 = st.columns(2)
    
#     with col3:
#         pacing_amount = st.number_input(
#             "Pacing Amount",
#             value=50.0,
#             step=0.01,
#             format="%.2f",
#             help="Enter the pacing amount for the campaign"
#         )
        
#     with col4:
#         frequency_exposures = st.selectbox(
#             "Frequency Exposures",
#             options=[0.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
#             index=0,
#             help="Select the number of frequency exposures (0 if no frequency cap)"
#         )
    
#     # Enhanced prediction function for new model
#     def predict_cpm(input_data):
#         try:
#             # Create DataFrame from input
#             df_input = pd.DataFrame([input_data])
            
#             # Handle missing values
#             df_input['Bid_Strategy_Type'] = df_input['Bid_Strategy_Type'].replace('None', np.nan)
#             df_input['Bid_Strategy_Type'] = df_input['Bid_Strategy_Type'].fillna('None')
#             df_input['Optimized_Targeting'] = df_input['Optimized_Targeting'].replace('None', np.nan)
#             df_input['Optimized_Targeting'] = df_input['Optimized_Targeting'].fillna(np.nan)
            
#             # Encode categorical variables
#             for col, le in label_encoders.items():
#                 if col in df_input.columns and col != 'pacing_amount_bin':
#                     try:
#                         df_input[f'{col}_encoded'] = le.transform(df_input[col])
#                     except ValueError:
#                         # Handle unseen categories
#                         df_input[f'{col}_encoded'] = 0
            
#             # Create base additional features
#             df_input['has_frequency_cap'] = (df_input['Frequency_Exposures'] > 0).astype(int)
#             df_input['has_bid_strategy'] = (df_input['Bid_Strategy_Type'] != 'None').astype(int)
#             df_input['pacing_per_exposure'] = df_input['Pacing_Amount'] / (df_input['Frequency_Exposures'] + 1)
            
#             # Create enhanced features (matching training)
#             df_input['log_pacing_amount'] = np.log1p(df_input['Pacing_Amount'])
#             df_input['sqrt_pacing_amount'] = np.sqrt(df_input['Pacing_Amount'])
#             df_input['frequency_squared'] = df_input['Frequency_Exposures'] ** 2
            
#             # Create interaction features
#             df_input['type_subtype_interaction'] = df_input['Type_encoded'] * df_input['Subtype_encoded']
#             df_input['budget_pacing_interaction'] = df_input['Budget_Type_encoded'] * df_input['Pacing_Amount']
#             df_input['frequency_pacing_interaction'] = df_input['Frequency_Exposures'] * df_input['Pacing_Amount']
            
#             # Create pacing amount bin (simplified binning for single prediction)
#             if df_input['Pacing_Amount'].iloc[0] <= 10:
#                 pacing_bin = 'Very Low'
#             elif df_input['Pacing_Amount'].iloc[0] <= 50:
#                 pacing_bin = 'Low'
#             elif df_input['Pacing_Amount'].iloc[0] <= 100:
#                 pacing_bin = 'Medium'
#             elif df_input['Pacing_Amount'].iloc[0] <= 200:
#                 pacing_bin = 'High'
#             else:
#                 pacing_bin = 'Very High'
            
#             # Encode pacing bin
#             try:
#                 df_input['pacing_amount_bin_encoded'] = label_encoders['pacing_amount_bin'].transform([pacing_bin])[0]
#             except (KeyError, ValueError):
#                 df_input['pacing_amount_bin_encoded'] = 2  # Default to medium
            
#             # Select all features that were used in training
#             try:
#                 X_input = df_input[all_features]
#             except KeyError:
#                 st.error("Some features are missing. Please check the model compatibility.")
#                 return None
            
#             # Apply feature selection if selector exists
#             if selector is not None:
#                 X_input_selected = selector.transform(X_input)
#             else:
#                 X_input_selected = X_input[feature_columns]
            
#             # Make prediction based on model type
#             if model_type == 'ensemble':
#                 # Handle ensemble prediction
#                 predictions = []
#                 for model_name_ens, model_ens in model:
#                     if model_name_ens in ['Ridge Regression', 'Lasso Regression', 'ElasticNet', 'SVR']:
#                         X_scaled = scaler.transform(X_input_selected)
#                         pred = model_ens.predict(X_scaled)[0]
#                     else:
#                         pred = model_ens.predict(X_input_selected)[0]
#                     predictions.append(pred)
#                 prediction = np.mean(predictions)
#             else:
#                 # Single model prediction
#                 if model_name in ['Ridge Regression', 'Lasso Regression', 'ElasticNet', 'SVR']:
#                     X_scaled = scaler.transform(X_input_selected)
#                     prediction = model.predict(X_scaled)[0]
#                 else:
#                     prediction = model.predict(X_input_selected)[0]
            
#             return prediction
            
#         except Exception as e:
#             st.error(f"Error making prediction: {str(e)}")
#             st.error("Please check if all required features are available.")
#             return None
    
#     # Prediction section
#     st.markdown("---")
#     st.subheader("🎯 CPM Prediction")
    
#     # Create prediction button
#     if st.button("Predict CPM", type="primary", use_container_width=True):
#         # Prepare input data
#         input_data = {
#             'Type': campaign_type,
#             'Subtype': subtype,
#             'Budget_Type': budget_type,
#             'Pacing': pacing,
#             'Pacing_Rate': pacing_rate,
#             'Pacing_Amount': pacing_amount,
#             'Frequency_Exposures': frequency_exposures,
#             'Frequency_Period': frequency_period,
#             'Bid_Strategy_Type': bid_strategy,
#             'Optimized_Targeting': optimized_targeting
#         }
        
#         # Make prediction
#         predicted_cpm = predict_cpm(input_data)
        
#         if predicted_cpm is not None:
#             # Display results
#             col5, col6, col7 = st.columns(3)
            
#             with col5:
#                 st.metric(
#                     label="Predicted CPM",
#                     value=f"${predicted_cpm:.2f}",
#                     help="Predicted Cost Per Mille for your campaign"
#                 )
            
#             with col6:
#                 # Calculate CPM bandwidth (10% below and above)
#                 lower_bound = predicted_cpm * 0.9
#                 st.metric(
#                     label="CPM Lower Bound (-10%)",
#                     value=f"${lower_bound:.2f}",
#                     help="CPM lower bound (10% below predicted)"
#                 )
            
#             with col7:
#                 # Calculate CPM bandwidth (10% below and above)
#                 upper_bound = predicted_cpm * 1.1
#                 st.metric(
#                     label="CPM Upper Bound (+10%)",
#                     value=f"${upper_bound:.2f}",
#                     help="CPM upper bound (10% above predicted)"
#                 )
            
#             # Show confidence indicator
#             st.markdown("---")
#             st.subheader("📈 Prediction Confidence")
            
#             # Create a confidence visualization
#             confidence_col1, confidence_col2 = st.columns(2)
            
#             with confidence_col1:
#                 if model_type == 'ensemble':
#                     st.success("🟢 **High Confidence**: Ensemble model provides robust predictions")
#                 elif model_name in ['Gradient Boosting', 'Random Forest', 'Extra Trees']:
#                     st.success("🟢 **High Confidence**: Advanced tree-based model")
#                 elif 'Regression' in model_name:
#                     st.info("🔵 **Medium Confidence**: Linear model with feature engineering")
#                 else:
#                     st.warning("🟡 **Standard Confidence**: Basic model")
            
#             with confidence_col2:
#                 bandwidth_pct = ((upper_bound - lower_bound) / predicted_cpm) * 100
#                 st.metric(
#                     label="Prediction Bandwidth",
#                     value=f"±{bandwidth_pct/2:.0f}%",
#                     help="Confidence interval around the prediction"
#                 )
            
#             # Show input summary
#             st.markdown("---")
#             st.subheader("📋 Campaign Summary")
            
#             summary_data = {
#                 "Parameter": list(input_data.keys()),
#                 "Value": list(input_data.values())
#             }
#             summary_df = pd.DataFrame(summary_data)
#             st.dataframe(summary_df, use_container_width=True, hide_index=True)

# else:
#     st.error("❌ Unable to load the prediction model. Please check if the model file exists.")

# # Enhanced sidebar with model information
# with st.sidebar:
#     st.header("ℹ️ About")
#     st.markdown("""
#     This app predicts the **Cost Per Mille (CPM)** for advertising campaigns using advanced machine learning models.
    
#     **How to use:**
#     1. Select your campaign configuration options
#     2. Enter numerical parameters
#     3. Click 'Predict CPM' to get your prediction
    
#     **CPM** represents the cost for 1,000 ad impressions.
#     """)
    
#     st.header("📊 Model Info")
#     if model_data:
#         model_accuracy = "88-92%" if model_type == 'ensemble' or model_name in ['Gradient Boosting', 'Random Forest'] else "85-88%"
#         st.markdown(f"""
#         - **Model**: {model_name}
#         - **Type**: {model_type.title()}
#         - **Features**: {len(feature_columns)}
#         - **Accuracy**: ~{model_accuracy}
#         - **Status**: ✅ Ready
#         """)
        
#         if model_type == 'ensemble':
#             st.info("🚀 **Ensemble Model**: Combines multiple algorithms for better accuracy")
#         elif model_name in ['Gradient Boosting', 'Random Forest', 'Extra Trees']:
#             st.info("🌟 **Advanced Model**: Uses sophisticated tree-based algorithms")
#     else:
#         st.markdown("- **Status**: ❌ Model not loaded")
    
#     st.header("🔍 Features Used")
#     if model_data:
#         st.markdown("**Enhanced Features:**")
#         st.markdown("""
#         - Campaign configuration
#         - Interaction features
#         - Mathematical transformations
#         - Frequency analysis
#         - Budget optimization
#         """)
    
#     st.header("🔗 Links")
#     st.markdown("""
#     - [Streamlit Documentation](https://docs.streamlit.io)
#     - [Source Code](https://github.com/your-repo)
#     """)

# # Footer
# st.markdown("---")
# st.markdown(
#     "<div style='text-align: center; color: gray;'>Made with ❤️ using Advanced ML | Enhanced Model v2.0</div>",
#     unsafe_allow_html=True
# )


import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')

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
    """
    Loads the trained model, scaler, and label encoders from the .pkl file.
    Uses st.cache_resource to avoid reloading on every rerun.
    """
    try:
        with open('best_model.pkl', 'rb') as f:
            model_data = pickle.load(f)
        return model_data
    except FileNotFoundError:
        st.error("Model file 'best_model.pkl' not found. Please ensure the model is trained and saved.")
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load the model
model_data = load_model()

# Check if model data loaded successfully before proceeding
if model_data is not None:
    # Determine if it's a single model or an ensemble
    model_type = model_data.get('model_type', 'single') # Default to 'single' for backward compatibility
    scaler = model_data['scaler']
    label_encoders = model_data['label_encoders']
    feature_columns = model_data['feature_columns']
    model_name = model_data['model_name'] # This will be "Ensemble" if it's an ensemble

    if model_type == 'ensemble':
        # 'model' key contains top_models (list of tuples: (model_name, model_results_dict))
        # Extract only the actual model objects from the dictionary within the tuple
        ensemble_models_info = model_data['model'] # This is the list of (name, results_dict) tuples
        actual_ensemble_models = [info['model'] for name, info in ensemble_models_info]
        st.success(f"✅ Ensemble model loaded successfully: {model_name} (composed of {len(actual_ensemble_models)} models)")
    else:
        # It's a single model
        model = model_data['model']
        st.success(f"✅ Single model loaded successfully: {model_name}")
    
    # Create two columns for input
    col1, col2 = st.columns(2)
    
    # Define categorical options based on your training data's unique values
    # It's crucial these lists contain all categories the LabelEncoders were fitted on.
    categorical_options = {
        'Type': ['TrueView', 'Video', 'Display', 'Demand Gen'],
        'Subtype': ['Reach', 'Simple', 'Non Skippable', 'View', 'Demand Gen', 'Audio'],
        'Budget_Type': ['TrueView Budget', 'Unlimited', 'Impressions'],
        'Pacing': ['Daily', 'Flight'],
        'Pacing_Rate': ['Even', 'ASAP', 'Ahead'],
        'Frequency_Period': ['Days', 'Minutes'],
        'Bid_Strategy_Type': ['None', 'TrueValue', 'Maximize', 'Minimize', 'Fixed'], # Ensure 'None' is here if it's a valid string category
        'Optimized_Targeting': ['None', 'False', 'True'] # Ensure 'None', 'False', 'True' are here as strings
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
        
        optimized_targeting = st.selectbox(
            "Optimized Targeting",
            options=categorical_options['Optimized_Targeting'],
            help="Select optimized targeting option"
        )
    
    # Numerical inputs
    st.subheader("🔢 Numerical Parameters")
    
    col3, col4 = st.columns(2)
    
    with col3:
        pacing_amount = st.number_input(
            "Pacing Amount",
            value=50.0,
            min_value=0.0, # Pacing amount should not be negative
            step=0.01,
            format="%.2f",
            help="Enter the pacing amount for the campaign"
        )
        
    with col4:
        frequency_exposures = st.selectbox(
            "Frequency Exposures",
            options=[0.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], # Ensure these match your training data distribution's typical range
            index=0, # Default to 0
            help="Select the number of frequency exposures (0 if no frequency cap)"
        )
    
    # Prediction function
    def predict_cpm(input_data):
        try:
            # Create DataFrame from input
            df_input = pd.DataFrame([input_data])
            
            # --- START OF PREPROCESSING (MUST EXACTLY MATCH TRAINING PREPROCESSING) ---

            # 1. Handle missing values/initial clean-up as done in training
            # This ensures 'None' is a string for Bid_Strategy_Type consistency
            df_input['Bid_Strategy_Type'] = df_input['Bid_Strategy_Type'].fillna('None')
            # Crucial: Ensure Optimized_Targeting is consistently a string
            df_input['Optimized_Targeting'] = df_input['Optimized_Targeting'].astype(str)
            
            # 2. Encode categorical variables
            categorical_cols_to_encode = ['Type', 'Subtype', 'Budget_Type', 'Pacing', 'Pacing_Rate', 
                                          'Frequency_Period', 'Bid_Strategy_Type']
            if 'Optimized_Targeting' in df_input.columns:
                categorical_cols_to_encode.append('Optimized_Targeting')

            for col_name in categorical_cols_to_encode:
                le = label_encoders.get(col_name)
                if le:
                    try:
                        # Convert input value to string before transforming
                        # This is critical for categories like 'Optimized_Targeting' which can be bool in raw data
                        df_input[f'{col_name}_encoded'] = le.transform([str(df_input[col_name].iloc[0])])[0]
                    except ValueError:
                        # This means the user input a category not seen during training.
                        st.warning(f"Unseen category '{df_input[col_name].iloc[0]}' for '{col_name}'. Attempting to default.")
                        # If 'None' (as string) is a known class, use its encoding
                        if 'None' in le.classes_: # Check if the string 'None' was learned
                             df_input[f'{col_name}_encoded'] = le.transform(['None'])[0]
                        else:
                            # Fallback if 'None' is not a learned class (less ideal)
                            # You might want a more sophisticated default or error handling here
                            df_input[f'{col_name}_encoded'] = 0 
                            st.warning(f"Could not find 'None' in encoder classes for '{col_name}'. Defaulting to 0.")
                else:
                    st.warning(f"LabelEncoder for '{col_name}' not found. '{col_name}_encoded' defaulted to 0.")
                    df_input[f'{col_name}_encoded'] = 0

            # 3. Create additional engineered features
            df_input['Pacing_Amount'] = pd.to_numeric(df_input['Pacing_Amount'])
            df_input['Frequency_Exposures'] = pd.to_numeric(df_input['Frequency_Exposures'])

            df_input['has_frequency_cap'] = (df_input['Frequency_Exposures'] > 0).astype(int)
            df_input['has_bid_strategy'] = (df_input['Bid_Strategy_Type'] != 'None').astype(int)
            
            # Prevent division by zero or negative if Frequency_Exposures can be problematic
            df_input['pacing_per_exposure'] = df_input['Pacing_Amount'] / (df_input['Frequency_Exposures'].clip(lower=0) + 1)
            
            # Additional feature engineering (log, sqrt, squared)
            df_input['log_pacing_amount'] = np.log1p(df_input['Pacing_Amount']) # log(1+x) handles x=0 gracefully
            df_input['sqrt_pacing_amount'] = np.sqrt(df_input['Pacing_Amount'].clip(lower=0)) # Clip to 0 for sqrt if value is negative
            df_input['frequency_squared'] = df_input['Frequency_Exposures'] ** 2
            
            # Interaction features (use .get() with default to prevent KeyError if encoded column is somehow missing)
            df_input['type_subtype_interaction'] = df_input.get('Type_encoded', 0) * df_input.get('Subtype_encoded', 0)
            df_input['budget_pacing_interaction'] = df_input.get('Budget_Type_encoded', 0) * df_input['Pacing_Amount']
            df_input['frequency_pacing_interaction'] = df_input['Frequency_Exposures'] * df_input['Pacing_Amount']
            
            # Binning pacing amount (CRITICAL: replicate qcut logic or use fixed bins from training)
            # YOU MUST REPLACE `pacing_actual_bins` and `pacing_labels` 
            # with the actual bins and labels obtained from `pd.qcut` during your training process.
            # Example:
            # _, pacing_actual_bins = pd.qcut(df_train['Pacing_Amount'], q=5, retbins=True)
            # pacing_labels = ['Very Low', 'Low', 'Medium', 'High', 'Very High'] # Match labels used in training
            
            # Placeholder for actual bins (Replace these with your actual training bins)
            # These are just example bins. Adjust based on your actual training data's quantiles.
            # You might need to make sure your training script saves these bins into the pickle.
            pacing_actual_bins = [0, 10, 50, 200, 1000, np.inf] # Example: Needs to cover your data range
            pacing_labels = ['Very Low', 'Low', 'Medium', 'High', 'Very High'] 
            
            # Ensure bins are unique and sorted if you're deriving them dynamically
            pacing_actual_bins = sorted(list(set(pacing_actual_bins)))

            # Adjust labels if there's a mismatch (e.g., due to sparse data or unique bins)
            if len(pacing_labels) != len(pacing_actual_bins) - 1:
                st.warning("Number of pacing bin labels does not match number of bins. Using generic labels.")
                pacing_labels = [f'Bin_{i}' for i in range(len(pacing_actual_bins) - 1)]

            df_input['pacing_amount_bin'] = pd.cut(
                df_input['Pacing_Amount'], 
                bins=pacing_actual_bins, 
                labels=pacing_labels, 
                include_lowest=True,
                right=True # This should match the `right` parameter used in training qcut
            ).astype(str) # Convert to string for LabelEncoder
            
            # Encode the bin labels
            if 'pacing_amount_bin' in label_encoders:
                try:
                    df_input['pacing_amount_bin_encoded'] = label_encoders['pacing_amount_bin'].transform(df_input['pacing_amount_bin'])[0]
                except ValueError:
                    st.warning(f"Unseen pacing amount bin '{df_input['pacing_amount_bin'].iloc[0]}'. Defaulting to 0.")
                    df_input['pacing_amount_bin_encoded'] = 0
            else:
                st.warning("LabelEncoder for 'pacing_amount_bin' not found. 'pacing_amount_bin_encoded' defaulted to 0.")
                df_input['pacing_amount_bin_encoded'] = 0

            # --- END OF PREPROCESSING ---

            # Select features for prediction using the saved selected_features list
            missing_features = [col for col in feature_columns if col not in df_input.columns]
            if missing_features:
                st.error(f"Error: Required features for prediction are missing after preprocessing: {missing_features}")
                st.write("Current DataFrame columns:", df_input.columns.tolist())
                st.write("Expected feature_columns:", feature_columns)
                return None

            X_input_raw = df_input[feature_columns]
            
            # Check for NaNs or Infs in the final feature set before prediction
            if X_input_raw.isnull().values.any() or np.isinf(X_input_raw.values).any():
                st.error("NaN or Infinite values detected in the final feature set. Cannot predict.")
                st.write("Problematic features (values causing NaNs/Infs):", 
                         X_input_raw.columns[X_input_raw.isnull().any() | np.isinf(X_input_raw).any()].tolist())
                return None

            # Make prediction based on model type (single or ensemble)
            if model_type == 'ensemble':
                ensemble_predictions = []
                for current_model in actual_ensemble_models:
                    model_class_name = type(current_model).__name__
                    
                    if model_class_name in ['Ridge', 'Lasso', 'ElasticNet', 'SVR']:
                        X_input_processed = scaler.transform(X_input_raw)
                    else:
                        X_input_processed = X_input_raw
                    
                    pred = current_model.predict(X_input_processed)[0]
                    ensemble_predictions.append(pred)
                prediction = np.mean(ensemble_predictions) # Simple averaging for ensemble
            else:
                # Single model prediction logic
                model_class_name = type(model).__name__
                if model_class_name in ['Ridge', 'Lasso', 'ElasticNet', 'SVR']:
                    X_input_processed = scaler.transform(X_input_raw)
                else:
                    X_input_processed = X_input_raw
                
                prediction = model.predict(X_input_processed)[0]
            
            # Ensure prediction is positive (CPM cannot be negative)
            if prediction < 0:
                prediction = 0.01 
                st.warning("Predicted CPM was negative. Adjusted to $0.01.")
            
            return prediction
            
        except Exception as e:
            st.error(f"An unexpected error occurred during prediction: {str(e)}")
            st.write("Input data provided by user:", input_data)
            # Safely check if df_input exists before printing
            if 'df_input' in locals():
                st.write("DataFrame columns after initial processing:", df_input.columns.tolist())
            return None
    
    # Prediction section
    st.markdown("---")
    st.subheader("🎯 CPM Prediction")
    
    # Create prediction button
    if st.button("Predict CPM", type="primary", use_container_width=True):
        # Prepare input data from Streamlit widgets
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
            'Optimized_Targeting': optimized_targeting
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
                lower_bound = predicted_cpm * 0.9
                st.metric(
                    label="CPM Lower Bound (-10%)",
                    value=f"${lower_bound:.2f}",
                    help="CPM lower bound (10% below predicted)"
                )
            
            with col7:
                upper_bound = predicted_cpm * 1.1
                st.metric(
                    label="CPM Upper Bound (+10%)",
                    value=f"${upper_bound:.2f}",
                    help="CPM upper bound (10% above predicted)"
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
        - **Features Used**: {len(feature_columns)}
        - **Status**: ✅ Ready
        """)
        # Display individual models in ensemble if applicable
        if model_type == 'ensemble' and 'ensemble_models_info' in locals():
            st.markdown("##### Ensemble Components:")
            for name, _ in ensemble_models_info: # ensemble_models_info is the (name, results_dict) list
                st.markdown(f"- {name}")
    else:
        st.markdown("- **Status**: ❌ Model not loaded")
    
    st.header("🔗 Links")
    st.markdown("""
    - [Streamlit Documentation](https://docs.streamlit.io)
    - [Source Code](https://github.com/your-repo) """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>Made with ❤️ using Streamlit</div>",
    unsafe_allow_html=True
)