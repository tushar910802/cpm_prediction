# CPM Prediction App 💰

A Streamlit web application that predicts Cost Per Mille (CPM) for advertising campaigns using machine learning.

## 🚀 Live Demo

[Access the app here](https://your-app-name.streamlit.app) *(Replace with your actual Streamlit Community Cloud URL)*

## 📖 About

This application uses a trained machine learning model to predict CPM (Cost Per Mille) based on various campaign parameters including:

- Campaign Type and Subtype
- Budget Configuration
- Pacing Settings
- Frequency Controls
- Bid Strategy

## 🛠️ Features

- **Interactive Interface**: Easy-to-use dropdowns and input fields
- **Real-time Predictions**: Get instant CPM predictions
- **Cost Estimates**: View costs for 1K and 10K impressions
- **Campaign Summary**: Review your input parameters
- **Smart Interpretation**: Get insights about your predicted CPM

## 📊 Model Information

The app uses a trained regression model that analyzes historical campaign data to predict CPM values. The model considers both categorical and numerical features to provide accurate predictions.

## 🏃‍♂️ Running Locally

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. Clone this repository:
```bash
git clone https://github.com/your-username/cpm-prediction-app.git
cd cpm-prediction-app
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Ensure you have the trained model file:
   - The app expects a `best_model.pkl` file in the root directory
   - This file should contain the trained model, scaler, label encoders, and feature columns

4. Run the Streamlit app:
```bash
streamlit run app.py
```

5. Open your browser and navigate to `http://localhost:8501`

## 📁 File Structure

```
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── best_model.pkl        # Trained model file (you need to generate this)
└── model.csv             # Training data (optional, for model training)
```

## 🔧 Model Training

To train and save the model:

1. Ensure you have your training data in `model.csv`
2. Run your model training script to generate `best_model.pkl`
3. The pickle file should contain:
   - Trained model
   - Feature scaler
   - Label encoders
   - Feature column names
   - Model name

Example of saving the model:
```python
import pickle

model_data = {
    'model': best_model,
    'scaler': scaler,
    'label_encoders': label_encoders,
    'feature_columns': feature_columns,
    'model_name': best_model_name
}

with open('best_model.pkl', 'wb') as f:
    pickle.dump(model_data, f)
```

## 🌐 Deployment on Streamlit Community Cloud

1. **Push to GitHub**: Upload your code to a GitHub repository
2. **Include Required Files**: Ensure `app.py`, `requirements.txt`, and `best_model.pkl` are in your repo
3. **Deploy**: 
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Connect your GitHub account
   - Select your repository
   - Choose `app.py` as your main file
   - Click "Deploy"

## 📝 Usage

1. **Select Campaign Type**: Choose between TrueView and Video campaigns
2. **Configure Settings**: Select subtype, budget type, and other categorical parameters
3. **Enter Numbers**: Input pacing amount and frequency exposures
4. **Get Prediction**: Click "Predict CPM" to see your results
5. **Review Results**: Check the predicted CPM and cost estimates

## 🎯 Input Parameters

### Categorical Parameters:
- **Type**: Campaign type (TrueView, Video)
- **Subtype**: Campaign subtype (Reach, Simple, Non Skippable)
- **Budget Type**: Budget configuration (TrueView Budget, Unlimited)
- **Pacing**: Pacing strategy (Daily)
- **Pacing Rate**: Rate of pacing (Even)
- **Frequency Period**: Period for frequency capping (Days)
- **Bid Strategy Type**: Bidding strategy (None, Maximize)

### Numerical Parameters:
- **Pacing Amount**: Budget pacing amount (0-1000)
- **Frequency Exposures**: Number of frequency exposures (0-20)

## 📊 Output

The app provides:
- **Predicted CPM**: Main prediction result
- **Cost per 1K Impressions**: Direct cost calculation
- **Cost per 10K Impressions**: Scaled cost estimate
- **Campaign Summary**: Overview of input parameters
- **Interpretation**: Guidance on CPM level (Low/Medium/High)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

If you encounter any issues or have questions:

1. Check the [Issues](https://github.com/your-username/cpm-prediction-app/issues) page
2. Create a new issue if your problem isn't already listed
3. Provide detailed information about your problem

## 🙏 Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- Machine learning powered by [scikit-learn](https://scikit-learn.org/)
- Data processing with [pandas](https://pandas.pydata.org/)

---

Made with ❤️ for better advertising campaign optimization