# 💼 Employee Salary Predictor

A machine learning web application that predicts employee salaries based on professional and demographic factors using Linear Regression.

## 🤔 What can this app do?

### 🎯 Salary Prediction Capabilities
This application can predict employee salaries based on key professional and demographic factors:

**📊 Input Parameters:**
- **Age** (18-65 years): Employee's current age
- **Gender**: Male or Female  
- **Education Level**: Bachelor's, Master's, or PhD
- **Job Title**: Various positions from the dataset
- **Experience**: Years of professional experience (0-40 years)

**🤖 Machine Learning Model:**
- **Algorithm**: Linear Regression
- **Current Accuracy**: 89.61% (R² Score)
- **Training Data**: Real employee salary dataset with 373 records

**💡 What you can expect:**
- Get instant salary predictions in Indian Rupees (₹)
- Compare salaries across different roles and experience levels
- Understand how education and experience impact compensation
- Make informed career and hiring decisions

**🎯 Perfect for:**
- Job seekers evaluating salary expectations
- HR professionals benchmarking compensation
- Career planning and progression analysis
- Salary negotiation preparation

## 📈 Model Performance & Dataset Info

**🎯 Model Statistics:**
- **Algorithm**: Linear Regression
- **R² Score**: 89.61%
- **Features Used**: 5 key parameters
- **Data Processing**: Standardized & Encoded

**📊 Dataset Overview:**
- **Total Records**: 373 employees
- **Job Titles**: 174 unique positions
- **Salary Range**: ₹350 - ₹250,000
- **Data Quality**: Clean, preprocessed data

## 🚀 Getting Started

### Prerequisites
- Python 3.7+
- Required packages listed in `requirements.txt`

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Loki0865/Employee_salary_Prediction_using-ML.git
   cd Employee_salary_Prediction_using-ML
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   streamlit run app.py
   ```

## 📊 Usage
1. Open the web application in your browser
2. Expand "What can this app do?" to understand the capabilities
3. Adjust the input parameters:
   - Select your age using the slider
   - Choose gender from the dropdown
   - Select education level
   - Pick a job title from available options
   - Set years of experience
4. Click "Predict Salary" to get your prediction
5. View additional model information in the expandable sections

## 🔧 Technical Details
- **Framework**: Streamlit for web interface
- **ML Library**: scikit-learn
- **Data Processing**: pandas, numpy
- **Model**: Linear Regression with feature scaling and encoding
- **Features**: Age (scaled), Gender (encoded), Education (encoded), Job Title (encoded), Experience (scaled)

## 📝 License
This project is open source and available under the [MIT License](LICENSE).