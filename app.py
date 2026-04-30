import streamlit as st
import pandas as pd
import joblib

# ===============================
# Page Config
# ===============================
st.set_page_config(
    page_title="Credit Risk Assessment System",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)



# ===============================
# Theme
# ===============================
st.markdown(
    """
<style>
body, .stApp { background-color:#0b0b0b; color:#f5c77a; }
.block-container { padding: 2.5rem; }
.card {
  background: linear-gradient(145deg, #0f0f0f, #1a1a1a);
  border-radius: 18px;
  padding: 25px;
  margin-bottom: 20px;
  border: 1px solid rgba(245, 199, 122, 0.2);
  box-shadow: 0 0 25px rgba(245, 199, 122, 0.15);
}
h1, h2, h3 { color:#f5c77a !important; font-weight:800 !important; letter-spacing:1px; }
label { font-size:20px !important; font-weight:800 !important; color:#f5c77a !important; }
.stSelectbox > div { background-color:#121212 !important; border:1px solid #f5c77a !important; border-radius:10px; color:white !important; }
.stButton button {
  background: linear-gradient(90deg, #f5c77a, #ffd98e);
  color:black; border-radius:12px;
  padding:0.8rem 1.5rem;
  font-size:18px; font-weight:800;
  border:none; box-shadow: 0 0 15px rgba(245,199,122,0.4);
  transition:0.3s;
}
.stButton button:hover { transform:scale(1.05); box-shadow: 0 0 25px rgba(245,199,122,0.7); }
.sidebar .sidebar-content { background-color:#0f0f0f; }
</style>
""",
    unsafe_allow_html=True,
)

# ===
# ===============================
# Load Artifacts
# ===============================
def safe_load(path, name):
    try:
        return joblib.load(path)
    except Exception as e:
        st.error(f"Could not load {name} from {path}: {e}")
        return None

model = safe_load("svm_model.pkl", "SVM model")
rf_model = safe_load("rf_model.pkl", "Random Forest model")
ensemble_model = safe_load("ensemble_model.pkl", "Ensemble model")
scaler = safe_load("scaler.pkl", "scaler")
encoder = safe_load("encoder.pkl", "encoder")
selected_features = safe_load("selected_features.pkl", "selected features")

# ===============================
# Load Dataset (for feature options)
# ===============================
data = None
default_values = {}

try:
    path = "combined_solar_dataset.csv"
    data = pd.read_csv(path)

    data.columns = [
        'Gender',
        'Age',
        'Marital_Status',
        'Employment',
        'Residence',
        'Home_Ownership',
        'Number_Dependents',
        'Loan_Amount',
        'Decision'
    ]

    # Only keep important predictors
    important_features = ['Gender','Age','Marital_Status','Employment','Residence', 'Home_Ownership','Number_Dependents','Loan_Amount']

    for col in important_features:
        try:
            default_values[col] = data[col].mode(dropna=True).iloc[0]
        except Exception:
            default_values[col] = ""
except Exception as e:
    st.error(f"Could not load dataset: {e}")
    data = pd.DataFrame()
    important_features = []
    default_values = {}

# ===============================
# Sidebar
# ===============================
st.sidebar.title("💳 Credit Risk")
page = st.sidebar.radio(
    "Navigation", ["Home", "Predict", "Model Info", "Feature Guide", "About"]
)

# ===============================
# HOME
# ===============================
if page == "Home":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.title("💳 Credit Risk Assessment system")
    st.write(
        """
A **decision support system** built with:

• Boruta Feature Selection  
• Ordinal Encoding  
• Standard Scaling  
• Support Vector Machines  
• Random Forest Ensemble  

Designed for **real-world deployment**.

By BENSON T MAJAWA
"""
    )
    st.markdown("</div>", unsafe_allow_html=True)
# ===============================
# PREDICT
# ===============================
elif page == "Predict":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.title("🔮 Credit Risk Assessment")
    st.write("### Select values for the predictors:")

    if selected_features is None:
        st.error("Selected features are not loaded. Check selected_features.pkl.")
    elif data is None or data.empty:
        st.error("Dataset not loaded correctly. Check combined_solar_dataset.csv.")
    else:
        user_inputs = {}

        # Gender (categorical)
        gender_options = sorted(data['Gender'].dropna().unique().tolist())
        user_inputs['Gender'] = st.selectbox("🔸 Gender", gender_options)

        # Age (continuous, must be >= 18)
        age = st.number_input("🔸 Age", min_value=0, step=1)
        if age < 18:
            st.error("⚠️ Age must be 18 years or older.")
        user_inputs['Age'] = age

        # Marital Status (categorical)
        marital_options = sorted(data['Marital_Status'].dropna().unique().tolist())
        user_inputs['Marital_Status'] = st.selectbox("🔸 Marital Status", marital_options)

        # Employment (categorical)
        employment_options = sorted(data['Employment'].dropna().unique().tolist())
        employment = st.selectbox("🔸 Employment", employment_options)
        user_inputs['Employment'] = employment

        # Residence of Employer (only if not Unemployed)
        if employment != "Unemployed":
            residence_options = sorted(data['Residence'].dropna().unique().tolist())
            user_inputs['Residence'] = st.selectbox("🔸 Residence of Employer", residence_options)
        else:
            st.warning("⚠️ Residence of Employer not applicable for Unemployed.")
            user_inputs['Residence'] = None

        # Home Ownership (categorical, filter Employer if Self Employed)
        home_options = sorted(data['Home_Ownership'].dropna().unique().tolist())
        if employment == "Self Employed":   # match dataset spelling exactly
            home_options = [opt for opt in home_options if opt != "Employer"]
        user_inputs['Home_Ownership'] = st.selectbox("🔸 Home Ownership", home_options)

        # Number of Dependents (continuous)
        user_inputs['Number_Dependents'] = st.number_input("🔸 Number of Dependents", min_value=0, step=1)

        # Loan Amount (continuous)
        user_inputs['Loan_Amount'] = st.number_input("🔸 Loan Amount", min_value=0.0, step=100.0)

        # Sidebar model choice
        st.sidebar.subheader("Choose Model")
        model_choice = st.sidebar.radio("Select model:", ["SVM", "Random Forest", "Ensemble"])

        if st.button("✨ Predict Decision"):
            if age < 18:
                st.error("⚠️ Age must be 18 years or older to proceed.")
            else:
                try:
                    if scaler is None or encoder is None:
                        st.error("Scaler/encoder not loaded. Check pickle files.")
                    else:
                        full_input = dict(default_values)
                        full_input.update(user_inputs)

                        input_df = pd.DataFrame([full_input])[important_features]

                        # Split categorical vs continuous
                        categorical_cols = ['Gender','Marital_Status','Employment','Residence','Home_Ownership']
                        continuous_cols = ['Age','Number_Dependents','Loan_Amount']

                        # Encode categorical only
                        encoded_cat = encoder.transform(input_df[categorical_cols])
                        encoded_cat_df = pd.DataFrame(encoded_cat, columns=categorical_cols)

                        # Keep continuous as is
                        cont_df = input_df[continuous_cols].astype(float)

                        # Combine back
                        encoded_df = pd.concat([encoded_cat_df, cont_df], axis=1)

                        # Scale
                        scaled = scaler.transform(encoded_df)

                        # Predict
                        if model_choice == "SVM":
                            pred = model.predict(scaled)[0]
                            probs = model.predict_proba(scaled)[0]
                        elif model_choice == "Random Forest":
                            pred = rf_model.predict(scaled)[0]
                            probs = rf_model.predict_proba(scaled)[0]
                        else:
                            pred = ensemble_model.predict(scaled)[0]
                            probs = ensemble_model.predict_proba(scaled)[0]

                        st.markdown("---")
                        if pred == 1:
                            st.success("✅ Loan Approved (High Potential)")
                        else:
                            st.error("❌ Loan Rejected (Low Potential)")

                        col1, col2 = st.columns(2)
                        col1.metric("Approval Confidence", f"{probs[1]*100:.2f}%")
                        col2.metric("Rejection Confidence", f"{probs[0]*100:.2f}%")

                except Exception as e:
                    st.error(f"System Error: {e}")

    st.markdown("</div>", unsafe_allow_html=True)


# ===============================
# MODEL INFO
# ===============================
elif page == "Model Info":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.title("🧠 Model Information")
    st.write(
        """
**Models:**  
• Support Vector Machine (SVM)  
• Random Forest  
• Ensemble (VotingClassifier)  

**Feature Selection:** Boruta  
**Scaling:** StandardScaler  
**Encoding:** OrdinalEncoder
"""
    )

    st.subheader("Important Predictors:")
    for f in ['Gender','Age','Marital_Status','Employment','Residence', 'Home_Ownership','Number_Dependents','Loan_Amount']:
        st.write(f"• {f}")

    st.markdown("</div>", unsafe_allow_html=True)

# ===============================
# FEATURE GUIDE
# ===============================
elif page == "Feature Guide":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.title("📘 Feature Guide")

    if data is not None and not data.empty:
        for col in ['Gender','Age','Marital_Status','Employment','Residence', 'Home_Ownership','Number_Dependents','Loan_Amount']:
            st.subheader(col)
            if col in data.columns:
                try:
                    vals = sorted(data[col].dropna().unique().tolist())
                    st.write("Possible values:")
                    st.write(vals)
                except Exception as e:
                    st.write(f"Could not list values: {e}")
            else:
                st.write(f"Column '{col}' not found in loaded dataset.")
    else:
        st.warning("Dataset not available. Check combined_solar_dataset.csv.")

    st.markdown("</div>", unsafe_allow_html=True)

# ===============================
# ABOUT
# ===============================
elif page == "About":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.title("ℹ️ About")
    st.write(
        """
This system was engineered for:

• Real-world deployment  
• Decision support  

Built with reliability and scalability.

BY BENSON MAJAWA
"""
    )
    st.markdown("</div>", unsafe_allow_html=True)
