
import torch
import torch.nn as nn
import torch.optim as optim
import streamlit as st
from ucimlrepo import fetch_ucirepo
import pandas as pd

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from scipy.sparse import hstack
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)





st.set_page_config(
    page_title="Student Performance AI",
    page_icon="🎓",
    layout="wide"
)




student_performance = fetch_ucirepo(id=320)

x = student_performance.data.features
y = student_performance.data.targets.copy()




at_risk = (y['G3'] < 10).sum()

average = ((y['G3'] >= 10) & (y['G3'] <= 14)).sum()

high_performer = ((y['G3'] >= 15) & (y['G3'] <= 20)).sum()

y['performance_category'] = pd.cut(
    y['G3'],
    bins=[-1, 9, 14, 20],
    labels=['At Risk', 'Average', 'High Performer']
)




X = pd.concat([x, y[['G1', 'G2']]], axis=1)

Y = y['performance_category']



X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=42,
    stratify=Y
)




categorical_cols = X_train.select_dtypes(
    include='object'
).columns

numerical_cols = X_train.select_dtypes(
    include='number'
).columns



encoder = OneHotEncoder(handle_unknown='ignore')

X_train_cat = encoder.fit_transform(
    X_train[categorical_cols]
)

X_test_cat = encoder.transform(
    X_test[categorical_cols]
)




scaler = StandardScaler()

X_train_num = scaler.fit_transform(
    X_train[numerical_cols]
)

X_test_num = scaler.transform(
    X_test[numerical_cols]
)




X_train_final = hstack([
    X_train_cat,
    X_train_num
]).toarray()

X_test_final = hstack([
    X_test_cat,
    X_test_num
]).toarray()




rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train_final, Y_train)

Y_pred = rf_model.predict(X_test_final)




rf_accuracy = accuracy_score(
    Y_test,
    Y_pred
)

rf_report = classification_report(
    Y_test,
    Y_pred,
    labels=['At Risk', 'Average', 'High Performer'],
    output_dict=True,
    zero_division=0
)

rf_cm = confusion_matrix(
    Y_test,
    Y_pred,
    labels=['At Risk', 'Average', 'High Performer']
)



class_mapping = {
    'At Risk': 0,
    'Average': 1,
    'High Performer': 2
}

Y_train_encoded = Y_train.map(class_mapping)

Y_test_encoded = Y_test.map(class_mapping)



X_train_tensor = torch.tensor(
    X_train_final,
    dtype=torch.float32
)

X_test_tensor = torch.tensor(
    X_test_final,
    dtype=torch.float32
)

Y_train_tensor = torch.tensor(
    Y_train_encoded.to_numpy(),
    dtype=torch.long
)

Y_test_tensor = torch.tensor(
    Y_test_encoded.to_numpy(),
    dtype=torch.long
)



class StudentDNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(X_train_final.shape[1], 64),
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 3)

        )

    def forward(self, x):

        return self.network(x)


torch.manual_seed(42)

dnn_model = StudentDNN()



criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    dnn_model.parameters(),
    lr=0.001
)

epochs = 100




for epoch in range(epochs):

    outputs = dnn_model(X_train_tensor)

    loss = criterion(
        outputs,
        Y_train_tensor
    )

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    if (epoch + 1) % 10 == 0:

        print(
            f"Epoch [{epoch + 1}/{epochs}], "
            f"Loss: {loss.item():.4f}"
        )



dnn_model.eval()

with torch.no_grad():

    test_outputs = dnn_model(X_test_tensor)

    _, predicted = torch.max(test_outputs, 1)




dnn_accuracy = accuracy_score(
    Y_test_encoded,
    predicted.numpy()
)

dnn_report = classification_report(
    Y_test_encoded,
    predicted.numpy(),
    labels=[0, 1, 2],
    target_names=['At Risk', 'Average', 'High Performer'],
    output_dict=True,
    zero_division=0
)

dnn_cm = confusion_matrix(
    Y_test_encoded,
    predicted.numpy(),
    labels=[0, 1, 2]
)



comparison = pd.DataFrame({

    'Random Forest': [
        rf_accuracy,
        rf_report['macro avg']['precision'],
        rf_report['macro avg']['recall'],
        rf_report['macro avg']['f1-score']
    ],

    'DNN': [
        dnn_accuracy,
        dnn_report['macro avg']['precision'],
        dnn_report['macro avg']['recall'],
        dnn_report['macro avg']['f1-score']
    ]

}, index=[
    'Accuracy',
    'Macro Precision',
    'Macro Recall',
    'Macro F1-score'
])



st.markdown("""
<style>

.stApp {
    background-color: #0F172A;
    color: #F8FAFC;
}

[data-testid="stSidebar"] {
    background-color: #111827;
}

h1, h2, h3 {
    color: #F8FAFC !important;
}

p, label, .stMarkdown {
    color: #CBD5E1;
}

[data-testid="stMetric"] {
    background-color: #1E293B;
    border: 1px solid #334155;
    padding: 18px;
    border-radius: 12px;
}

.stButton > button {
    background-color: #2563EB;
    color: white;
    border-radius: 8px;
    border: none;
}

.stButton > button:hover {
    background-color: #1D4ED8;
    color: white;
}

[data-testid="stDataFrame"] {
    border: 1px solid #334155;
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------------------

st.sidebar.title("🎓 Student AI")

st.sidebar.caption("Academic Performance System")

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Exploratory Data Analysis",
        "Model Evaluation",
        "Student Prediction"
    ]
)


# --------------------------------------------------
# MAIN PAGE TITLE
# --------------------------------------------------

st.title("🎓 Student Performance Prediction")

st.caption(
    "AI-powered academic performance analysis and prediction"
)


# ==================================================
# DASHBOARD
# ==================================================

if page == "Dashboard":

    st.header("Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Students", len(y))

    col2.metric("At Risk", at_risk)

    col3.metric("Average", average)

    col4.metric("High Performers", high_performer)

    st.divider()

    st.subheader("Project Information")

    st.write("""
    This project uses machine learning to predict student
    academic performance based on demographic, academic,
    social and behavioural attributes.

    The final grade (G3) is divided into three categories:
    """)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.info("At Risk: 0–9")

    with col2:
        st.info("Average: 10–14")

    with col3:
        st.info("High Performer: 15–20")

    st.divider()

    st.subheader("Dataset Details")

    st.write("Dataset: UCI Student Performance")

    st.write("Dataset version: Portuguese")

    st.write("Total records:", len(x))

    st.write("Input features:", x.shape[1])


# ==================================================
# EXPLORATORY DATA ANALYSIS
# ==================================================

elif page == "Exploratory Data Analysis":

    st.header("Exploratory Data Analysis")

    st.subheader("Dataset Information")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Features", x.shape[1])

    col2.metric(
        "Numerical Features",
        len(x.select_dtypes(include='number').columns)
    )

    col3.metric(
        "Categorical Features",
        len(x.select_dtypes(include='object').columns)
    )

    st.write(
        "Missing values in features:",
        x.isnull().sum().sum()
    )

    st.write(
        "Missing values in targets:",
        y.isnull().sum().sum()
    )

    with st.expander("View Descriptive Statistics"):

        st.dataframe(
            x.describe(),
            use_container_width=True
        )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Performance Category Distribution")

        st.bar_chart(Y.value_counts())

    with col2:

        st.subheader("Study Time vs Final Grade")

        studytime_avg = y.groupby(
            x['studytime']
        )['G3'].mean()

        st.bar_chart(studytime_avg)

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Previous Failures vs Final Grade")

        failures_avg = y.groupby(
            x['failures']
        )['G3'].mean()

        st.bar_chart(failures_avg)

    with col2:

        st.subheader("Alcohol Consumption vs Final Grade")

        alcohol_avg = pd.DataFrame({

            'Weekday (Dalc)': y.groupby(
                x['Dalc']
            )['G3'].mean(),

            'Weekend (Walc)': y.groupby(
                x['Walc']
            )['G3'].mean()

        })

        st.line_chart(alcohol_avg)

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Age vs Final Grade")

        age_avg = y.groupby(
            x['age']
        )['G3'].mean()

        st.bar_chart(age_avg)

    with col2:

        st.subheader("Correlation with Final Grade")

        correlation = student_performance.data.original.corr(
            numeric_only=True
        )

        st.dataframe(
            correlation['G3'].sort_values(ascending=False),
            use_container_width=True
        )


# ==================================================
# MODEL EVALUATION
# ==================================================

elif page == "Model Evaluation":

    st.header("Deep Neural Network Evaluation")

    col1, col2 = st.columns(2)

    col1.metric(
        "DNN Model Accuracy",
        f"{dnn_accuracy * 100:.2f}%"
    )

    col2.metric(
        "Test Samples",
        len(Y_test)
    )

    st.divider()

    st.subheader("DNN Classification Report")

    dnn_report_df = pd.DataFrame(
        dnn_report
    ).transpose()

    st.dataframe(
        dnn_report_df.round(3),
        use_container_width=True
    )

    st.divider()

    st.subheader("DNN Confusion Matrix")

    dnn_cm_df = pd.DataFrame(
        dnn_cm,

        index=[
            'Actual At Risk',
            'Actual Average',
            'Actual High Performer'
        ],

        columns=[
            'Predicted At Risk',
            'Predicted Average',
            'Predicted High Performer'
        ]
    )

    st.dataframe(
        dnn_cm_df,
        use_container_width=True
    )

    st.caption(
        "Rows represent actual categories and columns represent predicted categories."
    )

    st.divider()

    st.subheader("Random Forest vs DNN Comparison")

    st.dataframe(
        comparison.round(3),
        use_container_width=True
    )


# ==================================================
# STUDENT PERFORMANCE PREDICTION
# ==================================================

elif page == "Student Prediction":

    st.header("Predict Student Performance")

    st.write(
        "Enter student details to predict their academic performance."
    )

    input_data = {}

    with st.form("student_prediction_form"):

        st.subheader("Student Details")

        col1, col2 = st.columns(2)

        for i, feature in enumerate(X_train.columns):

            with [col1, col2][i % 2]:

                if feature in categorical_cols:

                    input_data[feature] = st.selectbox(
                        feature,
                        options=sorted(
                            X_train[feature].unique().tolist()
                        )
                    )

                else:

                    input_data[feature] = st.number_input(
                        feature,

                        min_value=int(
                            X_train[feature].min()
                        ),

                        max_value=int(
                            X_train[feature].max()
                        ),

                        value=int(
                            X_train[feature].median()
                        ),

                        step=1
                    )

        submit = st.form_submit_button(
            "Predict Performance"
        )


    # --------------------------------------------------
    # PROCESS USER INPUT AND PREDICT
    # --------------------------------------------------

    if submit:

        input_df = pd.DataFrame(
            [input_data],
            columns=X_train.columns
        )

        # Categorical preprocessing
        input_cat = encoder.transform(
            input_df[categorical_cols]
        )

        # Numerical preprocessing
        input_num = scaler.transform(
            input_df[numerical_cols]
        )

        # Combine processed features
        input_final = hstack([
            input_cat,
            input_num
        ]).toarray()


        # RANDOM FOREST PREDICTION

        rf_prediction = rf_model.predict(
            input_final
        )[0]


        # DNN PREDICTION

        input_tensor = torch.tensor(
            input_final,
            dtype=torch.float32
        )

        dnn_model.eval()

        with torch.no_grad():

            dnn_output = dnn_model(input_tensor)

            dnn_predicted = torch.argmax(
                dnn_output,
                dim=1
            )


        reverse_mapping = {
            0: 'At Risk',
            1: 'Average',
            2: 'High Performer'
        }

        dnn_prediction = reverse_mapping[
            dnn_predicted.item()
        ]



        st.divider()


        st.subheader("Prediction Results")

        col1, col2 = st.columns(2)
    
    
        with col1:

            st.info("Random Forest Prediction")

            st.markdown(f"### {str(rf_prediction)}")

        with col2:

            st.success("Deep Neural Network Prediction")

            st.markdown(f"### {str(dnn_prediction)}")
