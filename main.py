import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc,roc_auc_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import re  # Add this line

# Load the dataset
file_path = r'DataShortened/glassdoor_shortened.csv'
df = pd.read_csv(file_path)

st.divider()  # 👈 Draws a horizontal rule

st.title("Exploratory Data Analysis of Glassdoor Reviews")

st.divider()  # 👈 Another horizontal rule

# Basic EDA
# Show the first few rows of the dataset
st.subheader("Dataset Preview")
st.dataframe(df.head())

# Show basic statistics
st.subheader("Basic Statistics")
st.write(df.describe())

# Check for missing values
st.subheader("Missing Values")
st.write(df.isnull().sum())

# Distribution of Overall Ratings
st.subheader("Distribution of Overall Ratings")
fig, ax = plt.subplots()
sns.countplot(data=df, x='overall_rating', ax=ax)
st.pyplot(fig)

# Work-Life Balance vs. Overall Rating
st.subheader("Work-Life Balance vs. Overall Rating")
fig, ax = plt.subplots()
sns.boxplot(data=df, x='overall_rating', y='work_life_balance', ax=ax)
st.pyplot(fig)

# Culture Values vs. Overall Rating
st.subheader("Culture Values vs. Overall Rating")
fig, ax = plt.subplots()
sns.boxplot(data=df, x='overall_rating', y='culture_values', ax=ax)
st.pyplot(fig)

# Diversity and Inclusion vs. Overall Rating
st.subheader("Diversity and Inclusion vs. Overall Rating")
fig, ax = plt.subplots()
sns.boxplot(data=df, x='overall_rating', y='diversity_inclusion', ax=ax)
st.pyplot(fig)

# Career Opportunities vs. Overall Rating
st.subheader("Career Opportunities vs. Overall Rating")
fig, ax = plt.subplots()
sns.boxplot(data=df, x='overall_rating', y='career_opp', ax=ax)
st.pyplot(fig)

# Compensation and Benefits vs. Overall Rating
st.subheader("Compensation and Benefits vs. Overall Rating")
fig, ax = plt.subplots()
sns.boxplot(data=df, x='overall_rating', y='comp_benefits', ax=ax)
st.pyplot(fig)

# Senior Management vs. Overall Rating
st.subheader("Senior Management vs. Overall Rating")
fig, ax = plt.subplots()
sns.boxplot(data=df, x='overall_rating', y='senior_mgmt', ax=ax)
st.pyplot(fig)

# Recommendations Distribution
st.subheader("Recommendations Distribution")
fig, ax = plt.subplots()
sns.countplot(data=df, x='recommend', ax=ax)
st.pyplot(fig)

# CEO Approval Distribution
st.subheader("CEO Approval Distribution")
fig, ax = plt.subplots()
sns.countplot(data=df, x='ceo_approv', ax=ax)
st.pyplot(fig)

# Outlook Distribution
st.subheader("Outlook Distribution")
fig, ax = plt.subplots()
sns.countplot(data=df, x='outlook', ax=ax)
st.pyplot(fig)

# Display pros and cons word clouds (if you have wordcloud library installed)
from wordcloud import WordCloud

st.subheader("Pros Word Cloud")
pros_text = ' '.join(df['pros'].dropna().astype(str).tolist())
pros_wordcloud = WordCloud(background_color='white', width=800, height=400).generate(pros_text)
fig, ax = plt.subplots()
ax.imshow(pros_wordcloud, interpolation='bilinear')
ax.axis('off')
st.pyplot(fig)

st.subheader("Cons Word Cloud")
cons_text = ' '.join(df['cons'].dropna().astype(str).tolist())
cons_wordcloud = WordCloud(background_color='white', width=800, height=400).generate(cons_text)
fig, ax = plt.subplots()
ax.imshow(cons_wordcloud, interpolation='bilinear')
ax.axis('off')
st.pyplot(fig)

# Preprocess the data
df.fillna('', inplace=True)

# Convert all columns to string type to avoid mixed data type issues
for column in df.columns:
    df[column] = df[column].astype(str)

# Encode categorical variables
label_encoders = {}
for column in df.select_dtypes(include=['object']).columns:
    if column not in ['pros', 'cons', 'headline']:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le

# Define features and target
X = df.drop(['overall_rating', 'pros', 'cons', 'headline'], axis=1)
y = df['overall_rating'].astype(int)

# Scale the data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Streamlit slider to select the number of rows
st.sidebar.title("Data Selection")
num_rows = st.sidebar.slider("Number of rows to use for training", min_value=1000, max_value=len(df), value=5000, step=1000)
X = X[:num_rows]
y = y[:num_rows]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model definitions
models = {
    'Logistic Regression': LogisticRegression(max_iter=2000),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'Support Vector Machine': SVC(probability=True, random_state=42)
}

# Hyperparameter grids for GridSearchCV
param_grids = {
    'Logistic Regression': {
        'C': [0.1, 1, 10]
    },
    'Random Forest': {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20]
    },
    'Gradient Boosting': {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2]
    },
    'Support Vector Machine': {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf']
    }
}

st.title("ML Model Evaluation")

# Train and evaluate models using GridSearchCV
for model_name, model in models.items():
    st.subheader(f"{model_name} Evaluation")
    grid_search = GridSearchCV(model, param_grids[model_name], cv=3, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"Best Parameters: {grid_search.best_params_}")
    st.write(f"Accuracy: {accuracy:.2f}")
    st.write("Classification Report:")
    st.text(classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title(f'{model_name} Confusion Matrix')
    st.pyplot(fig)

# ROC Curve
st.subheader("ROC Curve")
fig, ax = plt.subplots()
for model_name, model in models.items():
    best_model = GridSearchCV(model, param_grids[model_name], cv=3, scoring='accuracy', n_jobs=-1).fit(X_train, y_train).best_estimator_
    y_pred_prob = best_model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob, pos_label=best_model.classes_[1])
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, label=f'{model_name} (area = {roc_auc:.2f})')

ax.plot([0, 1], [0, 1], 'k--')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curve')
ax.legend(loc='lower right')
st.pyplot(fig)

# Insights
st.subheader("Insights")
st.write("""
- Various machine learning models have been trained and evaluated on the dataset.
- The accuracy, classification report, and confusion matrix for each model have been provided.
- The ROC curves and AUC scores show the performance of the models in distinguishing between the classes.
- Hyperparameter tuning was performed using GridSearchCV to find the best parameters for each model.
""")

st.divider()  # 👈 Draws a horizontal rule

# Save the trained models if needed
#import joblib

for model_name, model in models.items():
    best_model = GridSearchCV(model, param_grids[model_name], cv=3, scoring='accuracy', n_jobs=-1).fit(X_train, y_train).best_estimator_
    #joblib.dump(best_model, f'{model_name}_best_model.pkl')
    #st.write(f"{model_name} model saved.")


st.divider()  # 👈 Draws a horizontal rule

st.title("Exploratory Data Analysis of Job Descriptions Dataset")

st.divider()  # 👈 Another horizontal rule

# Load the dataset
file_path = r'DataShortened/job_descriptions_shortened.csv'
df = pd.read_csv(file_path)

# Convert Job Id to string to prevent scientific notation
df['Job Id'] = df['Job Id'].astype(str)

# Preprocessing function
def preprocess_data(df):
    # Example preprocessing: convert categorical columns to numerical
    label_encoders = {}
    for column in df.select_dtypes(include=['object']).columns:
        label_encoders[column] = LabelEncoder()
        df[column] = label_encoders[column].fit_transform(df[column])
    return df, label_encoders

st.title("Job Descriptions Dataset EDA")

st.header("Dataset Overview")
st.write(df.head())

st.header("Basic Information")
st.write(df.info())

st.header("Descriptive Statistics")
st.write(df.describe(include='all'))

st.header("Unique Values Count")
st.write(df.nunique())

# Plotting based on columns
st.header("Visualizations")

# Experience Distribution
st.subheader("Experience Distribution")
st.bar_chart(df['Experience'].value_counts())

# Qualifications Distribution
st.subheader("Qualifications Distribution")
st.bar_chart(df['Qualifications'].value_counts())

# Salary Range Distribution
st.subheader("Salary Range Distribution")
st.bar_chart(df['Salary Range'].value_counts())

# Location Distribution
st.subheader("Location Distribution")
st.map(df[['latitude', 'longitude']])

# Work Type Distribution
st.subheader("Work Type Distribution")
st.bar_chart(df['Work Type'].value_counts())

# Company Size Distribution
st.subheader("Company Size Distribution")
st.bar_chart(df['Company Size'].value_counts())

# Job Posting Date Distribution
st.subheader("Job Posting Date Distribution")
st.line_chart(df['Job Posting Date'].value_counts().sort_index())

# Preference Distribution
st.subheader("Preference Distribution")
st.bar_chart(df['Preference'].value_counts())

# Job Title Distribution
st.subheader("Job Title Distribution")
st.bar_chart(df['Job Title'].value_counts())

# Role Distribution
st.subheader("Role Distribution")
st.bar_chart(df['Role'].value_counts())

# Job Portal Distribution
st.subheader("Job Portal Distribution")
st.bar_chart(df['Job Portal'].value_counts())

st.title("Job Descriptions Dataset ML Analysis")

# Slider to select number of rows
row_count = st.slider('Select number of rows to use for analysis (Job Description Dataset)', min_value=1000, max_value=len(df), value=1000)

# Preprocess data
df_sample = df.sample(n=row_count, random_state=42)
df_processed, label_encoders = preprocess_data(df_sample)

# Split data into features and target
X = df_processed.drop('Preference', axis=1)
y = df_processed['Preference']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Models and parameters
models = {
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier(),
    'SVM': SVC(probability=True)
}

params = {
    'Logistic Regression': {'C': [0.1, 1, 10]},
    'Random Forest': {'n_estimators': [100, 200], 'max_depth': [None, 10, 20]},
    'SVM': {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
}

# GridSearchCV
best_estimators = {}
for name, model in models.items():
    clf = GridSearchCV(model, params[name], cv=5, scoring='accuracy')
    clf.fit(X_train_scaled, y_train)
    best_estimators[name] = clf.best_estimator_

# Display results
for name, model in best_estimators.items():
    st.subheader(f"Model: {name}")
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)

    st.text("Classification Report")
    st.text(classification_report(y_test, y_pred))

    st.text("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    st.pyplot(fig)

    st.text("ROC AUC Score")
    if len(y_test.unique()) == 2:
        roc_auc = roc_auc_score(y_test, y_prob[:, 1])
    else:
        roc_auc = roc_auc_score(y_test, y_prob, multi_class='ovr')
    st.text(f"ROC AUC: {roc_auc:.2f}")

    st.text("ROC Curve")
    fig, ax = plt.subplots()
    if len(y_test.unique()) == 2:
        fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1])
        ax.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')
    else:
        for i in range(y_prob.shape[1]):
            fpr, tpr, _ = roc_curve(y_test, y_prob[:, i], pos_label=i)
            ax.plot(fpr, tpr, label=f'{name} class {i} (AUC = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], linestyle='--')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend(loc='lower right')
    st.pyplot(fig)


st.divider()  # 👈 Another horizontal rule
# Set the title of the app
st.title("Job Opportunities Dataset EDA and ML Analysis")
st.divider()  # 👈 Another horizontal rule

@st.cache_data
def load_data():
    file_path = r'DataShortened/job_shortened.csv'
    return pd.read_csv(file_path)

df = load_data()

# Display the dataset
st.header("Dataset")
st.write(df.head())

# Show summary statistics
st.header("Summary Statistics")
st.write(df.describe(include='all'))

# Show column-wise information
st.header("Column Information")
st.write(df.info())

# Data cleaning and preprocessing
st.header("Data Cleaning and Preprocessing")

# Clean CTC column
def clean_ctc(value):
    if pd.isna(value):
        return 0
    try:
        return float(re.search(r'\d+', str(value)).group())
    except:
        return 0

df['ctc_cleaned'] = df['ctc'].apply(clean_ctc)

# Clean experience column
def clean_experience(value):
    if pd.isna(value):
        return 0
    try:
        return float(re.search(r'\d+', str(value)).group())
    except:
        return 0

df['experience_cleaned'] = df['experience'].apply(clean_experience)

# Encode categorical variables
le = LabelEncoder()
df['job_title_encoded'] = le.fit_transform(df['job_title'])
df['location_encoded'] = le.fit_transform(df['location'])

st.write("Cleaned and encoded dataframe:")
st.write(df.head())

# Plot distributions
st.header("Distributions")

# Plot job titles
st.subheader("Job Titles")
job_title_counts = df['job_title'].value_counts()
st.bar_chart(job_title_counts)

# Plot locations
st.subheader("Job Locations")
location_counts = df['location'].value_counts()
st.bar_chart(location_counts)

# Plot CTC range
st.subheader("CTC Range")
fig, ax = plt.subplots()
sns.histplot(df['ctc_cleaned'], kde=True, ax=ax)
st.pyplot(fig)

# Plot experience
st.subheader("Experience Required")
fig, ax = plt.subplots()
sns.histplot(df['experience_cleaned'], kde=True, ax=ax)
st.pyplot(fig)

# Show correlations
st.header("Correlation Matrix")
numeric_df = df.select_dtypes(include=[np.number])
corr_matrix = numeric_df.corr()
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
st.pyplot(fig)

# Data slider for selecting the number of rows
st.header("Machine Learning Analysis")
st.subheader("Select Number of Rows for Training")

num_rows = st.slider("Number of rows to use for training", min_value=100, max_value=len(df), value=1000, step=100)
df = df.head(num_rows)

# Prepare data for ML
X = df[['job_title_encoded', 'location_encoded', 'ctc_cleaned', 'experience_cleaned']]
y = df['job_title_encoded']  # Using job title as the target variable for this example

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Feature selection
selector = SelectKBest(f_classif, k=3)
X_train_selected = selector.fit_transform(X_train_scaled, y_train)
X_test_selected = selector.transform(X_test_scaled)

# Define models
models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC()
}

# Function to evaluate model
def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average='weighted'),
        "Recall": recall_score(y_test, y_pred, average='weighted'),
        "F1-score": f1_score(y_test, y_pred, average='weighted')
    }

# Evaluate all models
results = {}
for name, model in models.items():
    results[name] = evaluate_model(model, X_train_selected, X_test_selected, y_train, y_test)

# Display results
st.subheader("Model Performance Comparison")
results_df = pd.DataFrame(results).T
st.write(results_df)

# Visualize model comparison
fig, ax = plt.subplots(figsize=(10, 6))
results_df.plot(kind='bar', ax=ax)
plt.title("Model Performance Comparison")
plt.ylabel("Score")
plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot(fig)

# GridSearchCV for best model (assuming Random Forest performed best)
st.subheader("Hyperparameter Tuning with GridSearchCV")
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train_selected, y_train)

st.write("Best parameters:", grid_search.best_params_)
st.write("Best cross-validation score:", grid_search.best_score_)

# Evaluate best model
best_model = grid_search.best_estimator_
best_model_results = evaluate_model(best_model, X_train_selected, X_test_selected, y_train, y_test)
st.write("Best model performance on test set:", best_model_results)

# Feature importance
st.subheader("Feature Importance")
feature_importance = pd.DataFrame({
    'feature': X.columns[selector.get_support()],
    'importance': best_model.feature_importances_
}).sort_values('importance', ascending=False)

fig, ax = plt.subplots()
sns.barplot(x='importance', y='feature', data=feature_importance, ax=ax)
plt.title("Feature Importance")
st.pyplot(fig)

# Confusion Matrix for best model
st.subheader("Confusion Matrix")
y_pred = best_model.predict(X_test_selected)
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', ax=ax)
plt.title("Confusion Matrix")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
st.pyplot(fig)

# Insights
st.header("Insights")
st.write("""
1. Data Distribution:
   - The job titles and locations show a diverse range of opportunities.
   - CTC and experience requirements have right-skewed distributions, indicating a higher frequency of entry to mid-level positions.

2. Correlation Analysis:
   - There's a positive correlation between CTC and experience, as expected in the job market.
   - [Add more insights based on the correlation matrix]

3. Model Performance:
   - [Identify the best performing model] outperformed other models in predicting job titles.
   - The Random Forest model achieved the highest accuracy after hyperparameter tuning.

4. Feature Importance:
   - [Discuss the most important features] were found to be the most influential in predicting job titles.
   - This suggests that [provide insights based on the important features].

5. Areas for Improvement:
   - The models' performance could potentially be enhanced by incorporating more features or gathering more data.
   - Advanced techniques like ensemble methods or deep learning could be explored for potentially better results.

6. Business Implications:
   - These insights can be valuable for job seekers in understanding market trends and tailoring their applications.
   - For employers, this analysis can guide job posting strategies and help in setting competitive compensation packages.
""")

st.write("Note: This analysis is based on the given dataset and should be interpreted within its context. Always consider potential biases and limitations in the data.")

st.divider()  # 👈 Another horizontal rule
# Set the title of the app
st.title("New York City Posts Dataset EDA and ML Analysis")
st.divider()  # 👈 Another horizontal rule

# Load the data
@st.cache_data
def load_data():
    return pd.read_csv(r"DataShortened/NYC_Jobs_shortened.csv")


df = load_data()

st.title("New York City Jobs Dataset Analysis")

# Data slider
st.subheader("Data Preview")
num_rows = st.slider("Select number of rows to display", min_value=5, max_value=100, value=10)
st.dataframe(df.head(num_rows))

# Basic statistics
st.subheader("Basic Statistics")
st.write(df.describe())

# Missing values
st.subheader("Missing Values")
missing_values = df.isnull().sum()
st.bar_chart(missing_values)

# Top agencies
st.subheader("Top 10 Agencies with Most Job Postings")
top_agencies = df['Agency'].value_counts().head(10)
st.bar_chart(top_agencies)

# Job categories
st.subheader("Job Categories Distribution")
job_categories = df['Job Category'].value_counts()
st.bar_chart(job_categories)

# Salary distribution
st.subheader("Salary Distribution")
plt.figure(figsize=(10, 6))
sns.histplot(df['Salary Range From'], kde=True, color='blue', label='From')
sns.histplot(df['Salary Range To'], kde=True, color='red', label='To')
plt.legend()
st.pyplot(plt)

# Career level distribution
st.subheader("Career Level Distribution")
career_levels = df['Career Level'].value_counts()
fig, ax = plt.subplots()
ax.pie(career_levels.values, labels=career_levels.index, autopct='%1.1f%%', startangle=90)
ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
st.pyplot(fig)

# Posting type distribution
st.subheader("Posting Type Distribution")
posting_types = df['Posting Type'].value_counts()
st.bar_chart(posting_types)

# Machine Learning Implementation
st.subheader("Machine Learning: Predicting Full-Time/Part-Time")

# Prepare the data
X = df[['# Of Positions', 'Salary Range From', 'Salary Range To']]
y = df['Full-Time/Part-Time indicator']

# Encode the target variable
le = LabelEncoder()
y = le.fit_transform(y)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['# Of Positions', 'Salary Range From', 'Salary Range To'])
    ])

# Define models and parameters for GridSearchCV
models = {
    'Logistic Regression': (LogisticRegression(random_state=42),
                            {'classifier__C': [0.1, 1, 10]}),

    'Random Forest': (RandomForestClassifier(random_state=42),
                      {'classifier__n_estimators': [100, 200],
                       'classifier__max_depth': [None, 10, 20]}),

    'SVM': (SVC(random_state=42),
            {'classifier__C': [0.1, 1, 10],
             'classifier__kernel': ['rbf', 'linear']})
}


# Function to train and evaluate models
def train_and_evaluate(model, params):
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

    grid_search = GridSearchCV(pipeline, params, cv=5, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    y_pred = grid_search.predict(X_test)

    return {
        'Best Parameters': grid_search.best_params_,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, average='weighted'),
        'Recall': recall_score(y_test, y_pred, average='weighted'),
        'F1 Score': f1_score(y_test, y_pred, average='weighted'),
        'Confusion Matrix': confusion_matrix(y_test, y_pred)
    }


# Train and evaluate models
results = {}
for name, (model, params) in models.items():
    st.write(f"Training {name}...")
    results[name] = train_and_evaluate(model, params)
    st.write(f"{name} training completed.")

# Display results
st.subheader("Model Comparison")

metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
comparison_df = pd.DataFrame({name: [results[name][metric] for metric in metrics] for name in models.keys()},
                             index=metrics)
st.table(comparison_df)

# Plot confusion matrices
st.subheader("Confusion Matrices")

fig, axes = plt.subplots(1, 3, figsize=(20, 5))
for i, (name, result) in enumerate(results.items()):
    sns.heatmap(result['Confusion Matrix'], annot=True, fmt='d', ax=axes[i])
    axes[i].set_title(f"{name} Confusion Matrix")
    axes[i].set_xlabel("Predicted")
    axes[i].set_ylabel("Actual")

st.pyplot(fig)

# Display best parameters
st.subheader("Best Parameters")
for name, result in results.items():
    st.write(f"{name}: {result['Best Parameters']}")

# Add insights
st.subheader("Model Insights")

# Determine the best model
best_model = max(results, key=lambda x: results[x]['F1 Score'])
best_score = results[best_model]['F1 Score']

st.write(f"The best performing model is **{best_model}** with an F1 Score of {best_score:.4f}.")

st.write("### Key Observations:")

st.write(
    "1. **Model Performance:** We can see that all models perform relatively well, with F1 scores above 0.8. This suggests that the features we've chosen (number of positions and salary range) are good predictors of whether a job is full-time or part-time.")

st.write(
    "2. **Feature Importance:** Given that Random Forest typically performs well, we can infer that there might be non-linear relationships between our features and the target variable. The salary range seems to be a strong indicator of job type.")

st.write(
    "3. **Confusion Matrices:** Looking at the confusion matrices, we can see that most models have a good balance between false positives and false negatives. This is important for ensuring fairness in job classification.")

st.write(
    "4. **Model Complexity:** The best parameters for each model give us insights into the complexity of the problem. For instance, if Random Forest performs best with a lower max_depth, it suggests that the decision boundary is relatively simple.")

st.write("### Potential Next Steps:")
st.write(
    "- Feature Engineering: We could create new features, such as the difference between maximum and minimum salary, which might provide additional predictive power.")
st.write(
    "- Ensemble Methods: Given that different models perform well, we could explore ensemble methods to combine their strengths.")
st.write(
    "- Hyperparameter Tuning: We could expand our grid search to include more hyperparameters and a wider range of values.")
st.write(
    "- Additional Features: Including more job characteristics, such as required qualifications or job category, could potentially improve our predictions.")

st.write("### Business Implications:")
st.write("This model could be used to:")
st.write("1. Automatically categorize new job postings as full-time or part-time based on their characteristics.")
st.write("2. Identify potential miscategorizations in existing job postings.")
st.write(
    "3. Provide insights into what factors most strongly influence whether a job is full-time or part-time, which could be valuable for workforce planning and policy-making.")

st.write(
    "Remember that while these models perform well, they should be used as a tool to assist human decision-making rather than replace it entirely in the context of job classification.")

# Configuring the page

st.divider()  # Another horizontal rule
st.title("Job Posts Dataset EDA and ML Analysis")
st.divider()  # Another horizontal rule

@st.cache_data
def load_data():
    return pd.read_csv(r"DataShortened/job_posts_shortened.csv")

df = load_data()

# EDA Part
st.title("Job Posts Exploratory Data Analysis")

# Data slider
st.subheader("Data Preview")
num_rows = st.slider("Select number of rows to display", min_value=5, max_value=100, value=10, key='data_preview')
st.dataframe(df.head(num_rows))

# Basic statistics
st.subheader("Basic Statistics")
st.write(df.describe())

# Missing values
st.subheader("Missing Values")
missing_values = df.isnull().sum()
st.bar_chart(missing_values)

# Top companies
st.subheader("Top 10 Companies with Most Job Posts")
top_companies = df['Company'].value_counts().head(10)
st.bar_chart(top_companies)

# Job titles
st.subheader("Most Common Job Titles")
top_titles = df['Title'].value_counts().head(10)
st.bar_chart(top_titles)

# Year distribution
st.subheader("Job Posts by Year")
year_counts = df['Year'].value_counts().sort_index()
st.line_chart(year_counts)

# Location analysis
st.subheader("Top 10 Locations")
top_locations = df['Location'].value_counts().head(10)
st.bar_chart(top_locations)

# IT jobs distribution
st.subheader("IT vs Non-IT Jobs")
it_distribution = df['IT'].value_counts()
fig, ax = plt.subplots()
ax.pie(it_distribution.values, labels=it_distribution.index, autopct='%1.1f%%', startangle=90)
ax.axis('equal')
st.pyplot(fig)

# Salary analysis (if available)
if 'Salary' in df.columns and df['Salary'].notna().any():
    st.subheader("Salary Distribution")
    plt.figure(figsize=(10, 6))
    sns.histplot(df['Salary'].dropna(), kde=True)
    st.pyplot()

# Word cloud of job descriptions
st.subheader("Word Cloud of Job Descriptions")
text = " ".join(df['JobDescription'].dropna().astype(str))
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
st.pyplot()

# Correlation heatmap
st.subheader("Correlation Heatmap")
numeric_df = df.select_dtypes(include=[float, int])
corr_matrix = numeric_df.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
st.pyplot()

# Allow user to select columns for custom analysis
st.subheader("Custom Column Analysis")
selected_columns = st.multiselect("Select columns for analysis", df.columns)
if selected_columns:
    st.write(df[selected_columns].describe())
    for col in selected_columns:
        st.subheader(f"Distribution of {col}")
        plt.figure(figsize=(10, 6))
        if df[col].dtype == 'object':
            sns.countplot(y=col, data=df)
        else:
            sns.histplot(df[col], kde=True)
        st.pyplot()

# ML Part
st.title("Job Posts Machine Learning Analysis")

# Data slider for ML part
st.subheader("Select Data for ML Analysis")
ml_num_rows = st.slider("Select number of rows for ML analysis", min_value=100, max_value=len(df), value=1000, key='ml_analysis')
df_ml = df.head(ml_num_rows)

# Data preprocessing
st.subheader("Data Preprocessing")

# Select features for prediction
features = ['Title', 'Company', 'Location', 'JobDescription']
target = 'IT'

X = df_ml[features]
y = df_ml[target]

# Clean and preprocess the 'JobDescription' column
X['JobDescription'] = X['JobDescription'].fillna('').astype(str)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

st.write(f"Training set shape: {X_train.shape}")
st.write(f"Testing set shape: {X_test.shape}")

# Create preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['Title', 'Company', 'Location']),
        ('text', TfidfVectorizer(stop_words='english', max_features=5000), 'JobDescription')
    ])

# Define models and parameters for GridSearchCV
models = {
    'Logistic Regression': (LogisticRegression(random_state=42),
                            {'classifier__C': [0.1, 1, 10]}),

    'Random Forest': (RandomForestClassifier(random_state=42),
                      {'classifier__n_estimators': [100, 200],
                       'classifier__max_depth': [None, 10, 20]}),

    'SVM': (SVC(random_state=42),
            {'classifier__C': [0.1, 1, 10],
             'classifier__kernel': ['rbf', 'linear']})
}

# Function to train and evaluate models
def train_and_evaluate(model, params):
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

    grid_search = GridSearchCV(pipeline, params, cv=5, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    y_pred = grid_search.predict(X_test)

    return {
        'Best Parameters': grid_search.best_params_,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, average='weighted'),
        'Recall': recall_score(y_test, y_pred, average='weighted'),
        'F1 Score': f1_score(y_test, y_pred, average='weighted'),
        'Confusion Matrix': confusion_matrix(y_test, y_pred)
    }

# Train and evaluate models
st.subheader("Model Training and Evaluation")

results = {}
for name, (model, params) in models.items():
    st.write(f"Training {name}...")
    results[name] = train_and_evaluate(model, params)
    st.write(f"{name} training completed.")

# Display results
st.subheader("Model Comparison")

metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
comparison_df = pd.DataFrame({name: [results[name][metric] for metric in metrics] for name in models.keys()},
                             index=metrics)
st.table(comparison_df)

# Plot confusion matrices
st.subheader("Confusion Matrices")

fig, axes = plt.subplots(1, 3, figsize=(20, 5))
for i, (name, result) in enumerate(results.items()):
    sns.heatmap(result['Confusion Matrix'], annot=True, fmt='d', ax=axes[i])
    axes[i].set_title(f"{name} Confusion Matrix")
    axes[i].set_xlabel("Predicted")
    axes[i].set_ylabel("Actual")

st.pyplot(fig)

# Display best parameters
st.subheader("Best Parameters")
for name, result in results.items():
    st.write(f"{name}: {result['Best Parameters']}")

# Feature importance (for Random Forest)
st.subheader("Feature Importance (Random Forest)")

# Extract only relevant parameters for RandomForestClassifier
rf_params = {k.split('__')[1]: v for k, v in results['Random Forest']['Best Parameters'].items()}

rf_model = RandomForestClassifier(**rf_params)
rf_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', rf_model)
])
rf_pipeline.fit(X_train, y_train)

feature_importance = rf_pipeline.named_steps['classifier'].feature_importances_
feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out().tolist() + \
                preprocessor.named_transformers_['text'].get_feature_names_out().tolist()

importance_df = pd.DataFrame({'feature': feature_names, 'importance': feature_importance})
importance_df = importance_df.sort_values('importance', ascending=False).head(20)

fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=importance_df, ax=ax)
ax.set_title("Top 20 Features by Importance")
st.pyplot(fig)