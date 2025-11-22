import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score, classification_report
import warnings

warnings.filterwarnings('ignore', category=UserWarning)

# --- 1. CONFIGURATION ---
DATA_FILE = 'iris.data.csv' # Your specified local file name
TARGET_COLUMN = 'species'
RANDOM_SEED = 42

# Define the column names as the raw CSV usually lacks headers
COLUMN_NAMES = [
    'sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species'
]

# --- 2. DATA LOADING ---
try:
    # Load the CSV. header=None tells pandas the file has no header row.
    df = pd.read_csv(DATA_FILE, header=None)
    df.columns = COLUMN_NAMES # Assign the manual column names
    
except FileNotFoundError:
    print(f"\n--- ERROR ---")
    print(f"Error: The file '{DATA_FILE}' was not found in the directory.")
    print("Please ensure the CSV file is in the same directory as this script.")
    print("--- EXITING ---")
    exit()

# Separate features (X) and target (y)
X = df.drop(columns=[TARGET_COLUMN])
y = df[TARGET_COLUMN]

# Get the unique target names for the final report
target_names = y.unique()

print("\n--- Iris Dataset Info ---")
print(f"Dataset Shape: {df.shape}")
print(f"Target Species: {target_names}")
print("Features:\n", X.head())


# --- 3. PREPROCESSING AND SPLIT ---
# Split the data (80% training, 20% testing). Stratify ensures balanced species distribution.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y 
)

# Scale the numerical features (essential for Logistic Regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n--- Data Split Complete ---")
print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")


# --- 4. MODEL TRAINING ---
# Use Logistic Regression for classification
# 'C' parameter is adjusted for slightly better performance
log_reg = LogisticRegression(random_state=RANDOM_SEED, solver='liblinear', C=10) 
log_reg.fit(X_train_scaled, y_train)

print("\n--- Model Training Complete (Logistic Regression) ---")

# --- 5. EVALUATION ---
# Make predictions on the test set
y_pred = log_reg.predict(X_test_scaled)

# Calculate Accuracy Score
accuracy = accuracy_score(y_test, y_pred)

print("\n--- Model Evaluation (Accuracy) ---")
print(f"Model Used: Logistic Regression")
print(f"Accuracy Score: {accuracy * 100:.2f}%")

# Generate a detailed classification report
print("\nClassification Report (Precision, Recall, F1-Score):")
print(classification_report(y_test, y_pred, target_names=target_names))