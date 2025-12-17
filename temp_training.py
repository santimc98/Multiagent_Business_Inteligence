import pandas as pd
import numpy as np
import re
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import xgboost as xgb

def fuzzy_find_column(df, candidate_name):
    """
    Finds a column in a DataFrame using a fuzzy, normalized matching approach.
    """
    def clean(s):
        return re.sub(r'[^a-z0-9]', '', str(s).lower())
    
    clean_candidate = clean(candidate_name)
    
    for col in df.columns:
        if clean(col) == clean_candidate:
            return col
    return None

def main():
    """
    Main function to execute the ML pipeline.
    """
    # --- Configuration ---
    DATA_FILE_PATH = 'data/cleaned_data.csv'
    PLOTS_DIR = 'static/plots/'
    MODEL_ID = 'icp' # Ideal Customer Profile

    # --- Ensure plot directory exists ---
    try:
        os.makedirs(PLOTS_DIR, exist_ok=True)
    except OSError as e:
        print(f"Error creating directory {PLOTS_DIR}: {e}")
        return

    # --- [1] Data Loading & Interpretation ---
    print("[1] Loading and interpreting data...")
    try:
        df = pd.read_csv(DATA_FILE_PATH, encoding='utf-8')
        print(f"Successfully loaded data from '{DATA_FILE_PATH}'. Shape: {df.shape}")
    except FileNotFoundError:
        print(f"Error: Data file not found at '{DATA_FILE_PATH}'. Please check the path.")
        return
    except Exception as e:
        print(f"An error occurred while loading the data: {e}")
        return

    # Map semantic target to actual column
    target_col_semantic = 'CurrentPhase'
    target_col_actual = fuzzy_find_column(df, target_col_semantic)

    if not target_col_actual:
        print(f"FATAL: Target column '{target_col_semantic}' not found in the dataset. Aborting.")
        return
    
    print(f"Mapped Strategy Target '{target_col_semantic}' -> Found in CSV as '{target_col_actual}'")

    # Create binary target variable: 1 for 'Contract', 0 for 'No Contract'
    # Assuming 'Contract' is the positive class
    df['target'] = np.where(df[target_col_actual] == 'Contract', 1, 0)
    
    print("Target variable created. Distribution:")
    print(df['target'].value_counts(normalize=True).rename({1: 'Contract', 0: 'No Contract'}))
    
    # --- [2] Feature Engineering & Selection ---
    print("\n[2] Performing feature engineering and selection...")
    
    suggested_features = [
        'Country', 'Segment', 'Size', 'SizeRank', 'Sector', 'ERP', 
        'Debtors', 'DebtorsRank', '1stYearAmount', 'AnnualSubscriptionFee', 
        'ConsultancyFee', 'Insurance'
    ]
    
    features_to_use = []
    for feature in suggested_features:
        actual_col = fuzzy_find_column(df, feature)
        if actual_col:
            features_to_use.append(actual_col)
            print(f"  - Mapped feature '{feature}' -> Found as '{actual_col}'")
        else:
            print(f"  - WARNING: Suggested feature '{feature}' not found in dataset. It will be skipped.")

    # Define feature matrix (X) and target vector (y)
    X = df[features_to_use]
    y = df['target']

    # Identify numerical and categorical features
    numerical_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

    print(f"\nIdentified {len(numerical_features)} numerical features: {numerical_features}")
    print(f"Identified {len(categorical_features)} categorical features: {categorical_features}")

    # --- [3] Model Selection, Training, and Validation ---
    print("\n[3] Selecting, training, and validating the model...")

    # Split data into training and testing sets, stratifying by the target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Data split into training ({len(X_train)} rows) and testing ({len(X_test)} rows) sets.")

    # Create preprocessing pipelines for numerical and categorical features
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'
    )

    # Address class imbalance using scale_pos_weight for XGBoost
    scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]

    # Define the model pipeline
    # FIX: Removed deprecated 'use_label_encoder' parameter.
    # FIX: Added 'scale_pos_weight' to handle class imbalance.
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        scale_pos_weight=scale_pos_weight,
        random_state=42
    )

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

    # Train the model
    print("Training the XGBoost pipeline...")
    pipeline.fit(X_train, y_train)
    print("Training complete.")

    # --- [4] Evaluation & Visualization ---
    print("\n[4] Evaluating model performance and generating insights...")

    # Make predictions
    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

    # Print classification report
    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred, target_names=['No Contract', 'Contract']))
    
    # Print ROC AUC Score
    auc_score = roc_auc_score(y_test, y_pred_proba)
    print(f"ROC AUC Score: {auc_score:.4f}\n")

    print(f"Plots will be saved to '{PLOTS_DIR}'")

    # Plot 1: Confusion Matrix
    try:
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['No Contract', 'Contract'], 
                    yticklabels=['No Contract', 'Contract'])
        plt.title('Confusion Matrix for ICP Prediction')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        cm_path = os.path.join(PLOTS_DIR, f'{MODEL_ID}_confusion_matrix.png')
        plt.savefig(cm_path)
        plt.close()
        print(f"  - Saved Confusion Matrix to '{cm_path}'")
    except Exception as e:
        print(f"  - Error generating Confusion Matrix plot: {e}")

    # Plot 2: Feature Importance
    try:
        # Extract feature names after one-hot encoding
        ohe_feature_names = pipeline.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(categorical_features)
        all_feature_names = numerical_features + list(ohe_feature_names)
        
        importances = pipeline.named_steps['classifier'].feature_importances_
        
        feature_importance_df = pd.DataFrame({
            'feature': all_feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False).head(15)

        plt.figure(figsize=(10, 8))
        # FIX: Updated sns.barplot call to avoid FutureWarning
        sns.barplot(x='importance', y='feature', data=feature_importance_df, hue='feature', palette='viridis', dodge=False, legend=False)
        plt.title('Top 15 Feature Importances for ICP Model')
        plt.xlabel('Importance Score')
        plt.ylabel('Feature')
        plt.tight_layout()
        fi_path = os.path.join(PLOTS_DIR, f'{MODEL_ID}_feature_importance.png')
        plt.savefig(fi_path)
        plt.close()
        print(f"  - Saved Feature Importance plot to '{fi_path}'")
    except Exception as e:
        print(f"  - Error generating Feature Importance plot: {e}")

    # Plot 3: ROC Curve
    try:
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auc_score:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.tight_layout()
        roc_path = os.path.join(PLOTS_DIR, f'{MODEL_ID}_roc_curve.png')
        plt.savefig(roc_path)
        plt.close()
        print(f"  - Saved ROC Curve plot to '{roc_path}'")
    except Exception as e:
        print(f"  - Error generating ROC Curve plot: {e}")

    print("\n--- Mission Complete ---")

if __name__ == "__main__":
    main()