import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import joblib
import json
import math

# Load the dataset
# Note: You'll need to download the dataset from the provided link
# and place it in the same directory as this script
df = pd.read_csv('weather.csv')

# Prepare features and target
X = df[['Temperature', 'Humidity', 'Pressure', 'Wind Speed', 'Wind Direction']]
y = df['Rainfall']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize models
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'SVM': SVC(probability=True, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=3)
}

# Dictionary to store results
results = {}

# Train and evaluate each model
for name, model in models.items():
    print(f"\nTraining {name}...")
    
    # Train the model
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
    conf_matrix = confusion_matrix(y_test, y_pred).tolist()
    
    # Calculate ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    # Calculate learning curve
    train_sizes, train_scores, test_scores = learning_curve(
        model, X_train_scaled, y_train, cv=5, n_jobs=-1, 
        train_sizes=np.linspace(0.1, 1.0, 10)
    )
    
    # Calculate feature importance if available
    feature_importance = None
    if hasattr(model, 'feature_importances_'):
        feature_importance = model.feature_importances_.tolist()
    elif hasattr(model, 'coef_'):
        feature_importance = np.abs(model.coef_[0]).tolist()
    
    # Store results
    results[name] = {
        'model': model,
        'accuracy': accuracy,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'confusion_matrix': conf_matrix,
        'predictions': y_pred.tolist(),
        'probabilities': y_pred_proba.tolist(),
        'true_values': y_test.tolist(),
        'roc_curve': {
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'auc': float(roc_auc)
        },
        'learning_curve': {
            'train_sizes': train_sizes.tolist(),
            'train_scores_mean': np.mean(train_scores, axis=1).tolist(),
            'train_scores_std': np.std(train_scores, axis=1).tolist(),
            'test_scores_mean': np.mean(test_scores, axis=1).tolist(),
            'test_scores_std': np.std(test_scores, axis=1).tolist()
        },
        'feature_importance': feature_importance
    }
    
    # Print results
    print(f"{name} Results:")
    print(f"Test Accuracy: {accuracy:.3f}")
    print(f"Cross-validation scores: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

# Find the best performing model
best_model_name = max(results, key=lambda x: results[x]['accuracy'])
best_model = results[best_model_name]['model']

print(f"\nBest performing model: {best_model_name}")
print(f"Best accuracy: {results[best_model_name]['accuracy']:.3f}")

# Save the best model and scaler
joblib.dump(best_model, 'rainfall_model.joblib')
joblib.dump(scaler, 'scaler.joblib')

# Save all models for comparison
joblib.dump(models, 'all_models.joblib')

# Save results for visualization
with open('model_results.json', 'w') as f:
    def clean_nans(obj):
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return None
        elif isinstance(obj, list):
            return [clean_nans(x) for x in obj]
        elif isinstance(obj, dict):
            return {k: clean_nans(v) for k, v in obj.items()}
        else:
            return obj
    json.dump(clean_nans({
        name: {
            'accuracy': float(results[name]['accuracy']),
            'cv_mean': float(results[name]['cv_mean']),
            'cv_std': float(results[name]['cv_std']),
            'confusion_matrix': results[name]['confusion_matrix'],
            'predictions': results[name]['predictions'],
            'probabilities': results[name]['probabilities'],
            'true_values': results[name]['true_values'],
            'roc_curve': results[name]['roc_curve'],
            'learning_curve': results[name]['learning_curve'],
            'feature_importance': results[name]['feature_importance']
        }
        for name in results
    }), f, indent=4, allow_nan=False) 