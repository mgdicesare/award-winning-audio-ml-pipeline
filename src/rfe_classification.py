import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

train_data = pd.read_csv('TableTrain_v2.csv')
test_data = pd.read_csv('TableTest_v2.csv')
validation_data = pd.read_csv('TableValid_v2.csv')

X_train = train_data.iloc[:, 14:27]  # GTCC [:, 1:14] e DeltaGTCC [:, 14:27]
y_train = train_data.iloc[:, 0]     # Labels
X_test = test_data.iloc[:, 14:27]    # GTCC [:, 1:14] e DeltaGTCC [:, 14:27]
y_test = test_data.iloc[:, 0]       # Labels
X_val = validation_data.iloc[:, 14:27]  # GTCC [:, 1:14] e DeltaGTCC [:, 14:27]
y_val = validation_data.iloc[:, 0]     

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_val_scaled = scaler.transform(X_val) 

model = LogisticRegression(random_state=42)

rfe = RFE(model, n_features_to_select=5)  # Specifica il numero  di features

X_train_rfe = rfe.fit_transform(X_train_scaled, y_train)

model.fit(X_train_rfe, y_train)

X_test_rfe = rfe.transform(X_test_scaled)
X_val_rfe = rfe.transform(X_val_scaled)

selected_features = X_train.columns[rfe.support_]
print('Selected Features:')
print(selected_features)

y_pred_test = model.predict(X_test_rfe)

accuracy_test = accuracy_score(y_test, y_pred_test)
precision_test = precision_score(y_test, y_pred_test)
recall_test = recall_score(y_test, y_pred_test)
f1_test = f1_score(y_test, y_pred_test)

print('Performance on Test Data:')
print(f'Accuracy: {accuracy_test}')
print(f'Precision: {precision_test}')
print(f'Recall: {recall_test}')
print(f'F1-score: {f1_test}')

y_pred_val = model.predict(X_val_rfe)

accuracy_val = accuracy_score(y_val, y_pred_val)
precision_val = precision_score(y_val, y_pred_val)
recall_val = recall_score(y_val, y_pred_val)
f1_val = f1_score(y_val, y_pred_val)

print('\nPerformance on Validation Data:')
print(f'Accuracy: {accuracy_val}')
print(f'Precision: {precision_val}')
print(f'Recall: {recall_val}')
print(f'F1-score: {f1_val}')

cm_test = confusion_matrix(y_test, y_pred_test)
print('\nConfusion Matrix on Test Data:')
print(cm_test)

cm_val = confusion_matrix(y_val, y_pred_val)
print('\nConfusion Matrix on Validation Data:')
print(cm_val)

plt.figure(figsize=(8, 6))
sns.heatmap(cm_test, annot=True, cmap='Blues', fmt='g', cbar=False)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix on Test Data')
plt.show()

plt.figure(figsize=(8, 6))
sns.heatmap(cm_val, annot=True, cmap='Blues', fmt='g', cbar=False)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix on Validation Data')
plt.show()
