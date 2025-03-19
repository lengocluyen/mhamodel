import pandas as pd
import numpy as np
import sys
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import time
import psutil
import torch
import pickle
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from deepctr_torch.callbacks import EarlyStopping
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names
from deepctr_torch.models.basemodel import BaseModel
from deepctr_torch.layers import PredictionLayer
from deepctr_torch.models import (
    AFM,
    AutoInt,
    CCPM,
    DCN,
    DeepFM,
    DIFM,
    FiBiNET,
    NFM,
    ONN,
    PNN,
    WDL,
    xDeepFM,
    IFM,
    MLR
)
from mha_model import *

# ----------------------------
# Step 1: Load and Merge Data
# ----------------------------

# Load Data
users = pd.read_csv("./ITM-Rec/users.csv")
items = pd.read_csv("./ITM-Rec/items.csv")
df_ratings = pd.read_csv("./ITM-Rec/ratings.csv")

# Load group data
group_df = pd.read_csv("./ITM-Rec/group.csv")
group_size_df = pd.read_csv("./ITM-Rec/group_size.csv")
group_ratings = pd.read_csv("./ITM-Rec/group_ratings.csv")

# Rename columns for clarity
group_ratings.columns = ['GroupID', "Item", 'GroupRating', 'GroupApp', 'GroupData', 'GroupEase', 'GroupClass', 'GroupSemester', 'GroupLockdown']

# Initial shapes
print("Shape of df_ratings before merge:", df_ratings.shape)
print("Shape of group_df:", group_df.shape)
print("Shape of group_size_df:", group_size_df.shape)
print("Shape of group_ratings:", group_ratings.shape)

# Select relevant columns from individual ratings
df_ratings = df_ratings[["UserID", "Item", "Class", "Semester", "Lockdown", "App", "Data", "Ease", "Rating"]]

# Merge with group information
df_ratings = pd.merge(df_ratings, group_df, on='UserID', how='left')

# Merge with group size
df_ratings = pd.merge(df_ratings, group_size_df, on='GroupID', how='left')

# Merge with group ratings
df_ratings = pd.merge(df_ratings, group_ratings, on=['GroupID', 'Item'], how='left')

# Fill missing GroupID with UserID (for users not in any group, treat them as single-member groups)
df_ratings['GroupID'] = df_ratings['GroupID'].fillna(df_ratings['UserID'])


# Fill missing group sizes with 1 (single-member groups)
df_ratings['Size'] = df_ratings['Size'].fillna(1)

# Fill missing group ratings with individual ratings
for col in ['Rating', 'App', 'Data', 'Ease', 'Class', 'Semester', 'Lockdown']:
    group_col = f'Group{col}'
    df_ratings[group_col] = df_ratings[group_col].fillna(df_ratings[col])

print("df_rating: ", df_ratings.head())

print("Rows with NaN values in df_ratings:")
print(df_ratings[df_ratings.isnull().any(axis=1)])
df_ratings = df_ratings.dropna()
print("Rows with NaN values in df_ratings:")
print(df_ratings[df_ratings.isnull().any(axis=1)])


print("df_rating", df_ratings.head())
# Check for any remaining missing values
print("\nMissing values after merging:")
print(df_ratings.isnull().sum())

# ----------------------------
# Step 2: Feature Engineering
# ----------------------------

# Initialize LabelEncoders
label_encoders = {}

# Ensure there is at least one argument (excluding script name)
if len(sys.argv) < 2:
    print("Usage: python main.py <case_number>")
    sys.exit(1)  # Exit if no argument is provided

# Get the argument value and convert it to an integer
try:
    case = int(sys.argv[1])  # Convert the first argument to an integer
except ValueError:
    print("Error: case_number must be an integer.")
    sys.exit(1)

# Decision based on case value
if case == 1:
    print("Executing Case 1: Running Model Training with all context and criteria")
    # Define dense features
    #===========================================================
    # test case 1: all context and criteria
    #Criteria ['GroupApp', 'GroupData', 'GroupEase']
    dense_features = ['App', 'Data', 'Ease', 
                    'GroupApp', 'GroupData', 'GroupEase']
    ## Context = ['Class', 'Semester', 'Lockdown']
    # Define categorical features
    sparse_features = ['UserID', 'Item', 'Class', 'Semester', 'Lockdown', 
                    'GroupID', 'GroupClass', 'GroupSemester', 'GroupLockdown']
    #===========================================================
elif case == 2:
    print("Executing Case 2: Running Model Training without all context and criteria")
    # Define dense features
    #===========================================================
    # test case 1: all context and criteria
    #Criteria ['GroupApp', 'GroupData', 'GroupEase']
    dense_features = ['App', 'Data', 'Ease']
    ## Context = ['Class', 'Semester', 'Lockdown'] 'Class', 'Semester', 'Lockdown',
    # Define categorical features
    sparse_features = ['UserID', 'Item', 'GroupID', ]
    #===========================================================
elif case == 3:
    print("Executing Case 3: Running Prediction with criteria, without context")
     #===========================================================
    # test case 1: all context and criteria
    #Criteria ['GroupApp', 'GroupData', 'GroupEase']
    dense_features = ['App', 'Data', 'Ease', 
                    'GroupApp', 'GroupData', 'GroupEase']
    ## Context = ['Class', 'Semester', 'Lockdown']
    # Define categorical features
    sparse_features = ['UserID', 'Item', 'GroupID', 'GroupClass', 'GroupSemester', 'GroupLockdown']
    #===========================================================
elif case == 4:
    print("Executing Case 4: Running Prediction without criteria, with context")
     #===========================================================
    # test case 1: all context and criteria
    #Criteria ['GroupApp', 'GroupData', 'GroupEase']
    dense_features = ['App', 'Data', 'Ease']
    ## Context = ['Class', 'Semester', 'Lockdown']
    # Define categorical features
    sparse_features = ['UserID', 'Item', 'Class', 'Semester', 'Lockdown', 'GroupID', 'GroupClass', 'GroupSemester', 'GroupLockdown']
    #===========================================================
elif case == 5:
    print("Executing Case 5: Running Prediction with criteria, with single context: class")
     #===========================================================
    # test case 1: all context and criteria
    #Criteria ['GroupApp', 'GroupData', 'GroupEase']
    dense_features = ['App', 'Data', 'Ease', 'GroupApp', 'GroupData', 'GroupEase']
    ## Context = ['Class', 'Semester', 'Lockdown']
    # Define categorical features
    sparse_features = ['UserID', 'Item', 'Class', 'GroupID', 'GroupClass', 'GroupSemester', 'GroupLockdown']
    #===========================================================
elif case == 6:
    print("Executing Case 6: Running Prediction with criteria, with single context: semester")
     #===========================================================
    # test case 1: all context and criteria
    #Criteria ['GroupApp', 'GroupData', 'GroupEase']
    dense_features = ['App', 'Data', 'Ease', 'GroupApp', 'GroupData', 'GroupEase']
    ## Context = ['Class', 'Semester', 'Lockdown']
    # Define categorical features
    sparse_features = ['UserID', 'Item', 'Semester', 'GroupID', 'GroupClass', 'GroupSemester', 'GroupLockdown']
    #===========================================================
elif case == 7:
    print("Executing Case 7: Running Prediction with criteria, with single context: Lockdown")
     #===========================================================
    # test case 1: all context and criteria
    #Criteria ['GroupApp', 'GroupData', 'GroupEase']
    dense_features = ['App', 'Data', 'Ease', 'GroupApp', 'GroupData', 'GroupEase']
    ## Context = ['Class', 'Semester', 'Lockdown']
    # Define categorical features
    sparse_features = ['UserID', 'Item', 'Lockdown', 'GroupID', 'GroupClass', 'GroupSemester', 'GroupLockdown']
    #===========================================================
elif case == 8:
    print("Executing Case 8: Running Prediction with single criteria: GroupApp, with single context: class")
     #===========================================================
    # test case 1: all context and criteria
    #Criteria ['GroupApp', 'GroupData', 'GroupEase']
    dense_features = ['App', 'Data', 'Ease', 'GroupApp']
    ## Context = ['Class', 'Semester', 'Lockdown']
    # Define categorical features
    sparse_features = ['UserID', 'Item', 'Class', 'GroupID', 'GroupClass', 'GroupSemester', 'GroupLockdown']
    #===========================================================
elif case == 9:
    print("Executing Case 9: Running Prediction with single criteria: GroupData, with single context: class")
     #===========================================================
    # test case 1: all context and criteria
    #Criteria ['GroupApp', 'GroupData', 'GroupEase']
    dense_features = ['App', 'Data', 'Ease', 'GroupData']
    ## Context = ['Class', 'Semester', 'Lockdown']
    # Define categorical features
    sparse_features = ['UserID', 'Item', 'Class', 'GroupID', 'GroupClass', 'GroupSemester', 'GroupLockdown']
    #===========================================================
elif case == 10:
    print("Executing Case 10: Running Prediction with single criteria: GroupEase, with single context: class")
     #===========================================================
    # test case 1: all context and criteria
    #Criteria ['GroupApp', 'GroupData', 'GroupEase']
    dense_features = ['App', 'Data', 'Ease', 'GroupEase']
    ## Context = ['Class', 'Semester', 'Lockdown']
    # Define categorical features
    sparse_features = ['UserID', 'Item', 'Class', 'GroupID', 'GroupClass', 'GroupSemester', 'GroupLockdown']
    #===========================================================
elif case == 11:
    print("Executing Case 11: Running Prediction with single criteria: GroupApp, with single context: Semester")
     #===========================================================
    # test case 1: all context and criteria
    #Criteria ['GroupApp', 'GroupData', 'GroupEase']
    dense_features = ['App', 'Data', 'Ease', 'GroupApp']
    ## Context = ['Class', 'Semester', 'Lockdown']
    # Define categorical features
    sparse_features = ['UserID', 'Item', 'Semester', 'GroupID', 'GroupClass', 'GroupSemester', 'GroupLockdown']
    #===========================================================
elif case == 12:
    print("Executing Case 12: Running Prediction with single criteria: GroupData, with single context: Semester")
     #===========================================================
    # test case 1: all context and criteria
    #Criteria ['GroupApp', 'GroupData', 'GroupEase']
    dense_features = ['App', 'Data', 'Ease', 'GroupData']
    ## Context = ['Class', 'Semester', 'Lockdown']
    # Define categorical features
    sparse_features = ['UserID', 'Item', 'Semester', 'GroupID', 'GroupClass', 'GroupSemester', 'GroupLockdown']
    #===========================================================
elif case == 13:
    print("Executing Case 13: Running Prediction with single criteria: GroupEase, with single context: Semester")
     #===========================================================
    # test case 1: all context and criteria
    #Criteria ['GroupApp', 'GroupData', 'GroupEase']
    dense_features = ['App', 'Data', 'Ease', 'GroupEase']
    ## Context = ['Class', 'Semester', 'Lockdown']
    # Define categorical features
    sparse_features = ['UserID', 'Item', 'Semester', 'GroupID', 'GroupClass', 'GroupSemester', 'GroupLockdown']
    #===========================================================
elif case == 14:
    print("Executing Case 14: Running Prediction with single criteria: GroupApp, with single context: Lockdown")
     #===========================================================
    # test case 1: all context and criteria
    #Criteria ['GroupApp', 'GroupData', 'GroupEase']
    dense_features = ['App', 'Data', 'Ease', 'GroupApp']
    ## Context = ['Class', 'Semester', 'Lockdown']
    # Define categorical features
    sparse_features = ['UserID', 'Item', 'Lockdown', 'GroupID', 'GroupClass', 'GroupSemester', 'GroupLockdown']
    #===========================================================
elif case == 15:
    print("Executing Case 15: Running Prediction with single criteria: GroupData, with single context: Lockdown")
     #===========================================================
    # test case 1: all context and criteria
    #Criteria ['GroupApp', 'GroupData', 'GroupEase']
    dense_features = ['App', 'Data', 'Ease', 'GroupData']
    ## Context = ['Class', 'Semester', 'Lockdown']
    # Define categorical features
    sparse_features = ['UserID', 'Item', 'Lockdown', 'GroupID', 'GroupClass', 'GroupSemester', 'GroupLockdown']
    #===========================================================
elif case == 16:
    print("Executing Case 16: Running Prediction with single criteria: GroupEase, with single context: Lockdown")
     #===========================================================
    # test case 1: all context and criteria
    #Criteria ['GroupApp', 'GroupData', 'GroupEase']
    dense_features = ['App', 'Data', 'Ease', 'GroupEase']
    ## Context = ['Class', 'Semester', 'Lockdown']
    # Define categorical features
    sparse_features = ['UserID', 'Item', 'Lockdown', 'GroupID', 'GroupClass', 'GroupSemester', 'GroupLockdown']
    #===========================================================
elif case == 17:
    print("Executing Case 7: Running Prediction with GroupApp criteria, with all context")
     #===========================================================
    # test case 1: all context and criteria
    #Criteria ['GroupApp', 'GroupData', 'GroupEase']
    dense_features = ['App', 'Data', 'Ease', 'GroupApp']
    ## Context = ['Class', 'Semester', 'Lockdown']
    # Define categorical features
    sparse_features = ['UserID', 'Item', 'Class', 'Semester', 'Lockdown', 'GroupID', 'GroupClass', 'GroupSemester', 'GroupLockdown']
    #===========================================================
elif case == 18:
    print("Executing Case 18: Running Prediction with GroupData criteria, with all context")
     #===========================================================
    # test case 1: all context and criteria
    #Criteria ['GroupApp', 'GroupData', 'GroupEase']
    dense_features = ['App', 'Data', 'Ease', 'GroupData']
    ## Context = ['Class', 'Semester', 'Lockdown']
    # Define categorical features
    sparse_features = ['UserID', 'Item', 'Class', 'Semester', 'Lockdown', 'GroupID', 'GroupClass', 'GroupSemester', 'GroupLockdown']
    #===========================================================
else:
    print(f"Unknown case: {case}. Please provide a valid case number (1, 2, or 3).")


# Initialize and fit LabelEncoders
print("sparse_features", sparse_features)
for feat in sparse_features:
    print("feat: ", feat)
    le = LabelEncoder()
    df_ratings[feat] = le.fit_transform(df_ratings[feat].astype(str))
    label_encoders[feat] = le


# Initialize MinMaxScaler
scaler = MinMaxScaler()

# Fit and transform the dense features
df_ratings[dense_features] = scaler.fit_transform(df_ratings[dense_features])

# Define target variables
#target = ['GroupRating', 'GroupApp', 'GroupData', 'GroupEase']
target = ['GroupRating']
# ----------------------------
# Step 3: Prepare Data for Modeling
# ----------------------------

# Split the data
train_data, test_data = train_test_split(df_ratings, test_size=0.2, random_state=42)

# Define feature columns
# Define feature columns
sparse_feature_columns = [SparseFeat(feat, vocabulary_size=df_ratings[feat].nunique(), embedding_dim=4) for feat in sparse_features]
dense_feature_columns = [DenseFeat(feat, 1) for feat in dense_features]

feature_columns = sparse_feature_columns + dense_feature_columns

print("feature columns: ", feature_columns)
# Get feature names
feature_names = get_feature_names(feature_columns)

# Generate input data for model
train_model_input = {name: train_data[name].values for name in feature_names}
test_model_input = {name: test_data[name].values for name in feature_names}

# Ensure data types are correct
for name in feature_names:
    if name in sparse_features:
        train_model_input[name] = train_model_input[name].astype(int)
        test_model_input[name] = test_model_input[name].astype(int)
    else:
        train_model_input[name] = train_model_input[name].astype(float)
        test_model_input[name] = test_model_input[name].astype(float)

# Update the models dictionary
models = {
    
    "AFM": (AFM, {
        'linear_feature_columns': sparse_feature_columns + dense_feature_columns,  # AFM still uses DenseFeat here
        'dnn_feature_columns': sparse_feature_columns,  # REMOVE DenseFeat here
        'attention_factor': 8,
        'l2_reg_att': 1e-5,
        'use_attention': True,
    }),
    "AutoInt": (AutoInt, {
        'linear_feature_columns': sparse_feature_columns + dense_feature_columns,
        'dnn_feature_columns': sparse_feature_columns + dense_feature_columns,  # Keep both for other models
        'att_layer_num': 3,
        'dnn_hidden_units': (256, 128),
    }),
    "CCPM": (CCPM, {
        'linear_feature_columns': sparse_feature_columns + dense_feature_columns,  # Keep DenseFeat in linear features
        'dnn_feature_columns': sparse_feature_columns,  # REMOVE DenseFeat from dnn_feature_columns
        'conv_kernel_width': (6, 5),
        'conv_filters': (4, 4),
        'dnn_hidden_units': (256, 128),
    }),
    "DCN": (DCN, {
        'linear_feature_columns': sparse_feature_columns + dense_feature_columns,
        'dnn_feature_columns': sparse_feature_columns + dense_feature_columns,
        'cross_num': 3,
        'dnn_hidden_units': (256, 128),
    }),
    "DeepFM": (DeepFM, {
        'dnn_feature_columns': feature_columns,
    'linear_feature_columns': feature_columns,
        'dnn_hidden_units': (256, 128),
    }),
    "DIFM": (DIFM, {
        'dnn_feature_columns': feature_columns,
        'linear_feature_columns': feature_columns,
        'dnn_hidden_units': (256, 128),
    }),
    "FiBiNET": (FiBiNET, {
        'linear_feature_columns': feature_columns,
        'dnn_feature_columns': feature_columns,
        'dnn_hidden_units': (256, 128),
    }),
    "NFM": (NFM, {
        'linear_feature_columns': feature_columns,
        'dnn_feature_columns': feature_columns,
        'dnn_hidden_units': (256, 128),
    }),
    "ONN": (ONN, {
        'linear_feature_columns': feature_columns,
        'dnn_feature_columns': feature_columns,
        'dnn_hidden_units': (256, 128),
    }),
    "PNN": (PNN, {
        'dnn_feature_columns': sparse_feature_columns + dense_feature_columns,  # Only dnn_feature_columns
        'dnn_hidden_units': (256, 128),
        'use_inner': True,
        'use_outter': False,
    }),
    "WDL": (WDL, {
        'linear_feature_columns': feature_columns,
        'dnn_feature_columns': feature_columns,
        'dnn_hidden_units': (256, 128),
    }),
    "xDeepFM": (xDeepFM, {
        'linear_feature_columns': feature_columns,
       'dnn_feature_columns': feature_columns,
        'dnn_hidden_units': (256, 128),
        'cin_layer_size': (128, 128),
    }),
    "IFM": (IFM, {
        'linear_feature_columns': sparse_feature_columns + dense_feature_columns,
        
        'dnn_feature_columns': sparse_feature_columns + dense_feature_columns,
        'dnn_hidden_units': (256, 128),
    }),

     "LS-PLM": (MLR, {
        'region_feature_columns': sparse_feature_columns + dense_feature_columns,
       'base_feature_columns': sparse_feature_columns,  # Example
       'bias_feature_columns': dense_feature_columns,  # Example
    }),

    # AFN (Adaptive Factorization Network)

    "MHA": (MultiHeadAttentionModel, {
        'linear_feature_columns': feature_columns,
        'dnn_feature_columns': feature_columns,
        'num_heads': 4,
        'dnn_hidden_units': (256, 128, 64),
        'l2_reg_dnn': 1e-3,
        'dnn_dropout': 0.5,
    })
}

# Initialize results dictionary
results = {}

# Device configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Function to measure CPU & memory usage
def get_resource_usage():
    process = psutil.Process()
    cpu_percent = psutil.cpu_percent(interval=1)  # CPU usage over 1 second
    memory_info = process.memory_info().rss / (1024 * 1024)  # Convert bytes to MB
    return cpu_percent, memory_info

# Train and evaluate each model
for model_name, (model_class, model_params) in models.items():
    print(f"\nTraining and evaluating {model_name}...")

    # Start measuring time and resource usage
    start_time = time.time()
    cpu_before, memory_before = get_resource_usage()

    # Define common parameters
    common_params = {
        'task': 'regression',
        'device': device
    }
    # Update with model-specific parameters
    common_params.update(model_params)

    # Handle models that require special inputs
    if model_name in ['AFM']:
        common_params['use_attention'] = True

    # Define the model
    model = model_class(**common_params)
    model.to(device)

    # Compile the model
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mse','mae'])

    # Define Early Stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
    time.sleep(2)  # Simulate some work (replace with model training)
    # Train the model
    history = model.fit(
        train_model_input,
        train_data[target].values,
        batch_size=256,
        epochs=50,
        verbose=2,
        validation_split=0.2,
        callbacks=[early_stopping]
    )

    # Predict on test set
    predictions = model.predict(test_model_input, batch_size=256)

    # Calculate RMSE and MAE
    rmse = np.sqrt(mean_squared_error(test_data[target].values, predictions))
    mae = mean_absolute_error(test_data[target].values, predictions)

    # Convert MSE to RMSE and store both RMSE & MAE in history
    history.history['RMSE'] = np.sqrt(history.history['mse'])  # Convert MSE to RMSE
    history.history['val_RMSE'] = np.sqrt(history.history['val_mse'])  # Convert validation MSE to RMSE


    # Measure execution time and resource usage
    end_time = time.time()
    cpu_after, memory_after = get_resource_usage()
    execution_time = end_time - start_time
    cpu_usage = cpu_after - cpu_before
    memory_usage = memory_after - memory_before

    # Store the results
    results[model_name] = {
        'RMSE': rmse,
        'MAE': mae,
        'Execution Time (s)': execution_time,
        'CPU Usage (%)': cpu_usage,
        'Memory Usage (MB)': memory_usage
    }

    print(f"{model_name} Model RMSE: {rmse:.4f}")
    print(f"{model_name} Model MAE: {mae:.4f}")
    print(f"{model_name} Execution Time: {execution_time:.2f} seconds")
    print(f"{model_name} CPU Usage: {cpu_usage:.2f}%")
    print(f"{model_name} Memory Usage: {memory_usage:.2f} MB")

    # Save the history for MAE and RMSE after training each model
    with open(f'./outputs/pickles/{model_name}_history.pkl', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

# Display all results
print("\nFinal Results:")
for model_name, metrics in results.items():
    print(f"{model_name} - RMSE: {metrics['RMSE']:.4f}, MAE: {metrics['MAE']:.4f}, "
          f"Time: {metrics['Execution Time (s)']:.2f}s, CPU: {metrics['CPU Usage (%)']:.2f}%, "
          f"Memory: {metrics['Memory Usage (MB)']:.2f}MB")
