import matplotlib.pyplot as plt
import pickle

# List of model names
model_names = ["AutoInt", "DCN", "CCPM", "DIFM", "PNN", "IFM", "AFM","LS-PLM","DeepFM","FiBiNET","NFM","ONN","xDeepFM","WDL", "MHA"]  # Add other models as necessary

# Dictionary to store loaded histories
histories = {}

# Load the histories for all models
for model_name in model_names:
    with open(f'{model_name}_mvl_history.pkl', 'rb') as file_pi:
        histories[model_name] = pickle.load(file_pi)
print(histories[model_name].keys())  # See available keys

# Create a figure with 2 rows and 2 columns (for MAE and RMSE on train and validation)
fig, axs = plt.subplots(2, 2, figsize=(14, 10))

# Plot Training and Validation MAE
for model_name in model_names:
    axs[0, 0].plot(histories[model_name]['mse'], label=f'{model_name} Train MSE')
    axs[0, 1].plot(histories[model_name]['val_mse'], label=f'{model_name} Val MSE', linestyle='--')

# Plot Training and Validation RMSE
for model_name in model_names:
    axs[1, 0].plot(histories[model_name]['RMSE'], label=f'{model_name} Train RMSE')
    axs[1, 1].plot(histories[model_name]['val_RMSE'], label=f'{model_name} Val RMSE', linestyle='--')

# Set titles for each subplot
axs[0, 0].set_title('Training MSE')
axs[0, 1].set_title('Validation MSE')
axs[1, 0].set_title('Training RMSE')
axs[1, 1].set_title('Validation RMSE')

# Set common labels
for ax in axs.flat:
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Metric Value')
    ax.legend()

# Adjust layout
plt.tight_layout()
# Save as PDF
plt.savefig("mvl_plot.pdf", format="pdf", bbox_inches="tight")

print("Plot saved as 'plot.pdf' successfully!")

