import pandas as pd
import numpy as np
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from datetime import datetime

np.random.seed(42)

# ---------------------------
# Load MovieLens 100k Data
# ---------------------------
data_path = "./ml-100k/"

ratings = pd.read_csv(
    os.path.join(data_path, 'u.data'),
    sep='\t',
    names=['UserID', 'MovieID', 'Rating', 'Timestamp'],
    encoding='latin1'
)

users = pd.read_csv(
    os.path.join(data_path, 'u.user'),
    sep='|',
    names=['UserID', 'Age', 'Gender', 'Occupation', 'Zip-code'],
    encoding='latin1'
)

# Define genre columns
genre_columns = [
    'Unknown', 'Action', 'Adventure', 'Animation', "Children's", 'Comedy', 'Crime',
    'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical',
    'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'
]
movie_columns = ['MovieID', 'Title', 'ReleaseDate', 'VideoReleaseDate', 'IMDbURL'] + genre_columns

# Load movies with genre indicators
movies = pd.read_csv(
    os.path.join(data_path, 'u.item'),
    sep='|',
    names=movie_columns,
    usecols=range(24),
    encoding='latin1',
    engine='python'
)

# Reconstruct genre labels
movies['Genres'] = movies[genre_columns].apply(
    lambda row: '|'.join([g for g, val in zip(genre_columns, row) if val == 1]), axis=1
)
movies = movies[['MovieID', 'Title', 'Genres'] + genre_columns]

# ---------------------------
# Merge Datasets
# ---------------------------
merged = ratings.merge(users, on='UserID').merge(movies, on='MovieID')

# ---------------------------
# Add Contextual Features
# ---------------------------
merged['Datetime'] = pd.to_datetime(merged['Timestamp'], unit='s')
merged['Hour'] = merged['Datetime'].dt.hour
merged['DayOfWeek'] = merged['Datetime'].dt.dayofweek

merged['TimeOfDay'] = pd.cut(
    merged['Hour'],
    bins=[0, 6, 12, 18, 24],
    labels=['Night', 'Morning', 'Afternoon', 'Evening'],
    right=False
)
merged['DayType'] = merged['DayOfWeek'].apply(lambda x: 'Weekend' if x >= 5 else 'Weekday')
merged['Month'] = merged['Datetime'].dt.month

def get_season(month):
    if month in [12, 1, 2]: return 'Winter'
    elif month in [3, 4, 5]: return 'Spring'
    elif month in [6, 7, 8]: return 'Summer'
    else: return 'Autumn'

merged['Season'] = merged['Month'].apply(get_season)
merged.drop(columns=['Hour', 'DayOfWeek', 'Month', 'Datetime', 'Zip-code'], inplace=True)

# ---------------------------
# Simulate Group IDs via KMeans
# ---------------------------
le_occ = LabelEncoder()
users['OccupationEncoded'] = le_occ.fit_transform(users['Occupation'])

user_genre_matrix = merged[['UserID'] + genre_columns].groupby('UserID').mean()
user_demo = users.set_index('UserID')[['Age', 'OccupationEncoded']].join(user_genre_matrix).fillna(0)

n_clusters = min(250, len(user_demo))
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
users['GroupID'] = kmeans.fit_predict(user_demo)

# ✔ Ensure each user is in only one group
assert users['UserID'].is_unique, "Duplicate users detected!"
assert users.groupby('UserID')['GroupID'].nunique().max() == 1, "A user is in multiple groups!"

# ➕ Optional: show group distribution
group_counts = users['GroupID'].value_counts().sort_index()
print(f"👥 Number of users per group:\n{group_counts}\n")

# Merge GroupID into full data
merged = merged.merge(users[['UserID', 'GroupID']], on='UserID')

# ---------------------------
# Simulate Multi-Criteria Ratings
# ---------------------------
merged['Storyline'] = np.clip(
    merged['Rating'] + np.random.normal(0, 4.0, len(merged)), 1, 5).astype(np.float32)
merged['Visuals'] = np.clip(
    merged['Rating'] + np.random.normal(0, 4.0, len(merged)), 1, 5).astype(np.float32)
merged['Emotion'] = np.clip(
    merged['Rating'] + np.random.normal(0, 4.0, len(merged)), 1, 5).astype(np.float32)

# ---------------------------
# Add User Demographics (Renamed)
# ---------------------------
user_info = users[['UserID', 'Age', 'OccupationEncoded']].rename(columns={
    'Age': 'UserAge',
    'OccupationEncoded': 'UserOccupation'
})
final = merged.merge(user_info, on='UserID', how='left')

# ---------------------------
# Final Formatting
# ---------------------------
final_cols = ['UserID', 'GroupID', 'MovieID', 'Title', 'Genres', 'UserAge', 'UserOccupation',
              'TimeOfDay', 'DayType', 'Season', 'Storyline', 'Visuals', 'Emotion', 'Rating']
final = final[final_cols]
final = final.rename(columns={'Rating': 'OverallRating'})

final = final.dropna()

# Save to CSV
output_path = 'final_ml100k_user_context_criteria.csv'
final.to_csv(output_path, index=False)
print(f"✅ Saved dataset to {output_path}")

# ---------------------------
# Statistics Summary
# ---------------------------
num_users = final['UserID'].nunique()
num_groups = final['GroupID'].nunique()
num_items = final['MovieID'].nunique()

contexts = ['TimeOfDay', 'DayType', 'Season']
criteria = ['Storyline', 'Visuals', 'Emotion']
rating_scale = '[1,5]'

sparsity_individual = 100 * (1 - ratings.shape[0] / (num_users * num_items))
sparsity_group = 100 * (1 - final.groupby(['GroupID', 'MovieID']).ngroups / (num_groups * num_items))

stats = pd.DataFrame({
    'Object': ['Individual', 'Group'],
    'Quantity': [num_users, num_groups],
    'Nb of Item': [num_items, ''],
    'Contexts': [', '.join(contexts), ''],
    'Criteria': [', '.join(criteria), ''],
    'Rating Scale': [rating_scale, ''],
    'Data Sparsity': [f"{sparsity_individual:.2f}%", f"{sparsity_group:.2f}%"],
    'Nb of Rating': [ratings.shape[0], final.shape[0]]
})

stats.to_csv("250_ml100k_statistics_summary.csv", index=False)
print("📊 Saved statistics to ml100k_statistics_summary.csv")
