# Steam Video Games Analysis and Recommendation System

This project explores and analyzes user behavior from Steam video games data, performs data preprocessing, and builds machine learning models to recommend games to users. The workflow includes data loading, exploratory data analysis (EDA), feature engineering, dimensionality reduction, clustering, and regression modeling.

# Contents of .zip file

This zip folder simply contains the README and source code (Jupyter notebook) created for the COS781 Semester project.

## Installation

Make sure you have the following dependencies installed:

```bash
pip install pandas scikit-learn kagglehub numpy matplotlib seaborn time tqdm
```

# Running the Code

To run the code, follow these steps:

1. Import the `.ipynb` (Jupyter notebook) file into a platform that supports running Jupyter notebooks, such as [Google Colab](https://colab.research.google.com/) or Jupyter Notebook locally.
2. Once the notebook is open, make sure to run all cells sequentially to ensure the environment is set up and all dependencies are loaded correctly.

In Google Colab:

- Upload the `.ipynb` file via the _File > Upload notebook_ option or use the _Google Drive_ integration to open the notebook directly.
- After uploading, click on the _Runtime > Run all_ to execute all the cells in the notebook.

This will ensure that the code executes correctly and all necessary outputs are generated.

# Code explanation

## Data Loading

The data is automatically downloaded from Kaggle datasets using kagglehub and processed to create user and game features:

```
data_original, steam_descriptions_original = load_data()
data = data_original.copy()
steam_descriptions = steam_descriptions_original.copy()
```

## Exploratory Data Analysis (EDA)

Perform EDA to understand the dataset:

```
perform_EDA(game_features)
perform_EDA(user_features)
plot_total_playtime(game_features, label_encoder)
plot_average_playtime(game_features, label_encoder)
plot_avg_hours_distribution(user_features)
plot_purchase_vs_num_bought(user_features)
plot_user_distribution(user_features)
```

## Feature Engineering and Normalization

Create new inferred characteristics and normalize data:

```
game_features, user_features, data, label_encoder = create_inferred_charactersitics(data, steam_descriptions, seed)
data = normalise_data(data)
game_features = normalise_data(game_features)
user_features = normalise_data(user_features)
```

## Dimensionality Reduction and Clustering

Use PCA for dimensionality reduction and KMeans for clustering:

```
user_features, game_features = perform_PCA(user_features, game_features)
user_features, game_features = perform_KMeans(user_features=user_features, game_features=game_features, seed=seed)
```

## Regression Modeling

Build and evaluate a regression model to predict playtime:

```
data_copy = data.merge(user_features[user_features.columns.values], on='user_id', how='left')
data_copy = data.merge(game_features[game_features.columns.values], on='game-title', how='left')
data_copy['behavior-name'] = data_copy['behavior-name'].apply(lambda x: 0 if x == 'purchase' else 1)
data_copy.dropna(inplace=True)
model = regression_model(data_copy, seed)
random_testing(data_copy, model, game_features)
```

## Game Recommendations

Recommend games to users based on similar user clusters and game playtime:

```
user_id = data['user_id'].sample(1).iloc[0]
recommended_games = recommend_games(user_id, data, user_features, game_features, label_encoder)
print("Recommended Games:", recommended_games)

recommendations = recommend_games_for_all_users(data, user_features, game_features, label_encoder)
for user, details in recommendations.items():
print(f"User {user} (Cluster {details['user_cluster']}): Recommended Games - {details['recommended_games']} in Clusters {details['game_clusters']}")
```

## Functions

- `load_data()`: Load Steam games data from Kaggle.
- `perform_EDA(data)`: Perform exploratory data analysis.
- `create_inferred_charactersitics(data, steam_descriptions, seed)`: Create new inferred characteristics for users and games.
- `normalise_data(data)`: Normalize numerical data.
- `perform_PCA(user_features, game_features)`: Perform PCA on user and game features.
- `perform_KMeans(user_features, game_features, seed)`: Cluster users and games using KMeans.
- `regression_model(data, seed)`: Build and evaluate a regression model.
- `random_testing(data, model, game_features)`: Test the regression model with random user-game instances.
- `recommend_games(user_id, data, user_features, game_features, label_encoder, top_n=10)`: Recommend games for a specific user.
- `recommend_games_for_all_users(data, user_features, game_features, label_encoder, top_n=10)`: Recommend games for all users and visualize the results.
- `visualize_recommendations(recommendations)`: Visualize the game recommendations.
