import os
import logging
import pandas as pd
import numpy as np
import plotly.express as px
from data_ingestion import ingest_data
from data_processing import process_data
from model import train_model, predict_next_gameweek

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("fpl_prediction.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def plot_top100_price_vs_points(predictions: dict, player_gw_df: pd.DataFrame, 
                                player_idlist_df: pd.DataFrame, playerraw_df: pd.DataFrame):
    """
    Plots a scatter plot of price vs. predicted fantasy points for the top 100 players.
    Each point is labeled with the player's name and colored by position:
      - Goalkeeper (element_type 1): blue
      - Defender (element_type 2): green
      - Midfielder (element_type 3): red
      - Striker (element_type 4): purple
    If playerraw_df is None, assigns default position (Unknown) with color gray.
    The 'value' is divided by 10 to display the dollar price.
    """
    # Convert predictions dictionary to DataFrame
    pred_df = pd.DataFrame(list(predictions.items()), columns=['player_id', 'predicted_points'])
    pred_df['player_id'] = pred_df['player_id'].astype(int)
    
    # Get the latest record per player (assuming higher gameweek means more recent)
    latest_player = player_gw_df.sort_values('gameweek').groupby('player_id').tail(1)
    latest_player = latest_player[['player_id', 'value']]
    latest_player['dollar_value'] = latest_player['value'] / 10.0  # convert to dollars
    
    # Merge predictions with latest price data
    merged = pd.merge(pred_df, latest_player, on='player_id', how='inner')
    
    # Merge with player_idlist to get full names
    merged = pd.merge(merged, player_idlist_df[['id', 'first_name', 'second_name']], 
                      left_on='player_id', right_on='id', how='left')
    merged['full_name'] = merged['first_name'] + " " + merged['second_name']
    
    # Merge with playerraw_df if available to get position; if not, assign default
    if playerraw_df is not None:
        merged = pd.merge(merged, playerraw_df[['id', 'element_type']], 
                          left_on='player_id', right_on='id', how='left')
    else:
        merged['element_type'] = 0  # unknown position
    
    # Map element_type to colors
    position_colors = {1: 'blue', 2: 'green', 3: 'red', 4: 'purple'}
    merged['color'] = merged['element_type'].map(position_colors)
    merged['color'] = merged['color'].fillna('gray')
    
    # Sort by predicted points descending and select top 100 players
    top100 = merged.sort_values('predicted_points', ascending=False).head(100)
    
    # Create interactive scatter plot with Plotly
    fig = px.scatter(
        top100, x='dollar_value', y='predicted_points',
        color='color', hover_name='full_name',
        labels={'dollar_value': 'Price ($)', 'predicted_points': 'Predicted Fantasy Points'},
        title='Top 100 Players by Predicted Fantasy Points vs Price'
    )
    fig.update_traces(marker=dict(size=10, line=dict(width=1, color='black')))
    fig.show()

def analyze_errors(model, X_test: np.ndarray, y_test: np.ndarray, 
                   player_gw_df: pd.DataFrame, player_idlist_df: pd.DataFrame):
    """
    Analyzes prediction errors and identifies players with the largest errors.
    """
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate errors
    errors = np.abs(y_test - y_pred.flatten())
    
    # Get player IDs and gameweeks from X_test
    player_ids = player_gw_df['player_id'].values
    gameweeks = player_gw_df['gameweek'].values
    
    # Create DataFrame for error analysis
    error_df = pd.DataFrame({
        'player_id': player_ids[-len(y_test):],  # Match test set size
        'gameweek': gameweeks[-len(y_test):],
        'actual_points': y_test,
        'predicted_points': y_pred.flatten(),
        'error': errors
    })
    
    # Merge with player_idlist to get names
    error_df = pd.merge(error_df, player_idlist_df[['id', 'first_name', 'second_name']], 
                        left_on='player_id', right_on='id', how='left')
    error_df['full_name'] = error_df['first_name'] + " " + error_df['second_name']
    
    # Sort by largest errors
    top_errors = error_df.sort_values('error', ascending=False).head(10)
    
    # Log top errors
    logger.info("\nTop 10 Players with Largest Prediction Errors:")
    logger.info(top_errors[['full_name', 'gameweek', 'actual_points', 'predicted_points', 'error']])

def main():
    logger.info("Starting data ingestion...")
    ingest_data()
    logger.info("Data ingestion completed.\n")
    
    logger.info("Starting data processing...")
    data = process_data()
    if data["X"] is None or data["y"] is None or data["player_gw_df"] is None:
        logger.error("No sequences created or no player gameweek data available. Exiting.")
        return
    logger.info("Data processing completed.\n")
    
    logger.info("Starting model training...")
    X = data["X"]
    y = data["y"]
    model, history = train_model(X, y)
    logger.info("Model training completed.\n")
    
    logger.info("Predicting next gameweek fantasy points for each player...")
    predictions = predict_next_gameweek(model=model, player_gw_df=data["player_gw_df"])
    
    # Use player_idlist and playerraw_df for mapping names and positions
    player_idlist_df = data["player_idlist_df"]
    playerraw_df = data["playerraw_df"]
    
    logger.info("\nPredictions:")
    for pid, pred in predictions.items():
        name_info = player_idlist_df.loc[player_idlist_df['id'] == pid, ['first_name', 'second_name']]
        if not name_info.empty:
            full_name = name_info.iloc[0]['first_name'] + " " + name_info.iloc[0]['second_name']
        else:
            full_name = f"Player {pid}"
        logger.info(f"{full_name} (ID: {pid}): Predicted fantasy points for next gameweek = {pred:.2f}")
    
    logger.info("\nPlotting Top 100 Players: Price vs Predicted Fantasy Points...")
    plot_top100_price_vs_points(predictions, data["player_gw_df"], player_idlist_df, playerraw_df)
    
    logger.info("\nAnalyzing prediction errors...")
    analyze_errors(model, X_test=X[-len(y)//5:], y_test=y[-len(y)//5:],  # Use last 20% as test set
                   player_gw_df=data["player_gw_df"], player_idlist_df=player_idlist_df)

if __name__ == "__main__":
    main()