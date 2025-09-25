# File: visualization.py
# Authors: [Your Name] 
# Date: 08/09/2025
# Task: COS30018 - Option C - Task C.3: Data Visualization (v0.2)

# This file implements advanced visualization techniques for stock market data
# Built upon the enhanced data processing function from Task C.2
# Reference tutorial: https://coderzcolumn.com/tutorials/data-science/candlestick-chart-in-python-mplfinance-plotly-bokeh

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

# Import our enhanced data processing function from Task C.2
from data_processing import load_and_process_data

#------------------------------------------------------------------------------
# Task C.3.1: Candlestick Chart Visualization Function
# A candlestick chart is a style of financial chart used to describe price movements
# Each "candle" represents price data for a single trading period (day, week, etc.)
#------------------------------------------------------------------------------

def plot_candlestick_chart(ticker, start_date, end_date, n_days=1, 
                          chart_title=None, save_path=None, figsize=(12, 8)):
    """
    Display stock market financial data using candlestick chart
    
    A candlestick chart shows four key price points for each trading period:
    - Open: Opening price of the period
    - High: Highest price reached during the period  
    - Low: Lowest price reached during the period
    - Close: Closing price of the period
    
    Each candle consists of:
    - Body: Rectangle between Open and Close prices
    - Wicks/Shadows: Lines extending to High and Low prices
    - Color: Green/White if Close > Open (bullish), Red/Black if Close < Open (bearish)
    
    Parameters:
    -----------
    ticker (str): Stock ticker symbol (e.g., 'CBA.AX', 'AAPL')
    start_date (str): Start date in 'YYYY-MM-DD' format
    end_date (str): End date in 'YYYY-MM-DD' format
    n_days (int): Number of trading days each candlestick represents (n ≥ 1)
                  - n=1: Each candle = 1 day (daily chart)
                  - n=5: Each candle = 1 week (weekly chart)  
                  - n=20: Each candle = ~1 month (monthly chart)
    chart_title (str): Custom title for the chart. If None, auto-generated
    save_path (str): Path to save the chart image. If None, display only
    figsize (tuple): Figure size as (width, height) in inches
    
    Returns:
    --------
    dict: Dictionary containing chart data and metadata
    """
    
    print(f"=== Task C.3.1: Creating Candlestick Chart for {ticker} ===")
    print(f"Period: {start_date} to {end_date}")
    print(f"Aggregation: {n_days} day(s) per candle")
    
    # Step 1: Load raw stock data using yfinance
    # We need OHLC (Open, High, Low, Close) data for candlestick chart
    print("📊 Loading stock data...")
    
    try:
        # Download data using yfinance
        # yfinance provides OHLC data which is perfect for candlestick charts
        data = yf.download(ticker, start=start_date, end=end_date)
        
        if data.empty:
            raise ValueError(f"No data found for ticker {ticker}")
            
        print(f"✅ Successfully loaded {len(data)} trading days of data")
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None
    
    # Step 2: Aggregate data if n_days > 1
    # This allows us to create weekly, monthly, or custom period candles
    if n_days > 1:
        print(f"📈 Aggregating data into {n_days}-day periods...")
        
        # Group data into n_days periods and aggregate OHLC values
        # - Open: First Open price of the period
        # - High: Maximum High price of the period  
        # - Low: Minimum Low price of the period
        # Close: Last Close price of the period
        # Volume: Sum of all volumes in the period
        
        # Calculate number of complete periods
        total_days = len(data)
        n_periods = total_days // n_days
        
        if n_periods == 0:
            print(f"⚠️  Warning: Not enough data for {n_days}-day aggregation")
            n_days = 1  # Fallback to daily
        else:
            # Create aggregated data
            aggregated_data = []
            
            for i in range(n_periods):
                # Get data slice for this period
                start_idx = i * n_days
                end_idx = min(start_idx + n_days, total_days)
                period_data = data.iloc[start_idx:end_idx]
                
                # Aggregate OHLC data
                # Open: First day's opening price
                period_open = period_data['Open'].iloc[0]
                
                # High: Highest price during the period
                period_high = period_data['High'].max()
                
                # Low: Lowest price during the period  
                period_low = period_data['Low'].min()
                
                # Close: Last day's closing price
                period_close = period_data['Close'].iloc[-1]
                
                # Volume: Total volume during the period
                period_volume = period_data['Volume'].sum()
                
                # Use the last date of the period as the period date
                period_date = period_data.index[-1]
                
                aggregated_data.append({
                    'Date': period_date,
                    'Open': float(period_open),
                    'High': float(period_high), 
                    'Low': float(period_low),
                    'Close': float(period_close),
                    'Volume': float(period_volume)
                })
            
            # Convert to DataFrame and set Date as index
            data = pd.DataFrame(aggregated_data)
            data.set_index('Date', inplace=True)
            
            print(f"✅ Aggregated into {len(data)} periods of {n_days} day(s) each")
    
    # Step 3: Prepare the candlestick chart
    print("🎨 Creating candlestick chart...")
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Set chart title
    if chart_title is None:
        period_text = f"{n_days}-Day" if n_days > 1 else "Daily"
        chart_title = f"{ticker} Stock Price - {period_text} Candlestick Chart\n{start_date} to {end_date}"
    
    ax.set_title(chart_title, fontsize=14, fontweight='bold', pad=20)
    
    # Step 4: Draw candlesticks
    # Each candlestick consists of:
    # 1. A rectangle (body) representing the range between Open and Close
    # 2. Vertical lines (wicks) representing the range between High/Low and the body
    
    candle_width = 0.8  # Width of each candle body (relative to date spacing)
    
    for i, (date, row) in enumerate(data.iterrows()):
        # Extract OHLC values for this candle
        # Convert to float to avoid pandas Series comparison issues
        open_price = float(row['Open'])
        high_price = float(row['High'])
        low_price = float(row['Low'])
        close_price = float(row['Close'])
        
        # Determine candle color based on price movement
        # Bullish candle (Close > Open): Green color, price went up
        # Bearish candle (Close < Open): Red color, price went down
        if close_price >= open_price:
            # Bullish candle: price increased
            body_color = 'green'
            edge_color = 'darkgreen'
            is_bullish = True
        else:
            # Bearish candle: price decreased
            body_color = 'red'
            edge_color = 'darkred'
            is_bullish = False
        
        # Calculate candle body dimensions
        # Body height = absolute difference between Open and Close
        # Body bottom = minimum of Open and Close
        body_height = abs(close_price - open_price)
        body_bottom = min(open_price, close_price)
        
        # Draw the candle body (rectangle)
        # Rectangle parameters:
        # - xy: bottom-left corner coordinates (date position, body_bottom)
        # - width: candle_width (aesthetic choice)
        # - height: body_height (price range)
        # - facecolor: fill color (green for bullish, red for bearish)
        # - edgecolor: border color (darker shade)
        body = Rectangle((i - candle_width/2, body_bottom), 
                        candle_width, body_height,
                        facecolor=body_color, 
                        edgecolor=edge_color,
                        linewidth=1.5,
                        alpha=0.8)
        ax.add_patch(body)
        
        # Draw the upper wick (high to top of body)
        # Line from the top of the body to the high price
        body_top = max(open_price, close_price)
        if high_price > body_top:
            ax.plot([i, i], [body_top, high_price], 
                   color=edge_color, linewidth=1.5, alpha=0.8)
        
        # Draw the lower wick (low to bottom of body)  
        # Line from the bottom of the body to the low price
        if low_price < body_bottom:
            ax.plot([i, i], [low_price, body_bottom], 
                   color=edge_color, linewidth=1.5, alpha=0.8)
    
    # Step 5: Format the chart
    print("🎯 Formatting chart appearance...")
    
    # Set x-axis with proper date labels
    # Convert dates to strings for x-axis labels
    date_labels = [date.strftime('%Y-%m-%d') for date in data.index]
    ax.set_xticks(range(len(data)))
    ax.set_xticklabels(date_labels, rotation=45, ha='right')
    
    # Show only every nth label to avoid overcrowding
    n_labels = min(10, len(data))  # Show at most 10 labels
    step = max(1, len(data) // n_labels)
    for i, label in enumerate(ax.get_xticklabels()):
        if i % step != 0:
            label.set_visible(False)
    
    # Set y-axis labels and formatting
    ax.set_ylabel('Price (AUD)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)  # Put grid behind the candles
    
    # Format y-axis to show currency values
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:.2f}'))
    
    # Add statistical information text box
    # Convert pandas values to float to avoid format string errors
    # Handle both scalar and Series cases
    try:
        period_high = float(data['High'].max())
        period_low = float(data['Low'].min())
        start_price = float(data['Open'].iloc[0])
        end_price = float(data['Close'].iloc[-1])
    except (ValueError, TypeError):
        # Handle pandas Series with ambiguous truth values
        high_vals = data['High'].values
        low_vals = data['Low'].values
        open_vals = data['Open'].values
        close_vals = data['Close'].values
        
        period_high = float(high_vals.max())
        period_low = float(low_vals.min())
        start_price = float(open_vals[0])
        end_price = float(close_vals[-1])
    
    total_return = ((end_price / start_price) - 1) * 100
    
    price_stats = {
        'Period High': f"${period_high:.2f}",
        'Period Low': f"${period_low:.2f}", 
        'Start Price': f"${start_price:.2f}",
        'End Price': f"${end_price:.2f}",
        'Total Return': f"{total_return:.2f}%"
    }
    
    stats_text = '\n'.join([f'{k}: {v}' for k, v in price_stats.items()])
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
           verticalalignment='top', fontsize=10,
           bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Step 6: Save or display the chart
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Chart saved to: {save_path}")
    
    plt.show()
    
    # Step 7: Return chart data and metadata
    result = {
        'data': data,
        'chart_info': {
            'ticker': ticker,
            'start_date': start_date,
            'end_date': end_date,
            'n_days': n_days,
            'total_periods': len(data),
            'price_stats': price_stats
        },
        'figure': fig
    }
    
    print("✅ Candlestick chart creation completed!")
    return result

#------------------------------------------------------------------------------
# Task C.3.2: Box Plot Visualization Function  
# Box plots are useful for displaying statistical distribution of stock prices
# over moving windows of n consecutive trading days
#------------------------------------------------------------------------------

def plot_boxplot_chart(ticker, start_date, end_date, window_size=20, 
                      price_column='Close', chart_title=None, save_path=None, figsize=(14, 8)):
    """
    Display stock market financial data using box plot chart
    
    A box plot (box-and-whisker plot) shows the statistical distribution of data:
    - Box: Represents the interquartile range (IQR) from Q1 to Q3
    - Line in box: Represents the median (Q2)  
    - Whiskers: Extend to show the range of data (typically 1.5 * IQR)
    - Outliers: Points beyond the whiskers (unusual price movements)
    
    This is particularly useful for analyzing price volatility and distribution
    over moving windows of consecutive trading days.
    
    Parameters:
    -----------
    ticker (str): Stock ticker symbol (e.g., 'CBA.AX', 'AAPL')
    start_date (str): Start date in 'YYYY-MM-DD' format
    end_date (str): End date in 'YYYY-MM-DD' format
    window_size (int): Number of consecutive trading days for each box plot
                      - Typical values: 5 (week), 20 (month), 60 (quarter)
    price_column (str): Which price to analyze ('Open', 'High', 'Low', 'Close')
    chart_title (str): Custom title for the chart. If None, auto-generated
    save_path (str): Path to save the chart image. If None, display only
    figsize (tuple): Figure size as (width, height) in inches
    
    Returns:
    --------
    dict: Dictionary containing chart data and statistical analysis
    """
    
    print(f"=== Task C.3.2: Creating Box Plot Chart for {ticker} ===")
    print(f"Period: {start_date} to {end_date}")
    print(f"Window size: {window_size} days per box")
    print(f"Analyzing: {price_column} prices")
    
    # Step 1: Load stock data
    print("📊 Loading stock data...")
    
    try:
        # Download data using yfinance
        data = yf.download(ticker, start=start_date, end=end_date)
        
        if data.empty:
            raise ValueError(f"No data found for ticker {ticker}")
            
        print(f"✅ Successfully loaded {len(data)} trading days of data")
        
        # Validate price column exists
        if price_column not in data.columns:
            print(f"⚠️  Warning: Column '{price_column}' not found. Available: {list(data.columns)}")
            price_column = 'Close'  # Fallback to Close price
            print(f"Using '{price_column}' instead")
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None
    
    # Step 2: Create moving windows of data
    print(f"📈 Creating moving windows of {window_size} days...")
    
    # Calculate how many complete windows we can create
    total_days = len(data)
    if total_days < window_size:
        print(f"❌ Error: Not enough data ({total_days} days) for window size {window_size}")
        return None
    
    # Create overlapping windows
    # Each window contains window_size consecutive days
    # Windows overlap by (window_size - 1) days for smoother analysis
    windows_data = []
    window_labels = []
    window_stats = []
    
    # Calculate step size for windows
    # We'll create windows with some overlap for better analysis
    step_size = max(1, window_size // 4)  # 25% step size for good overlap
    
    for start_idx in range(0, total_days - window_size + 1, step_size):
        end_idx = start_idx + window_size
        
        # Extract price data for this window
        window_prices = data[price_column].iloc[start_idx:end_idx].values
        
        # Calculate window date range for labeling
        start_date_window = data.index[start_idx]
        end_date_window = data.index[end_idx - 1]
        
        # Create label for this window
        label = f"{start_date_window.strftime('%m/%d')}-{end_date_window.strftime('%m/%d')}"
        
        # Store window data (flatten to ensure 1D array)
        windows_data.append(window_prices.flatten())
        window_labels.append(label)
        
        # Calculate statistical measures for this window
        window_stat = {
            'start_date': start_date_window,
            'end_date': end_date_window,
            'mean': np.mean(window_prices),
            'median': np.median(window_prices),
            'std': np.std(window_prices),
            'min': np.min(window_prices),
            'max': np.max(window_prices),
            'q1': np.percentile(window_prices, 25),  # First quartile
            'q3': np.percentile(window_prices, 75),  # Third quartile
            'iqr': np.percentile(window_prices, 75) - np.percentile(window_prices, 25)  # Interquartile range
        }
        window_stats.append(window_stat)
    
    print(f"✅ Created {len(windows_data)} overlapping windows")
    
    # Step 3: Create the box plot
    print("🎨 Creating box plot chart...")
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, height_ratios=[3, 1])
    
    # Main box plot
    box_plot = ax1.boxplot(windows_data, 
                          labels=window_labels,
                          patch_artist=True,  # Enable coloring
                          notch=True,         # Show confidence interval for median
                          showfliers=True,    # Show outliers
                          whis=1.5)          # Whisker length (1.5 * IQR)
    
    # Color the boxes with a gradient
    colors = plt.cm.viridis(np.linspace(0, 1, len(windows_data)))
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Customize box plot appearance
    for element in ['whiskers', 'fliers', 'medians', 'caps']:
        plt.setp(box_plot[element], color='black', linewidth=1.5)
    
    # Set main plot title and labels
    if chart_title is None:
        chart_title = f"{ticker} {price_column} Price Distribution\n{window_size}-Day Moving Windows ({start_date} to {end_date})"
    
    ax1.set_title(chart_title, fontsize=14, fontweight='bold', pad=20)
    ax1.set_ylabel(f'{price_column} Price (AUD)', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Format y-axis to show currency values
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:.2f}'))
    
    # Rotate x-axis labels for better readability
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
    
    # Show only every nth label to avoid overcrowding
    n_labels = min(15, len(window_labels))
    step = max(1, len(window_labels) // n_labels)
    for i, label in enumerate(ax1.get_xticklabels()):
        if i % step != 0:
            label.set_visible(False)
    
    # Step 4: Add volatility analysis subplot
    print("📊 Adding volatility analysis...")
    
    # Extract volatility measures (standard deviation) for each window
    volatilities = [stat['std'] for stat in window_stats]
    means = [stat['mean'] for stat in window_stats]
    
    # Plot volatility over time
    ax2.plot(range(len(volatilities)), volatilities, 
            color='red', linewidth=2, marker='o', markersize=4, alpha=0.7)
    ax2.set_ylabel('Volatility\n(Std Dev)', fontsize=10, fontweight='bold')
    ax2.set_xlabel('Window Period', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_title('Price Volatility Over Time', fontsize=12, fontweight='bold')
    
    # Format volatility axis
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:.2f}'))
    
    # Step 5: Add statistical summary
    print("📈 Calculating statistical summary...")
    
    # Calculate overall statistics across all windows
    all_prices = data[price_column].values
    overall_stats = {
        'Total Trading Days': len(all_prices),
        'Number of Windows': len(windows_data), 
        'Window Size': window_size,
        'Overall Mean': f"${np.mean(all_prices):.2f}",
        'Overall Volatility': f"${np.std(all_prices):.2f}",
        'Price Range': f"${np.min(all_prices):.2f} - ${np.max(all_prices):.2f}",
        'Average Window Volatility': f"${np.mean(volatilities):.2f}",
        'Max Window Volatility': f"${np.max(volatilities):.2f}",
        'Min Window Volatility': f"${np.min(volatilities):.2f}"
    }
    
    # Add summary text
    summary_text = '\n'.join([f'{k}: {v}' for k, v in overall_stats.items()])
    ax1.text(0.02, 0.98, summary_text, transform=ax1.transAxes,
            verticalalignment='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # Step 6: Add explanation text
    explanation = (
        "Box Plot Elements:\n"
        "• Box: Q1 to Q3 (50% of data)\n" 
        "• Line: Median price\n"
        "• Whiskers: 1.5 × IQR range\n"
        "• Dots: Outliers (unusual prices)"
    )
    
    ax1.text(0.98, 0.98, explanation, transform=ax1.transAxes,
            verticalalignment='top', horizontalalignment='right', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    # Adjust layout
    plt.tight_layout()
    
    # Step 7: Save or display the chart
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Chart saved to: {save_path}")
    
    plt.show()
    
    # Step 8: Return comprehensive analysis results
    result = {
        'raw_data': data,
        'windows_data': windows_data,
        'window_labels': window_labels,
        'window_stats': window_stats,
        'overall_stats': overall_stats,
        'chart_info': {
            'ticker': ticker,
            'start_date': start_date,
            'end_date': end_date,
            'window_size': window_size,
            'price_column': price_column,
            'num_windows': len(windows_data)
        },
        'figure': fig
    }
    
    print("✅ Box plot chart creation completed!")
    return result

#------------------------------------------------------------------------------
# Task C.3.3: Training History Visualization Function
# Visualize the training and validation loss/MAE over epochs for the prediction model
#------------------------------------------------------------------------------

def plot_training_history(history, save_path=None, title='Model Training History'):
    """
    Plots the training and validation loss and MAE from a Keras history object.

    Args:
        history (tf.keras.callbacks.History): History object from model.fit().
        save_path (str, optional): Path to save the plot image. Defaults to None.
        title (str, optional): Title for the plot. Defaults to 'Model Training History'.
    """
    print(f"🎨 Plotting training history: {title}")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Plot Loss
    ax1.plot(history.history['loss'], label='Training Loss')
    ax1.plot(history.history['val_loss'], label='Validation Loss')
    ax1.set_ylabel('Loss (MSE)')
    ax1.set_title('Loss Over Epochs')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot MAE
    ax2.plot(history.history['mae'], label='Training MAE')
    ax2.plot(history.history['val_mae'], label='Validation MAE')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Mean Absolute Error (MAE)')
    ax2.set_title('MAE Over Epochs')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    fig.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"✅ Training history plot saved to: {save_path}")
        
    plt.show()

#------------------------------------------------------------------------------
# Task C.3.4: Predictions Visualization Function
# Compare model predictions against actual values
#------------------------------------------------------------------------------

def plot_predictions_vs_actual(y_true, y_pred, save_path=None, title='Predictions vs. Actual Values'):
    """
    Plots the model's predictions against the actual true values.

    Args:
        y_true (np.ndarray): The actual values.
        y_pred (np.ndarray): The predicted values from the model.
        save_path (str, optional): Path to save the plot image. Defaults to None.
        title (str, optional): Title for the plot. Defaults to 'Predictions vs. Actual Values'.
    """
    print(f"🎨 Plotting predictions vs. actual values: {title}")
    
    plt.figure(figsize=(14, 7))
    plt.plot(y_true, color='blue', label='Actual Price')
    plt.plot(y_pred, color='red', label='Predicted Price', alpha=0.7)
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"✅ Predictions plot saved to: {save_path}")
        
    plt.show()

#------------------------------------------------------------------------------
# Example usage and testing functions
#------------------------------------------------------------------------------

def demo_candlestick_charts():
    """Demonstrate different candlestick chart configurations"""
    
    print("🚀 Demonstrating Candlestick Charts...")
    
    # Demo 1: Daily candlestick chart
    print("\n📊 Demo 1: Daily Candlestick Chart")
    daily_result = plot_candlestick_chart(
        ticker='CBA.AX',
        start_date='2024-01-01',
        end_date='2024-06-01',
        n_days=1,  # Daily candles
        chart_title="CBA Daily Candlestick Chart",
        save_path="candlestick_daily.png"
    )
    
    # Demo 2: Weekly candlestick chart (5-day aggregation)
    print("\n📊 Demo 2: Weekly Candlestick Chart") 
    weekly_result = plot_candlestick_chart(
        ticker='CBA.AX',
        start_date='2023-01-01',
        end_date='2024-01-01',
        n_days=5,  # Weekly candles (5 trading days)
        chart_title="CBA Weekly Candlestick Chart",
        save_path="candlestick_weekly.png"
    )
    
    return daily_result, weekly_result

def demo_boxplot_charts():
    """Demonstrate different box plot chart configurations"""
    
    print("🚀 Demonstrating Box Plot Charts...")
    
    # Demo 1: Monthly moving windows (20 days)
    print("\n📊 Demo 1: Monthly Box Plot Analysis")
    monthly_result = plot_boxplot_chart(
        ticker='CBA.AX',
        start_date='2023-01-01',
        end_date='2024-01-01', 
        window_size=20,  # ~1 month windows
        price_column='Close',
        chart_title="CBA Monthly Price Distribution Analysis",
        save_path="boxplot_monthly.png"
    )
    
    # Demo 2: Weekly moving windows (5 days)
    print("\n📊 Demo 2: Weekly Box Plot Analysis")
    weekly_result = plot_boxplot_chart(
        ticker='CBA.AX',
        start_date='2024-01-01',
        end_date='2024-06-01',
        window_size=5,   # Weekly windows
        price_column='Close',
        chart_title="CBA Weekly Price Distribution Analysis", 
        save_path="boxplot_weekly.png"
    )
    
    return monthly_result, weekly_result

if __name__ == "__main__":
    """
    Main execution block for testing the visualization functions
    """
    print("=" * 70)
    print("🎯 TASK C.3: ADVANCED DATA VISUALIZATION")
    print("📅 Date:", pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'))
    print("=" * 70)
    
    # Test candlestick charts
    daily_candle, weekly_candle = demo_candlestick_charts()
    
    # Test box plot charts  
    monthly_box, weekly_box = demo_boxplot_charts()
    
    print("\n" + "=" * 70)
    print("🎉 ALL VISUALIZATION DEMOS COMPLETED!")
    print("📁 Generated files:")
    print("   🕯️  candlestick_daily.png - Daily candlestick chart")
    print("   🕯️  candlestick_weekly.png - Weekly candlestick chart") 
    print("   📦 boxplot_monthly.png - Monthly box plot analysis")
    print("   📦 boxplot_weekly.png - Weekly box plot analysis")
    print("=" * 70)
