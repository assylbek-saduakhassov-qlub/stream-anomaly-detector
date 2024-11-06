# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
import logging
from collections import deque

# Configure logging
logging.basicConfig(filename='anomalies.log', level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')

# Data Stream Simulation Function
def data_stream():
    """
    Generator function to simulate a continuous data stream with
    regular patterns, multiple seasonal elements, concept drift, and random noise.
    Yields a tuple of (value, is_true_anomaly)
    """
    t = 0
    trend = 0.0
    seasonality_period = 20  # Base period for seasonality
    amplitude = 10  # Base amplitude for seasonality
    while True:
        # Update trend
        trend += 0.005  # Small incremental trend
        # Seasonal elements with changing amplitude and period to simulate concept drift
        seasonal = amplitude * np.sin((2 * np.pi / seasonality_period) * t)
        # Random noise
        noise = np.random.normal(0, 2)
        # Combine components
        value = trend + seasonal + noise
        is_true_anomaly = False

        # Simulate concept drift every 300 time steps
        if t % 300 == 0 and t != 0:
            # Change amplitude and period to simulate drift
            seasonality_period = random.choice([40, 50, 60])
            amplitude = random.uniform(8, 12)
            logging.info(f"Concept drift: new seasonality_period {seasonality_period}, amplitude {amplitude}")

        # Introduce anomalies
        # Point anomalies
        if random.random() < 0.02:
            anomaly = np.random.choice([20, -20])
            value += anomaly
            is_true_anomaly = True
            logging.info(f"Point anomaly injected: {value}")
        # Collective anomalies (sequence of anomalies)
        if t % 200 == 0 and t != 0:
            # Next 10 values are anomalies
            for _ in range(10):
                t += 1
                trend += np.random.uniform(-0.5, 0.5)  # Sudden trend change
                seasonal = amplitude * np.sin((2 * np.pi / seasonality_period) * t)
                noise = np.random.normal(0, 2)
                value = trend + seasonal + noise + np.random.choice([15, -15])
                is_true_anomaly = True
                logging.info(f"Collective anomaly injected: {value}")
                yield value, is_true_anomaly
            continue  # Skip the rest of the loop to yield the next value
        yield value, is_true_anomaly
        t += 1

# Anomaly Detection Class using Holt-Winters method with MAD
class HoltWintersMAD:
    """
    Holt-Winters method for online anomaly detection using Median Absolute Deviation.
    This class updates the level, trend, and seasonal components with each new data point
    and detects anomalies based on dynamic thresholds using MAD.
    """
    def __init__(self, alpha=0.5, beta=0.05, gamma=0.1, season_length=20, initial_threshold=3, warmup_period=1):
        """
        Initialize the Holt-Winters online model.

        Parameters:
        - alpha: Level smoothing coefficient.
        - beta: Trend smoothing coefficient.
        - gamma: Seasonal smoothing coefficient.
        - season_length: The length of the seasonality period.
        - initial_threshold: Number of MADs for anomaly detection.
        - warmup_period: Number of seasons to wait before starting anomaly detection.
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.season_length = season_length
        self.initial_threshold = initial_threshold
        self.threshold = initial_threshold
        self.level = None
        self.trend = None
        self.seasonals = [0] * season_length
        self.warmup_period = warmup_period * season_length
        self.seen = 0
        # For error calculation
        self.errors = deque(maxlen=season_length)
        self.anomaly_count = 0
        self.total_count = 0
        # For initializing series
        self.initial_series = []
        self.consecutive_anomalies = 0  # For detecting collective anomalies

    def initial_trend(self, series):
        """
        Calculate initial trend.
        """
        if len(series) < 2:
            return 0.0  # Not enough data to calculate trend
        sum = 0.0
        for i in range(len(series) - 1):
            sum += (series[i + 1] - series[i])
        return sum / (len(series) - 1)

    def initial_seasonal_components(self, series):
        """
        Calculate initial seasonal components.
        """
        seasonals = [0.0] * self.season_length
        if len(series) < self.season_length:
            return seasonals  # Not enough data for seasonal components
        season_averages = []
        n_seasons = max(1, int(len(series) / self.season_length))
        # Compute season averages
        for j in range(n_seasons):
            start = self.season_length * j
            end = start + self.season_length
            season = series[start:end]
            season_avg = sum(season) / float(len(season))
            season_averages.append(season_avg)
        # Compute initial seasonal components
        for i in range(self.season_length):
            sum_of_vals_over_avg = 0.0
            count = 0
            for j in range(n_seasons):
                index = self.season_length * j + i
                if index >= len(series):
                    break
                sum_of_vals_over_avg += series[index] - season_averages[j]
                count += 1
            if count > 0:
                seasonals[i] = sum_of_vals_over_avg / count
        return seasonals

    def update(self, value):
        """
        Update the Holt-Winters model with a new data point.

        Returns:
        - Tuple (is_anomaly, predicted_value, upper_limit, lower_limit)
        """
        self.total_count += 1
        self.seen += 1
        # Collect initial data for warm-up
        if self.seen <= self.warmup_period:
            self.initial_series.append(value)
            return (False, value, None, None)
        if self.level is None:
            # Initialize level, trend, and seasonal components
            self.level = sum(self.initial_series) / float(len(self.initial_series))
            self.trend = self.initial_trend(self.initial_series)
            self.seasonals = self.initial_seasonal_components(self.initial_series)
            # Proceed to update with the current value
        m = 1  # Assuming prediction for next time step
        if self.trend is None:
            self.trend = 0.0
        # Predict the next value
        seasonal = self.seasonals[self.total_count % self.season_length]
        predicted_value = self.level + m * self.trend + seasonal
        # Calculate error
        error = value - predicted_value
        # Update level, trend, and seasonals
        last_level = self.level
        self.level = self.alpha * (value - seasonal) + (1 - self.alpha) * (self.level + self.trend)
        self.trend = self.beta * (self.level - last_level) + (1 - self.beta) * self.trend
        self.seasonals[self.total_count % self.season_length] = self.gamma * (value - self.level) + (1 - self.gamma) * seasonal
        # Update errors for MAD calculation
        self.errors.append(error)
        # Calculate MAD
        errors = np.array(self.errors)
        median_error = np.median(errors)
        mad = np.median(np.abs(errors - median_error))
        # Convert MAD to standard deviation equivalent
        std = mad * 1.4826 if mad != 0 else np.std(errors)
        # Cap std to avoid extremely large thresholds
        std = min(std, 10)
        # Dynamic thresholds
        upper_limit = predicted_value + self.threshold * std
        lower_limit = predicted_value - self.threshold * std
        # Anomaly detection
        is_anomaly = value > upper_limit or value < lower_limit
        if is_anomaly:
            self.anomaly_count += 1
            self.consecutive_anomalies += 1  # Increment counter
            logging.info(f"Anomaly detected at value {value}")
            # Check for collective anomalies
            if self.consecutive_anomalies >= 3:
                logging.info(f"Collective anomaly detected starting at index {self.total_count - self.consecutive_anomalies + 1}")
        else:
            self.consecutive_anomalies = 0  # Reset counter
        # Adjust the threshold periodically
        if self.total_count % (5 * self.season_length) == 0:
            self.adjust_threshold()
        return (is_anomaly, predicted_value, upper_limit, lower_limit)

    def adjust_threshold(self):
        """
        Adjust the threshold based on the anomaly rate.
        """
        anomaly_rate = self.anomaly_count / self.total_count if self.total_count > 0 else 0
        # Adjust threshold based on anomaly rate
        if anomaly_rate > 0.05:
            self.threshold += 0.1  # Increase threshold
            logging.info(f"Increasing threshold to {self.threshold}")
        elif anomaly_rate < 0.01 and self.threshold > 1:
            self.threshold -= 0.1  # Decrease threshold
            logging.info(f"Decreasing threshold to {self.threshold}")

# Visualization Setup with Time in Seconds

fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.2)

# Use deque with a fixed maximum length for data storage
max_points = 100  # Adjust as needed
xdata = deque(maxlen=max_points)
ydata = deque(maxlen=max_points)
predicted_data = deque(maxlen=max_points)
upper_limit_data = deque(maxlen=max_points)
lower_limit_data = deque(maxlen=max_points)

# For anomalies, we will store time in seconds
anomalies_x = deque()
anomalies_y = deque()

# Initialize line objects
ln, = ax.plot([], [], 'b-', label='Data Stream')
pred_ln, = ax.plot([], [], 'g-', label='Predicted Value')
upper_ln, = ax.plot([], [], 'r--', label='Upper Limit')
lower_ln, = ax.plot([], [], 'r--', label='Lower Limit')
anomaly_ln, = ax.plot([], [], 'ro', label='Anomalies')

# Add text box for performance metrics
text_box = ax.text(0.02, 0.95, '', transform=ax.transAxes, verticalalignment='top')

def init():
    ax.set_xlim(0, max_points * 0.1)  # Initial x-axis limit in seconds
    ax.set_ylim(-50, 100)
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Value')
    ax.set_title('Real-Time Data Stream Anomaly Detection')
    ax.legend(loc='upper left')
    # Return the line objects to be updated
    return ln, pred_ln, upper_ln, lower_ln, anomaly_ln

# Variables for storing data
stream = data_stream()
detector = HoltWintersMAD(alpha=0.5, beta=0.05, gamma=0.1, season_length=20, initial_threshold=3, warmup_period=2)

# For performance metrics
true_labels = []
predicted_labels = []

def update(frame):
    # Get new data point
    try:
        value, is_true_anomaly = next(stream)
    except StopIteration:
        print("Data stream ended.")
        return ln, pred_ln, upper_ln, lower_ln, anomaly_ln
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        return ln, pred_ln, upper_ln, lower_ln, anomaly_ln

    # Calculate time in seconds
    time_in_seconds = frame * 0.1  # Since interval=100ms

    # Detect anomalies
    is_anomaly, predicted_value, upper_limit, lower_limit = detector.update(value)

    # Update data lists
    xdata.append(time_in_seconds)
    ydata.append(value)
    if predicted_value is not None:
        predicted_data.append(predicted_value)
    else:
        predicted_data.append(None)
    if upper_limit is not None and lower_limit is not None:
        upper_limit_data.append(upper_limit)
        lower_limit_data.append(lower_limit)
    else:
        upper_limit_data.append(None)
        lower_limit_data.append(None)

    # Update plot data using xdata directly
    ln.set_data(xdata, ydata)
    pred_ln.set_data(xdata, predicted_data)
    upper_ln.set_data(xdata, upper_limit_data)
    lower_ln.set_data(xdata, lower_limit_data)

    # Plot anomalies
    if is_anomaly:
        # Store the time in seconds
        anomalies_x.append(time_in_seconds)
        anomalies_y.append(value)
    # Remove anomalies that are no longer in the current window
    while anomalies_x and anomalies_x[0] < xdata[0]:
        anomalies_x.popleft()
        anomalies_y.popleft()
    # Update anomaly plot
    anomaly_ln.set_data(anomalies_x, anomalies_y)

    # Adjust x-axis limits
    ax.set_xlim(xdata[0], xdata[-1])

    # Record true labels and predictions
    true_labels.append(1 if is_true_anomaly else 0)
    predicted_labels.append(1 if is_anomaly else 0)

    # Compute performance metrics for the last window
    if len(true_labels) >= 10:
        from sklearn.metrics import precision_score, recall_score, f1_score
        precision = precision_score(true_labels[-100:], predicted_labels[-100:], zero_division=0)
        recall = recall_score(true_labels[-100:], predicted_labels[-100:], zero_division=0)
        f1 = f1_score(true_labels[-100:], predicted_labels[-100:], zero_division=0)
    else:
        precision, recall, f1 = 0, 0, 0

    # Update text box
    textstr = f'Precision: {precision:.2f}\nRecall: {recall:.2f}\nF1-score: {f1:.2f}'
    text_box.set_text(textstr)

    return ln, pred_ln, upper_ln, lower_ln, anomaly_ln, text_box

# Disable blitting to allow dynamic updates
ani = animation.FuncAnimation(fig, update, init_func=init, interval=100)

plt.show()
