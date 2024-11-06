# Efficient Data Stream Anomaly Detection

## Project Description
This project aims to develop a Python script capable of detecting anomalies in a continuous data stream in real-time. The data stream, simulating real-world sequences of floating-point numbers, may represent various metrics such as financial transactions or system performance indicators. The focus of the project is on identifying unusual patterns, such as exceptionally high values or deviations from the norm.

The project uses a modified Holt-Winters method for online anomaly detection, with Median Absolute Deviation (MAD) for robust anomaly detection. Seasonal variations, trend shifts, concept drift, and point/collective anomalies are simulated in the data stream.

## Objectives
- **Algorithm Selection**: Identify and implement a suitable algorithm for anomaly detection, capable of adapting to concept drift and seasonal variations.
- **Data Stream Simulation**: Design a function to emulate a data stream, incorporating regular patterns, seasonal elements, and random noise.
- **Anomaly Detection**: Develop a real-time mechanism to accurately flag anomalies as the data is streamed.
- **Optimization**: Ensure the algorithm is optimized for both speed and efficiency.
- **Visualization**: Create a real-time visualization tool to display both the data stream and any detected anomalies.

## Features
- **Data Stream Generation**: Simulates a continuous data stream with trends, seasonality, and noise.
- **Concept Drift and Seasonal Changes**: The data stream includes concept drift by adjusting seasonal parameters every few iterations.
- **Point and Collective Anomalies**: Introduces random anomalies to simulate realistic anomaly scenarios.
- **Real-Time Visualization**: Displays the data stream, predicted values, upper/lower bounds, and detected anomalies in real-time using `matplotlib`.
- **Dynamic Threshold Adjustment**: Uses MAD to calculate thresholds dynamically, adjusting based on detected anomaly rates.

## Requirements
- **Python 3.x**
- **Libraries**: The project requires the following libraries, which are listed in the `requirements.txt` file:
  - `numpy`
  - `matplotlib`
  - `scikit-learn`

To install all required libraries, run:
```bash
pip install -r requirements.txt
```

## File Structure
- `main.py`: The main Python script for simulating the data stream and detecting anomalies.
- `anomalies.log`: Log file to keep track of anomalies and concept drift.
- `requirements.txt`: Dependencies required for the project.

## Usage
1. **Clone the repository**:
   ```bash
   git clone https://github.com/assylbek-saduakhassov-qlub/StreamAnomalyDetector/
   ```
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the script**:
   ```bash
   python main.py
   ```
4. **Visualization**: The script will display the real-time data stream, predicted values, bounds, and anomalies.

## Anomaly Detection Algorithm
The anomaly detection is based on the **Holt-Winters Exponential Smoothing** model, enhanced with **Median Absolute Deviation (MAD)** for determining dynamic thresholds.

- **Level Smoothing**: Captures the current level of the data.
- **Trend Smoothing**: Tracks the rate of change in the level.
- **Seasonal Smoothing**: Accounts for repeating seasonal patterns.
- **Anomaly Detection**: Calculates a dynamic threshold using MAD and flags any value outside the predicted bounds as an anomaly.

The algorithm is particularly effective in handling **concept drift**, which is a gradual change in the underlying distribution of the data.

## Visualization
The project uses **matplotlib** for real-time visualization. The following elements are displayed:
- **Data Stream** (blue line): Represents the actual data points.
- **Predicted Value** (green line): The value predicted by the Holt-Winters model.
- **Upper/Lower Bounds** (red dashed lines): Dynamic limits calculated based on MAD.
- **Anomalies** (red dots): Data points flagged as anomalies.

## Error Handling
The script includes basic error handling to ensure the stability of the real-time simulation and logging of critical issues.

## Future Improvements
- **Model Optimization**: Improve the efficiency and accuracy of the anomaly detection model.
- **Scalability**: Adapt the solution to larger datasets and more complex data streams.
- **Integration with Stream Processing Frameworks**: Incorporate tools like Apache Kafka or Apache Flink for handling real-time streams at scale.
- **Enhanced GUI**: Provide a more interactive GUI for anomaly visualization.

## License
This project is licensed under the MIT License.

## Acknowledgments
Special thanks to the developers of the Holt-Winters and MAD methods, as well as open-source contributors who made this project possible.

Feel free to raise issues or contribute to the project by creating a pull request.

## Contact
For any questions or feedback, feel free to reach out to the project author.

