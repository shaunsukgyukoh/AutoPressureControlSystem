# pip install pandas matplotlib seaborn scikit-learn scipy numpy openpyxl

import pandas as pd
import matplotlib.pyplot as plt
import os
from openpyxl import load_workbook
from datetime import datetime
import numpy as np
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from scipy import stats
import seaborn as sns
from scipy.stats import sem, t
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

class DataPlotter:
    """
    A class to load data from a CSV file and plot a scatter plot for the response time experiment.
    """

    def __init__(self, file_path):
        """
        Initialize the DataPlotter with the CSV file path.
        
        :param file_path: str, path to the CSV file
        """
        self.file_path = file_path
        self.data = None

    def calibrated_data_regression_b-a_analysis(self):
        # # Data from excel
        gauge = np.array([100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350, 375, 400, 425, 450, 475, 500, 525, 550, 575, 600, 625, 650, 675])
        sensor = np.array([100.1022222, 125.5068421, 149.817037, 174.1431579, 200.3745455, 225.3185185, 248.5, 274.5654167, 298.7345833, 323.2391304, 
                    350.1111538, 376.4888235, 399.9691667, 424.2386667, 451.2264, 475.4955172, 497.4583333, 527.3904762, 550.045, 578.4444444, 
                    598.6525, 625.1460714, 650.2666667, 675.3565385])

        # Load data from Excel
        # file_path = 'C:\Users\Administ\Desktop\BM\Data\calibrated data test.xlsx'  # Replace with your actual file path
        # df = pd.read_excel(file_path)

        # # Extract gauge and sensor data from the DataFrame
        # gauge = df.iloc[:, 0].values
        # sensor = df.iloc[:, 1].values

        # Reshape the data for sklearn
        gauge_reshaped = gauge.reshape(-1, 1)

        # Linear regression model
        model = LinearRegression()
        model.fit(gauge_reshaped, sensor)

        # Predict values for the regression line
        sensor_pred = model.predict(gauge_reshaped)

        # R^2 value
        r_squared = model.score(gauge_reshaped, sensor)

        # Equation of the line
        slope = model.coef_[0]
        intercept = model.intercept_

        # Plot: Sensor vs Gauge Correlation
        plt.figure(figsize=(10, 6))
        plt.scatter(gauge, sensor, color='blue', label='Data Points')
        plt.plot(gauge, sensor_pred, color='red', label='Regression Line')
        plt.title('Sensor vs Gauge Correlation')
        plt.xlabel('Gauge (mmHg)')
        plt.ylabel('Sensor (mmHg)')

        # Display the equation and R^2 value on the plot
        equation_text = f'y = {slope:.4f}x + {intercept:.4f}\nR² = {r_squared:.4f}'
        plt.text(0.05, 0.95, equation_text, transform=plt.gca().transAxes, fontsize=12,
            verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))

        plt.legend()
        plt.grid(True)

        # Generate the current timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Save the plot in PNG and SVG formats
        plt.savefig(f'sensor_vs_gauge_correlation_{timestamp}.png')
        plt.savefig(f'sensor_vs_gauge_correlation_{timestamp}.svg')

        # Show the plot
        plt.show()

        # Bland-Altman Plot
        mean_values = (sensor + gauge) / 2
        differences = sensor - gauge
        mean_diff = np.mean(differences)
        std_diff = np.std(differences)
        std_diff = np.std(differences, ddof=1)  # Use ddof=1 for sample standard deviation
        n = len(differences)
        se_diff = std_diff / np.sqrt(n)  # Standard error of the mean difference

        # Limits of agreement
        ci_upper_95 = mean_diff + 1.96 * std_diff
        ci_lower_95 = mean_diff - 1.96 * std_diff

        # Plot: Bland-Altman
        plt.figure(figsize=(10, 6))
        plt.scatter(mean_values, differences, color='blue', label='Differences')
        plt.axhline(mean_diff, color='red', linestyle='--', label='Mean Difference')
        plt.axhline(ci_upper_95, color='green', linestyle='--', label='Upper LOA (+1.96 SD)')
        plt.axhline(ci_lower_95, color='orange', linestyle='--', label='Lower LOA (-1.96 SD)')

        plt.title('Bland-Altman Plot')
        plt.xlabel('Mean of Gauge and Sensor (mmHg)')
        plt.ylabel('Difference (Sensor - Gauge) (mmHg)')
        plt.legend()
        plt.grid(True)

        # Save the Bland-Altman plot in PNG and SVG formats with the current timestamp
        plt.savefig(f'bland_altman_plot_{timestamp}.png')
        plt.savefig(f'bland_altman_plot_{timestamp}.svg')

        # Show the Bland-Altman plot
        plt.show()


        # Format the table
        table = f"""
        Mean Bias\tStandard Deviation\tLower Limit of Agreement (LLA)\tUpper Limit of Agreement (ULA)\tStandard Error (SE)\tLower 95% CI of ULA\tUpper 95% CI of ULA
        {mean_diff:.4f}\t{std_diff:.4f}\t{se_diff:.4f}\t{ci_lower_95:.4f}\t{ci_upper_95:.4f}
        """

        print(table)

    def load_data(self):
        """
        Load data from the CSV file and extract necessary columns.
        """
        try:
            if not os.path.isfile(self.file_path):
                print(f"Error: The path '{self.file_path}' is not a valid file.")
                return

            # Load data based on file extension
            if self.file_path.endswith('.csv'):
                self.data = pd.read_csv(self.file_path, encoding='latin1')  # Adjust encoding if needed
                self.data = self.data.applymap(lambda x: x.strip('[]') if isinstance(x, str) else x)

            elif self.file_path.endswith('.xlsx'):
                # Load workbook with openpyxl using data_only=True
                wb = load_workbook(self.file_path, data_only=True)
                sheet = wb.active
                self.data = pd.DataFrame(sheet.values)

                # Set the first row as the header and remove NaN columns
                self.data.columns = self.data.iloc[0]
                self.data = self.data[1:]
                self.data.dropna(axis=1, how='all', inplace=True)  # Drop columns where all values are NaN
            else:
                print(f"Error: Unsupported file type for '{self.file_path}'")
                return
            
            

            print(f"Data loaded successfully from {self.file_path}.")
            print("Available columns:", self.data.columns)  # Print available columns to check
            
        except FileNotFoundError:
            print(f"Error: File not found. Please check the file path: {self.file_path}")
        except PermissionError:
            print(f"Error: Permission denied. Please check the file permissions: {self.file_path}")
        except Exception as e:
            print(f"An error occurred while loading the data: {e}")
    
    def generate_filename(self, prefix, fmat):
        """
        Generate a filename with the current timestamp.
        
        :param prefix: str, the prefix for the filename
        :return: str, the generated filename
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'{prefix}_{timestamp}.{fmat}'
        return filename

    def save_plot(self, filename):
        """
        Save the plot to a file with the given filename.
        
        :param filename: str, the filename to save the plot as
        """
        plt.savefig(filename)
        print(f"Plot saved as {filename}")

    def plot_relative_pressure_cycles(self):
        """
        Plot relative pressure for 10 repeated cycles based on the reset points, 
        each with a different color and line style.
        """
        if self.data is None:
            print("Data not loaded. Please load the data first.")
            return

        try:
                # Use relative pressure data for plotting
            relative_pressure = self.data.iloc[2:, 2].astype(float)  # 3rd column (Relative Pressure)

            # Determine the number of segments (each of 4000 indices)
            segment_length = 200 # 4000
            num_segments = len(relative_pressure) // segment_length

            # Define line styles and colors for plotting
            line_styles = [
                (0, (1, 10)),
                (5, (10, 3)),
                (0, (5, 10)),
                (0, (5, 5)),
                (0, (3, 10, 1, 10)),
                (0, (3, 5, 1, 5)),
                (0, (3, 1, 1, 1)),
                (0, (3, 5, 1, 5, 1, 5)),
                (0, (3, 10, 1, 10, 1, 10)),
                (0, (3, 1, 1, 1, 1, 1))]
            colors = plt.cm.viridis(np.linspace(0, 1, num_segments))

            plt.figure(figsize=(10, 6))

            # Plot each segment with a different color and line style
            for i in range(num_segments):
                start_index = i * segment_length
                end_index = (i + 1) * segment_length
                cycle_pressure = relative_pressure.iloc[start_index:end_index].reset_index(drop=True)  # Reset index for clean plotting

                # Map x-axis indices (0 to 4000) to time (0 to 200 seconds)
                time_seconds = np.linspace(0, 200, len(cycle_pressure))

                plt.plot(time_seconds, cycle_pressure, linestyle=line_styles[i % 10], color=colors[i], label=f'Cycle {i + 1}')

            plt.title('Suction Regulation Stability Test Over 10 Repeated Cycles', fontsize=14)
            plt.xlabel('Time (s)', fontsize=14)
            plt.ylabel('Relative Pressure (mmHg)', fontsize=14)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            plt.grid(True)
            plt.legend()

            filename = self.generate_filename('relative_pressure_cycles','png')
            self.save_plot(filename)
            plt.close()  # Close the plot after saving

        except KeyError as e:
            print(f"Error: {e}")
            print("Available columns:", self.data.columns)
        except Exception as e:
            print(f"An error occurred while plotting the data: {e}")

    def plot_relative_pressure_cycles_t(self, mode = 'mult'):
        """
        Plot relative pressure for 10 repeated cycles based on elapsed time points,
        each with a different color and line style.
        """
        if self.data is None:
            print("Data not loaded. Please load the data first.")
            return

        try:
            # Use elapsed time and relative pressure data for plotting
            elapsed_time = self.data['Elapsed Time (s)'].astype(float)  # Elapsed Time column
            relative_pressure = self.data.iloc[:, 2].astype(float)  # 3rd column (Relative Pressure)

            # Define the time points for cutting segments
            time_points = [1.7063934803009, 611.870562314987, 1221.98820424079, 1832.10683965682, 
                        2442.23349046707, 3052.35604953765, 3662.48283410072, 4272.59746432304, 
                        4882.69602203369, 5492.81426215171, 6102.85762763023]

            # Determine the number of segments based on time points
            num_segments = len(time_points) - 1

            # Define line styles and colors for plotting
            line_styles = [
                (0, (1, 10)),
                (5, (10, 3)),
                (0, (5, 10)),
                (0, (5, 5)),
                (0, (3, 10, 1, 10)),
                (0, (3, 5, 1, 5)),
                (0, (3, 1, 1, 1)),
                (0, (3, 5, 1, 5, 1, 5)),
                (0, (3, 10, 1, 10, 1, 10)),
                (0, (3, 1, 1, 1, 1, 1))]
            
            colors = plt.cm.viridis(np.linspace(0, 1, num_segments))

            plt.figure(figsize=(10, 6))

            # Reference pressure for Bland-Altman analysis
            reference_pressure = 10.0

            for i in range(num_segments):
                # Find the indices for the start and end times
                start_time = time_points[i]
                end_time = time_points[i + 1]
                segment_indices = elapsed_time[(elapsed_time >= start_time) & (elapsed_time < end_time)].index

                # Extract the corresponding pressure data and time data
                segment_pressure = relative_pressure.loc[segment_indices]
                segment_time = elapsed_time.loc[segment_indices]

                # Get the data for the last 600 seconds
                last_600_seconds_indices = segment_time[segment_time >= (end_time - 600)].index
                last_600_pressure = relative_pressure.loc[last_600_seconds_indices]

                # Bland-Altman analysis
                differences = last_600_pressure - reference_pressure
                mean_diff = np.mean(differences)
                std_diff = np.std(differences, ddof=1)
                std_error = std_diff / np.sqrt(len(differences))
                loa = 1.96 * std_diff
                upper_loa = mean_diff + loa
                lower_loa = mean_diff - loa

                # Print Bland-Altman statistics for each segment
                print(f"{i + 1}:")
                print(f"  Mean Difference: {mean_diff:.4f}")
                print(f"  Standard Deviation of Differences: {std_diff:.4f}")
                print(f"  Standard Error: {std_error:.4f}")
                print(f"  Upper 95% LoA: {upper_loa:.4f}")
                print(f"  Lower 95% LoA: {lower_loa:.4f}\n")

                if mode == 'one':
                    # Plot each segment with a different color
                    plt.plot(segment_time, segment_pressure, linestyle=line_styles[i % len(line_styles)], color=colors[i], label=f'Segment {i + 1}')
                else:
                    # Extract the corresponding pressure data
                    cycle_pressure = relative_pressure.loc[segment_indices].reset_index(drop=True)

                    # Map x-axis to relative time within each segment
                    segment_time = elapsed_time.loc[segment_indices] - start_time
                    plt.plot(segment_time, cycle_pressure, linestyle=line_styles[i % len(line_styles)], 
                            color=colors[i], label=f'Cycle {i + 1}')

            plt.title('Suction Regulation Stability Test', fontsize=14)
            plt.xlabel('Time (s)', fontsize=14)
            plt.ylabel('Relative Pressure (mmHg)', fontsize=14)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            plt.grid(True)
            plt.legend()

            filename = self.generate_filename('relative_pressure_cycles_time_mapped', 'png')
            self.save_plot(filename)
            plt.close()  # Close the plot after saving

        except KeyError as e:
            print(f"Error: {e}")
            print("Available columns:", self.data.columns)
        except Exception as e:
            print(f"An error occurred while plotting the data: {e}")

    def bland_altman_analysis_cycle(self):
        """
        Perform Bland-Altman analysis for the last 600 seconds of each segment and combined data.
        Save plots and show analysis results.
        """
        if self.data is None:
            print("Data not loaded. Please load the data first.")
            return

        try:
            # Use elapsed time and relative pressure data for plotting
            elapsed_time = self.data['Elapsed Time (s)'].astype(float)  # Elapsed Time column
            relative_pressure = self.data.iloc[:, 2].astype(float)  # 3rd column (Relative Pressure)

            # Define the time points for cutting segments
            time_points = [1.7063934803009, 611.870562314987, 1221.98820424079, 1832.10683965682, 
                        2442.23349046707, 3052.35604953765, 3662.48283410072, 4272.59746432304, 
                        4882.69602203369, 5492.81426215171, 6102.85762763023]

            # Reference pressure for Bland-Altman analysis
            reference_pressure = 10.0
            all_last_600_diffs = []  # To store differences for combined Bland-Altman analysis

            # Perform Bland-Altman analysis for each segment
            for i in range(len(time_points) - 1):
                # Get the start and end times for the segment
                start_time = time_points[i]
                end_time = time_points[i + 1]

                # Extract data for the current segment
                segment_indices = elapsed_time[(elapsed_time >= start_time) & (elapsed_time < end_time)].index
                segment_pressure = relative_pressure.loc[segment_indices]
                segment_time = elapsed_time.loc[segment_indices]

                # Get the data for the last 600 seconds of the current segment
                last_600_seconds_indices = segment_time[segment_time >= (end_time - 600)].index
                last_600_pressure = relative_pressure.loc[last_600_seconds_indices]

                # Bland-Altman analysis for the segment
                differences = last_600_pressure - reference_pressure
                mean_diff = np.mean(differences)
                std_diff = np.std(differences, ddof=1)
                std_error = std_diff / np.sqrt(len(differences))
                loa = 1.96 * std_diff
                upper_loa = mean_diff + loa
                lower_loa = mean_diff - loa

                # Store differences for combined analysis
                all_last_600_diffs.extend(differences)

                # Print Bland-Altman statistics for the segment
                print(f"Segment {i + 1}:")
                print(f"  Mean Difference: {mean_diff:.4f}")
                print(f"  Standard Deviation of Differences: {std_diff:.4f}")
                print(f"  Standard Error: {std_error:.4f}")
                print(f"  Upper 95% LoA: {upper_loa:.4f}")
                print(f"  Lower 95% LoA: {lower_loa:.4f}\n")

                # Plot Bland-Altman for the segment
                plt.figure(figsize=(10, 6))
                plt.scatter(last_600_pressure.index, differences, color='black', s=10)  # Black dots
                plt.axhline(mean_diff, color='r', linestyle='-', label=f'Mean diff: {mean_diff:.2f}')
                plt.axhline(upper_loa, color='g', linestyle='--', label=f'Upper LoA: {upper_loa:.2f}')
                plt.axhline(lower_loa, color='b', linestyle='--', label=f'Lower LoA: {lower_loa:.2f}')

                plt.title(f'Bland-Altman Plot for Segment {i + 1}', fontsize=14)
                plt.xlabel('Index (Relative to Last 600s)', fontsize=14)
                plt.ylabel('Difference from 10 mmHg (mmHg)', fontsize=14)
                plt.xticks(fontsize=14)
                plt.yticks(fontsize=14)
                plt.grid(True)
                plt.legend()

                segment_filename = self.generate_filename(f'bland_altman_segment_{i + 1}', 'png')
                self.save_plot(segment_filename)
                plt.close()  # Close the plot after saving

            # Combined Bland-Altman analysis for all last 600 seconds data
            all_last_600_diffs = np.array(all_last_600_diffs)
            combined_mean_diff = np.mean(all_last_600_diffs)
            combined_std_diff = np.std(all_last_600_diffs, ddof=1)
            combined_std_error = combined_std_diff / np.sqrt(len(all_last_600_diffs))
            combined_loa = 1.96 * combined_std_diff
            combined_upper_loa = combined_mean_diff + combined_loa
            combined_lower_loa = combined_mean_diff - combined_loa

            # Print combined Bland-Altman statistics
            print("Combined Data for Last 600 Seconds:")
            print(f"  Mean Difference: {combined_mean_diff:.4f}")
            print(f"  Standard Deviation of Differences: {combined_std_diff:.4f}")
            print(f"  Standard Error: {combined_std_error:.4f}")
            print(f"  Upper 95% LoA: {combined_upper_loa:.4f}")
            print(f"  Lower 95% LoA: {combined_lower_loa:.4f}\n")

            # Plot combined Bland-Altman
            plt.figure(figsize=(10, 6))
            plt.scatter(range(len(all_last_600_diffs)), all_last_600_diffs, color='black', s=10)  # Black dots
            plt.axhline(combined_mean_diff, color='r', linestyle='-', label=f'Mean diff: {combined_mean_diff:.2f}')
            plt.axhline(combined_upper_loa, color='g', linestyle='--', label=f'Upper LoA: {combined_upper_loa:.2f}')
            plt.axhline(combined_lower_loa, color='b', linestyle='--', label=f'Lower LoA: {combined_lower_loa:.2f}')

            plt.title('Bland-Altman Plot for Combined Last 600 Seconds Data', fontsize=14)
            plt.xlabel('Data Point Index', fontsize=14)
            plt.ylabel('Difference from 10 mmHg (mmHg)', fontsize=14)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            plt.grid(True)
            plt.legend()

            combined_filename = self.generate_filename('bland_altman_combined_last_600', 'png')
            self.save_plot(combined_filename)
            plt.close()  # Close the plot after saving

        except KeyError as e:
            print(f"Error: {e}")
            print("Available columns:", self.data.columns)
        except Exception as e:
            print(f"An error occurred while performing Bland-Altman analysis: {e}")        

    def analyze_segments(self):
        """
        Perform statistical analysis on each segment of 4000 indices.
        """
        if self.data is None:
            print("Data not loaded. Please load the data first.")
            return

        try:
            # Use relative pressure data for analysis
            relative_pressure = self.data.iloc[:, 2].astype(float)  # 3rd column (Relative Pressure)

            # Determine the number of segments (each of 4000 indices)
            segment_length = 4000
            num_segments = len(relative_pressure) // segment_length

            analysis_results = []

            # Analyze each segment
            for i in range(num_segments):
                start_index = i * segment_length + 2000  # Start from the 2000th index within each segment
                end_index = (i + 1) * segment_length
                cycle_pressure = relative_pressure.iloc[start_index:end_index].reset_index(drop=True)


                # Descriptive statistics
                mean_val = cycle_pressure.mean()
                median_val = cycle_pressure.median()
                std_dev = cycle_pressure.std()
                variance = cycle_pressure.var()
                min_val = cycle_pressure.min()
                max_val = cycle_pressure.max()

                # Linear regression analysis
                X = np.arange(len(cycle_pressure)).reshape(-1, 1)
                y = cycle_pressure.values
                reg = LinearRegression().fit(X, y)
                slope = reg.coef_[0]
                intercept = reg.intercept_
                r_squared = reg.score(X, y)

                # Save results for each segment
                analysis_results.append({
                    'Segment': i + 1,
                    'Mean': mean_val,
                    'Median': median_val,
                    'Std Dev': std_dev,
                    'Variance': variance,
                    'Min': min_val,
                    'Max': max_val,
                    'Slope': slope,
                    'Intercept': intercept,
                    'R^2': r_squared
                })

            # Convert results to DataFrame and save to CSV
            results_df = pd.DataFrame(analysis_results)
            filename = self.generate_filename('segment_analysis_results','csv')
            results_df.to_csv(filename, index=False)
            print("Statistical analysis results saved to 'segment_analysis_results.csv'.")

        except Exception as e:
            print(f"An error occurred while analyzing the data: {e}")

    def plot_bar_plot_after_2000(self):
        """
        Plot a bar plot with mean and standard deviation for data after the 2000th index 
        in each segment of 4000 indices.
        """
        if self.data is None:
            print("Data not loaded. Please load the data first.")
            return

        try:
            # Use relative pressure data for analysis
            relative_pressure = self.data.iloc[:, 2].astype(float)  # 3rd column (Relative Pressure)

            # Determine the number of segments (each of 4000 indices)
            segment_length = 4000
            num_segments = len(relative_pressure) // segment_length

            means = []
            std_devs = []

            # Calculate mean and standard deviation for data after the 2000th index
            for i in range(num_segments):
                start_index = i * segment_length + 2000  # Start from the 2000th index within each segment
                end_index = (i + 1) * segment_length
                cycle_pressure = relative_pressure.iloc[start_index:end_index]

                # Calculate mean and standard deviation
                mean_val = cycle_pressure.mean()
                std_dev = cycle_pressure.std()

                means.append(mean_val)
                std_devs.append(std_dev)

            # Prepare for bar plot
            x_positions = np.arange(1, num_segments + 1)

            plt.figure(figsize=(10, 6))

            # Plot bar with error bars
            plt.bar(x_positions, means, yerr=std_devs, capsize=5, color='skyblue', edgecolor='black', label='Mean ± SD')

            # Connect the bars with lines
            plt.plot(x_positions, means, color='red', linestyle='-', marker='o', label='Mean Line')

            plt.title('Statistics at Steady State Pressure', fontsize=14)
            plt.xlabel('Repeats', fontsize=14)
            plt.ylabel('Relative Pressure (mmHg)', fontsize=14)
            plt.xticks(x_positions, [f'{i + 1}' for i in range(num_segments)], fontsize=12)
            plt.yticks(fontsize=14)
            plt.grid(axis='y')
            plt.legend()

            filename = self.generate_filename('bar_plot_after_2000','png')
            self.save_plot(filename)
            plt.close()  # Close the plot after saving

        except Exception as e:
            print(f"An error occurred while plotting the bar plot: {e}")

    def bland_altman_analysis_after_2000(self):
        """
        Perform Bland-Altman analysis on data between indices 2000 and 4000 to prove stability
        and accuracy in maintaining a steady state at 10 mmHg.
        """
        if self.data is None:
            print("Data not loaded. Please load the data first.")
            return

        try:
            # Use relative pressure data for analysis
            relative_pressure = self.data.iloc[:, 2].astype(float)  # 3rd column (Relative Pressure)

            # Determine the number of segments (each of 4000 indices)
            segment_length = 4000
            num_segments = len(relative_pressure) // segment_length

            # Target pressure for steady state
            target_pressure = 10.0

            analysis_results = []

            # Perform Bland-Altman analysis on each segment between indices 2000 and 4000
            for i in range(num_segments):
                start_index = i * segment_length + 2000  # Start from the 2000th index within each segment
                end_index = (i + 1) * segment_length
                cycle_pressure = relative_pressure.iloc[start_index:end_index].reset_index(drop=True)

                # Calculate differences from the target pressure
                differences = cycle_pressure - target_pressure

                # Bland-Altman statistics
                mean_diff = np.mean(differences)
                std_diff = np.std(differences, ddof=1)
                loa = 1.96 * std_diff
                upper_loa = mean_diff + loa
                lower_loa = mean_diff - loa

                # Save results for each segment
                analysis_results.append({
                    'Segment': i + 1,
                    'Mean Difference': mean_diff,
                    'Std Dev of Difference': std_diff,
                    'Upper LoA': upper_loa,
                    'Lower LoA': lower_loa
                })

                # Plot Bland-Altman for each segment
                plt.figure(figsize=(10, 6))
                plt.scatter(cycle_pressure.index, differences, color='black', s=10)  # Black dots
                plt.axhline(mean_diff, color='r', linestyle='-', label=f'Mean diff: {mean_diff:.2f}')
                plt.axhline(upper_loa, color='g', linestyle='--', label=f'Upper LoA: {upper_loa:.2f}')
                plt.axhline(lower_loa, color='b', linestyle='--', label=f'Lower LoA: {lower_loa:.2f}')

                # Annotate the plot with the values
                plt.text(max(cycle_pressure.index), mean_diff, f'{mean_diff:.2f}', color='r', ha='left', va='bottom', fontsize=10)
                plt.text(max(cycle_pressure.index), upper_loa, f'{upper_loa:.2f}', color='g', ha='left', va='bottom', fontsize=10)
                plt.text(max(cycle_pressure.index), lower_loa, f'{lower_loa:.2f}', color='b', ha='left', va='top', fontsize=10)

                plt.title(f'Bland-Altman Plot for Segment {i + 1}', fontsize=14)
                plt.xlabel('Index (Relative to 2000-4000)', fontsize=14)
                plt.ylabel('Difference from Target (mmHg)', fontsize=14)
                plt.xticks(fontsize=14)
                plt.yticks(fontsize=14)
                plt.grid(True)
                plt.legend()

                filename = self.generate_filename(f'bland_altman_segment_{i + 1}','png')
                self.save_plot(filename)
                plt.close()  # Close the plot after saving

            # Convert results to DataFrame and save to CSV
            results_df = pd.DataFrame(analysis_results)
            filename = self.generate_filename(f'bland_altman_segment','csv')
            results_df.to_csv(filename, index=False)
            print("Bland-Altman analysis results saved to 'bland_altman_analysis_results.csv'.")

        except Exception as e:
            print(f"An error occurred while performing Bland-Altman analysis: {e}")

    def filter_data(self):
        """
        Filter the data to show motor speed between 100 and 250 and select the last 20 data points for each motor speed.
        """
        if self.data is None:
            print("Data not loaded. Please load the data first.")
            return

        try:
            # Ensure correct column names after inspecting output
            relative_pressure = self.data['Relative Pressure (mmHg)']  # Update with correct column name
            motor_speed_column = [col for col in self.data.columns if 'Motor Speed' in str(col)]
            if motor_speed_column:
                motor_speed = self.data[motor_speed_column[0]].astype(float)
            else:
                raise KeyError("Motor Speed column not found.")

            # Filter for motor speeds between 100 and 250
            filtered_data = self.data[(motor_speed >= 100) & (motor_speed <= 250)]
            
            # Select the last 20 data points for each motor speed
            last_20_per_speed = filtered_data.groupby(motor_speed_column[0]).tail(20)
            
            print("Filtered data:")
            print(last_20_per_speed)
            
            return last_20_per_speed

        except KeyError as e:
            print(f"Error: {e}")
            print("Available columns:", self.data.columns)
        except Exception as e:
            print(f"An error occurred while filtering the data: {e}")

    def plot_relative_pressure_vs_speed(self):
        """
        Plot a scatter plot of relative pressure vs. motor speed and save the plot.
        """
        last_20_per_speed = self.filter_data()
        if last_20_per_speed is None:
            print("Filtered data not available. Please check the data.")
            return

        try:
            # Use filtered data for plotting
            relative_pressure = last_20_per_speed['Relative Pressure (mmHg)'].astype(float)
            motor_speed = last_20_per_speed['Motor Speed (PWM)'].astype(float)

            plt.figure(figsize=(10, 6))
            plt.scatter(motor_speed, relative_pressure, color='orange', marker='.', alpha=0.7, s=10)  # Smaller blue dots

            # Annotate closest integer relative pressures
            max_pressure = int(relative_pressure.max())
            min_pressure = int(relative_pressure.min())

            for pressure in range(min_pressure, max_pressure + 1):
                # Find the closest index to the integer pressure value
                differences = (relative_pressure - pressure).abs()
                closest_index = differences.idxmin()

                # Ensure closest_index is a valid index
                if closest_index in relative_pressure.index:
                    speed_for_pressure = motor_speed.loc[closest_index]
                    pressure_value = relative_pressure.loc[closest_index]
                    # Add text annotation for the closest integer relative pressure
                    plt.text(speed_for_pressure, pressure_value, f'{pressure} mmHg @ {int(speed_for_pressure)}', fontsize=9, color='black')

            plt.title('Relative Pressure vs Motor Speed', fontsize=14)
            plt.xlabel('Motor Speed (PWM)', fontsize=14)
            plt.ylabel('Relative Pressure (mmHg)', fontsize=14)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            plt.grid(True)

            filename = self.generate_filename('relative_pressure_vs_speed_filtered','png')
            self.save_plot(filename)
            plt.close()  # Close the plot after saving

        except KeyError as e:
            print(f"Error: {e}")
            print("Available columns:", self.data.columns)
        except Exception as e:
            print(f"An error occurred while plotting the data: {e}")
        
    def plot_average_pressure_vs_speed(self):
        """
        Plot average relative pressure for each speed with standard deviation markers and regression line, and save the plot.
        """
        last_20_per_speed = self.filter_data()
        if last_20_per_speed is None:
            print("Filtered data not available. Please check the data.")
            return

        try:
            # Calculate the mean and standard deviation for each motor speed
            avg_data = last_20_per_speed.groupby('Motor Speed (PWM)')['Relative Pressure (mmHg)'].agg(['mean', 'std']).reset_index()

            plt.figure(figsize=(10, 6))
            plt.errorbar(avg_data['Motor Speed (PWM)'], avg_data['mean'], yerr=avg_data['std'], fmt='.', ecolor='blue', capsize=5, color='blue', alpha=0.7, markersize=5)

            # Perform linear regression on the average data
            X_avg = avg_data['Motor Speed (PWM)'].values.reshape(-1, 1)
            y_avg = avg_data['mean'].values
            reg = LinearRegression().fit(X_avg, y_avg)
            y_pred_avg = reg.predict(X_avg)

            # Draw regression line in red
            plt.plot(avg_data['Motor Speed (PWM)'], y_pred_avg, color='red', linewidth=2)

            # Calculate R^2
            r_squared_avg = reg.score(X_avg, y_avg)

            # Display regression equation and R^2 value
            equation_text_avg = f'y = {reg.coef_[0]:.4f}x + {reg.intercept_:.4f}\n$R^2$ = {r_squared_avg:.4f}'
            plt.text(0.05, 0.95, equation_text_avg, transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')

            plt.title('Average Relative Pressure vs Motor Speed with Regression', fontsize=14)
            plt.xlabel('Motor Speed (PWM)', fontsize=14)
            plt.ylabel('Average Relative Pressure (mmHg)', fontsize=14)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            plt.grid(True)

            filename = self.generate_filename('average_relative_pressure_vs_speed','png')
            self.save_plot(filename)
            plt.close()  # Close the plot after saving

        except KeyError as e:
            print(f"Error: {e}")
            print("Available columns:", self.data.columns)
        except Exception as e:
            print(f"An error occurred while plotting the data: {e}")

    def bland_altman_plot(self):
        """
        Draw a Bland-Altman plot for each 20 speed data and save it. 
        Calculate mean difference, standard deviation, standard error, and upper/lower 95% CI for LoA.
        """
        last_20_per_speed = self.filter_data()
        if last_20_per_speed is None:
            print("Filtered data not available. Please check the data.")
            return

        try:
            df = pd.read_excel(file_path)
            speed_column = 'Motor Speed (PWM)'

            # Initialize summary data
            summary_data = []

            for speed in range(101, 251):
                # Find the first occurrence of the current speed
                speed_index = df[df[speed_column] == speed].index.min()
                
                print(f"Processing speed: {speed}, Found index: {speed_index}")
                
                if pd.notna(speed_index) and speed_index >= 10:
                    # Calculate statistics for speed-1 based on 10 previous points
                    previous_speed = speed - 1
                    pressure_values = df.iloc[speed_index - 10:speed_index, df.columns.get_loc('Relative Pressure (mmHg)')]
                    
                    print(f"Pressure values for speed {previous_speed}: {pressure_values}")
                    
                    if len(pressure_values) == 10:  # Ensure there are exactly 10 previous points
                        mean_pressure = pressure_values.mean()
                        std_dev_pressure = pressure_values.std()
                        std_err_pressure = stats.sem(pressure_values)
                        ci95_loA_upper = mean_pressure + 1.96 * std_dev_pressure
                        ci95_loA_lower = mean_pressure - 1.96 * std_dev_pressure
                        
                        # Append results to summary data
                        summary_data.append([speed, mean_pressure, std_dev_pressure, std_err_pressure, ci95_loA_upper, ci95_loA_lower])
            # Convert summary data to DataFrame if it contains data
            if summary_data:
                summary_df = pd.DataFrame(summary_data, columns=['Speed', 'Mean Pressure', 'Std Dev', 'Std Error', 'Upper 95% CI LoA', 'Lower 95% CI LoA'])
            else:
                print("No valid data found for plotting.")
                exit()

            mean = summary_df['Mean Pressure']
            diff = summary_df['Std Dev']
            mean_diff = np.mean(diff)
            ci95 = 1.96 * np.std(diff)

            # Save the calculations to a text file
            with open(self.generate_filename('bland-altman_tats','txt'), 'w') as f:
                f.write(f"Mean difference: {mean_pressure:.4f}\n")
                f.write(f"Standard deviation of differences: {std_dev_pressure:.4f}\n")
                f.write(f"Standard error: {std_err_pressure:.4f}\n")
                f.write(f"Upper 95% LoA: {ci95_loA_upper:.4f}\n")
                f.write(f"Lower 95% LoA: {ci95_loA_lower:.4f}\n")

            # Bland-Altman plot
            plt.figure(figsize=(10, 6))
            mean = summary_df['Mean Pressure']
            diff = summary_df['Std Dev']
            mean_diff = np.mean(diff)
            ci95 = 1.96 * np.std(diff)

            plt.scatter(mean, diff, color='black', s=10)  # Smaller black points
            plt.axhline(mean_diff, color='blue', linestyle='-', label=f'Mean Difference: {mean_diff:.4f}')
            plt.axhline(mean_diff + ci95, color='green', linestyle='--', label=f'Upper 95% CI: {mean_diff + ci95:.4f}')
            plt.axhline(mean_diff - ci95, color='red', linestyle='-.', label=f'Lower 95% CI: {mean_diff - ci95:.4f}')

            # Place text annotations at the maximum of the mean value
            plt.text(max(mean)+0.9, mean_diff, f'Mean Diff: {mean_diff:.4f}', color='blue', ha='right', va='bottom')
            plt.text(max(mean)+0.9, mean_diff + ci95, f'Upper 95% CI: {mean_diff + ci95:.4f}', color='green', ha='right', va='bottom')
            plt.text(max(mean)+0.9, mean_diff - ci95, f'Lower 95% CI: {mean_diff - ci95:.4f}', color='red', ha='right', va='bottom')

            plt.title('Bland-Altman Plot', fontsize=14)
            plt.xlabel('Mean of Pressures (mmHg)', fontsize=14)
            plt.ylabel('Mean Difference (mmHg)', fontsize=14)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            plt.grid(True)
            plt.legend()

            filename = self.generate_filename('bland_altman_plot','png')
            self.save_plot(filename)
            plt.close()  # Close the plot after saving

        except Exception as e:
            print(f"An error occurred while plotting the Bland-Altman data: {e}")

    def plot_response_time(self):
        """
        Plot a scatter plot of response time vs. starting pressures and save the plot.
        """
        if self.data is None:
            print("Data not loaded. Please load the data first.")
            return

        # Extract data specific to the response time plot
        self.starting_pressures = self.data.iloc[:, 1]  # Second column
        self.response_time = self.data.iloc[:, 2]       # Third column

        plt.figure(figsize=(10, 6))
        plt.scatter(self.starting_pressures, self.response_time, color='blue', alpha=0.7)

        # Set the title and labels with increased fontsize
        plt.title('Response Time to Steady State vs Starting Pressures', fontsize=14)
        plt.xlabel('Starting Pressures (mmHg)', fontsize=14)
        plt.ylabel('Response Time to Steady State (s)', fontsize=14)
        plt.xticks(fontsize=14) # Increase tick label size
        plt.yticks(fontsize=14)
        plt.grid(True)

        filename = self.generate_filename('response_time_plot','png')
        self.save_plot(filename)
        plt.close()  # Close the plot after saving

    def plot_box_and_whisker_with_mean(self):
        """
        Plot a box and whisker plot with mean, minimum, maximum, and outliers for elasticity measured 
        between indices 2000 and 4000 for each segment.
        """
        if self.data is None:
            print("Data not loaded. Please load the data first.")
            return

        try:
            # Use elasticity data for analysis, assuming 'Elasticity' is the 3rd column
            elasticity_data = self.data.iloc[:, 2].astype(float)  # 3rd column (Elasticity)

            # Determine the number of segments (each of 4000 indices)
            segment_length = 4000
            num_segments = len(elasticity_data) // segment_length

            # Prepare data for each segment between indices 2000 and 4000
            segment_data = []

            for i in range(num_segments):
                start_index = i * segment_length + 2000  # Start from the 2000th index within each segment
                end_index = (i + 1) * segment_length
                segment_elasticity = elasticity_data.iloc[start_index:end_index].reset_index(drop=True)
                segment_data.append(segment_elasticity)

            # Create box and whisker plot
            plt.figure(figsize=(10, 6))
            box = plt.boxplot(segment_data, patch_artist=True, showfliers=True, meanline=False, showmeans=True)

            # Customize the plot appearance
            plt.title('Suction Regulation Stability Test Over 10 Repeated Cycles', fontsize=14)
            plt.xlabel('Cycles', fontsize=14)
            plt.ylabel('Relative Pressure (mmHg)', fontsize=14)
            plt.xticks(range(1, num_segments + 1), [f'{i + 1}' for i in range(num_segments)], fontsize=12)
            plt.yticks(fontsize=14)
            plt.grid(axis='y')

            # Customize colors for the boxes
            for patch, color in zip(box['boxes'], plt.cm.viridis(np.linspace(0, 1, num_segments))):
                patch.set_facecolor(color)

            # Create custom legend
            legend_elements = [
                Patch(facecolor='skyblue', edgecolor='black', label='Interquartile Range (IQR)'),
                Line2D([0], [0], color='black', label='Median'),
                Line2D([0], [0], color='red', linestyle='--', label='Mean'),
                Line2D([0], [0], marker='o', color='w', label='Outliers', markerfacecolor='black', markersize=6)
            ]

            plt.legend(handles=legend_elements, loc='upper right', fontsize=10)

            # Show the plot
            filename = self.generate_filename('box_whisker_plot_relative_pressures_segments_2000_4000','png')
            self.save_plot(filename)
            plt.close()  # Close the plot after saving

        except KeyError as e:
            print(f"Error: {e}")
            print("Available columns:", self.data.columns)
        except Exception as e:
            print(f"An error occurred while plotting the box and whisker plot: {e}")


    def run(self):
        """
        Run the data loading and plotting functions.
        """
        self.load_data()

        # self.plot_response_time()                 # Plot response time to reach steady state

        # self.plot_relative_pressure_vs_speed()    # Plot relative pressure vs speed data
        # self.plot_average_pressure_vs_speed()     # Plot relative pressure vs speed data
        # self.bland_altman_plot()                  # Plot relative pressure vs speed data

        # self.plot_relative_pressure_cycles()       # /Plot relative pressure vs time
        # self.analyze_segments()
        # self.bland_altman_analysis_after_2000()
        # self.plot_bar_plot_after_2000()
        # self.plot_box_and_whisker_with_mean()

        self.plot_relative_pressure_cycles_t('one')
        # self.bland_altman_analysis_cycle()
        
        
# Usage
file_path = r'C:\Users\BMC-Gram\Desktop\BM\BM\Data\2024-08-29_response_1\2024-08-29_17-55-16_pressure_data_mmHg.csv'


plotter = DataPlotter(file_path)
plotter.run()