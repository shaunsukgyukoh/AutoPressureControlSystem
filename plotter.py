# python -m pip install --upgrade pip
# pip3 install pyserial matplotlib keyboard pandas numpy scipy openpyxl

import serial
import serial.tools.list_ports
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import subprocess
import sys
import keyboard
import time
import pandas as pd
import numpy as np
from collections import deque
from datetime import datetime
import os
from scipy import stats

class ArduinoDataLogger:
    # def __init__(self, baud_rate=115200, timeout=0.1, pressure_bias=3, max_points=100):
    def __init__(self, baud_rate=115200, timeout=0.1, pressure_bias=6.508, max_points=100):
    
        self.serial_port = self.find_arduino()
        if not self.serial_port:
            raise Exception("Arduino not found. Please connect the Arduino and try again.")

        self.ser = serial.Serial(self.serial_port, baud_rate, timeout=timeout)
        self.pressure_bias = pressure_bias
        self.max_points = max_points  # Maximum points to display

        # Use deque for circular buffer behavior
        self.pressure_data = deque(maxlen=max_points)
        self.relative_pressure_data = deque(maxlen=max_points)
        self.temperature_data = deque(maxlen=max_points)
        self.time_data = deque(maxlen=max_points)

        # Lists to store all data
        self.pressure_data_all = []
        self.relative_pressure_data_all = []
        self.temperature_data_all = []
        self.time_data_all = []
        self.air_speed_data_all = []
        self.suction_speed_data_all = []

        # Data storage for analysis
        self.data_all = []
        self.data_all_SS = []

        self.start_time = time.time()
        self.system_start_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # Log start time
        self.initial_pressure = None # in hPA
        # self.unit_conversion_factor = 1.0  # Initially in hPa
        # self.unit = "hPa"  # Track current unit
        self.unit_conversion_factor = 0.750062  # Conversion from hPa to mmHg
        self.unit = "mmHg"  # Track current unit
        self.air_speed = 0
        self.suction_speed = 0

        self.motor_control_enabled = False  # Initialize motor control as enabled
        self.air_manual_control_enabled = False  # Initialize motor control as enabled
        self.suction_manual_control_enabled = False  # Initialize motor control as enabled
        self.last_f4_press_time = 0  # Initialize last F4 press 

        # Set up the figure and axes for pressure and temperature
        # self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(3, 1, figsize=(10, 12))
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(10, 8))
        self.line1, = self.ax1.plot([], [], 'g-', label='Absolute Pressure (mmHg)')
        self.line2, = self.ax2.plot([], [], 'b-', label='Relative Pressure (mmHg)')
        # self.line3, = self.ax3.plot([], [], 'b-', label='Temperature (°C)')

        self.raw_filename = self.generate_filename('raw_serial_data_log', 'txt')
        self.recent_filename = self.generate_filename('recent_data', 'csv')

        self.experiment_running = True

        # Experiment mode for 10 step incremental stabilization
        self.motor_experiment_increment_repeat_auto = False  
        self.loopCount = 0
        self.repeat = 0
        
        # Experiment mode for finding pressure for motor speed
        self.motor_experiment_find_speed_per_pressure_auto = False
        self.speedForPressure =[]
        self.pressuresAtSteadyState = []

        # Experiment mode for finding response time to become steady state from given pressure
        self.motor_experiment_find_response_time_from_each_pressure = False
        self.air_speed_for_each_relP = [117, 132, 146, 160, 174, 190, 205, 224, 241, 250]
        # self.air_speed_for_each_relP = [250, 205, 231]
        self.air_speed_for_each_relP_index = 0
        self.moving_avg_init = 0
        self.response_time_init = 0
        self.response_time_final = 0

        # Experiment mode for finding response time to become steady state from given pressure for manual
        self.motor_experiment_find_response_time_from_each_pressure_manual = False

        # Adjust spacing between subplots
        self.fig.subplots_adjust(hspace=0.4)  # Increase hspace to create more room

        self.iah_lines = []
        self.configure_plot()

    @staticmethod
    def find_arduino():
        ports = serial.tools.list_ports.comports()
        for port in ports:
            if 'Arduino' in port.description or 'CH340' in port.description:
                return port.device
        return None

    def save_raw_data(self, raw_data):
        with open(self.raw_filename, 'a') as f:
            f.write(raw_data + '\n')

    def save_experiment_data_to_csv(self):
        """Save experiment data to a CSV file."""
        if not self.data_all:
            print("No data to save.")
            return

        # Convert the data to a DataFrame
        df = pd.DataFrame(self.data_all)

        # Define the filename with a fixed name
        filename = self.generate_filename('response_time_experiment_data', 'csv')
        df.to_csv(filename, index=False)
        print(f"Experiment data saved to {filename}")

    # def save_data_to_csv(self):
        
    #     """Save all collected data to a CSV file, appending if the file already exists."""
    #     if not self.data_all:
    #         print("No data to save.")
    #         return

    #     # Combine all data into a single DataFrame
    #     combined_data = pd.DataFrame()

    #     for entry in self.data_all:
    #         # Check that each entry contains all required keys
    #         if 'Speed' in entry and 'Pressure' in entry and 'Relative Pressure' in entry and 'Temperature' in entry:
    #             df_entry = pd.DataFrame({
    #                 'Speed': entry['Speed'],
    #                 'Pressure': entry['Pressure'],
    #                 'Relative Pressure': entry['Relative Pressure'],
    #                 'Temperature': entry['Temperature']
    #             })
    #             combined_data = pd.concat([combined_data, df_entry], ignore_index=True)
    #         else:
    #             print(f"Incomplete data entry found and skipped: {entry}")

    #     # Define the filename with a fixed name
    #     filename = 'experiment_data.csv'
    #     combined_data.to_csv(filename, index=False, mode='a', header=not os.path.exists(filename))
    #     print(f"All data saved to {filename}")
    #     # # Check if the file exists
    #     # file_exists = os.path.isfile(filename)

    #     # # Save or append the combined data to CSV
    #     # with open(filename, 'a' if file_exists else 'w', newline='') as f:
    #     #     combined_data.to_csv(f, index=False, header=not file_exists)
        

    def perform_anova_analysis(self):
        # Check if the file exists before proceeding
        if not os.path.exists(self.recent_filename):
            print(f"File {self.recent_filename} does not exist. Cannot perform ANOVA analysis.")
            return

        try:
            # Load the data and check for required columns
            df = pd.read_csv(self.recent_filename, encoding='cp949')
            if 'Speed' not in df.columns or 'Relative Pressure' not in df.columns:
                raise KeyError("'Speed' or 'Relative Pressure' column is missing in the CSV file.")

            # ANOVA Analysis
            speeds = df['Speed'].unique()
            data_by_speed = [df[df['Speed'] == speed]['Relative Pressure'] for speed in speeds]

            # Check if data_by_speed is not empty and contains valid data
            if any(len(data) == 0 for data in data_by_speed):
                raise ValueError("Some speed groups are empty. Check the data for missing or incomplete entries.")

            f_value, p_value = stats.f_oneway(*data_by_speed)
            print(f"ANOVA F-value: {f_value}, P-value: {p_value}")

            # Plot ANOVA results
            plt.figure(figsize=(8, 6))
            plt.title('ANOVA Analysis of Relative Pressure by Speed')
            plt.bar(speeds, [np.mean(data) for data in data_by_speed], yerr=[np.std(data) for data in data_by_speed])
            plt.xlabel('Speed')
            plt.ylabel('Mean Relative Pressure (mmHg)')
            plt.grid(True)

            # Save ANOVA plot
            anova_filename = self.generate_filename('ANOVA_with_recent_data', 'png')
            plt.savefig(anova_filename)
            print(f"ANOVA plot saved as {anova_filename}")
            plt.close()

        except KeyError as e:
            print(f"Key error: {e}. Check if the CSV file contains the correct columns.")
        except ValueError as e:
            print(f"Value error: {e}. Check the data for missing or incomplete entries.")
        except Exception as e:
            print(f"Unexpected error during ANOVA analysis: {e}.")

    def perform_bland_altman_analysis(self):
        # Check if the file exists before proceeding
        if not os.path.exists(self.recent_filename):
            print(f"File {self.recent_filename} does not exist. Cannot perform Bland-Altman analysis.")
            return

        try:
            # Load the data and check for required columns
            df = pd.read_csv(self.recent_filename, encoding='cp949')
            if 'Elapsed Time (s)' not in df.columns or 'Relative Pressure' not in df.columns:
                raise KeyError("'Elapsed Time (s)' or 'Relative Pressure' column is missing in the CSV file.")

            # Ensure that the DataFrame has enough data to be divided into 10 repeats
            total_rows = len(df)
            # if total_rows < 2000: # 4000
            #     # print(f"Not enough data in {filename}. Found {total_rows} rows, but need at least 4000.")
            #     # return

            # Divide the DataFrame into 10 chunks for analysis
            chunk_size = total_rows // 10 # 10
            data_chunks = [df.iloc[i * chunk_size:(i + 1) * chunk_size] for i in range(10)] # 10

            # Bland-Altman Analysis
            for i in range(len(data_chunks) - 1):
                # Drop NaN values and ensure data chunks have valid data
                chunk1 = data_chunks[i].dropna(subset=['Relative Pressure'])
                chunk2 = data_chunks[i + 1].dropna(subset=['Relative Pressure'])

                if chunk1.empty or chunk2.empty:
                    print(f"No valid data found for chunk {i + 1} or chunk {i + 2}. Skipping.")
                    continue
                # Calculate the mean and difference of relative pressures between two consecutive chunks
                mean_rel_pressure = np.mean([data_chunks[i]['Relative Pressure'].values, data_chunks[i + 1]['Relative Pressure'].values], axis=0)
                diff_rel_pressure = data_chunks[i]['Relative Pressure'].values - data_chunks[i + 1]['Relative Pressure'].values

                # Calculate mean and standard deviation of differences
                mean_diff = np.mean(diff_rel_pressure)
                std_diff = np.std(diff_rel_pressure)

                # Calculate 95% CI limits
                loa_upper = mean_diff + 1.96 * std_diff
                loa_lower = mean_diff - 1.96 * std_diff

                # Plot Bland-Altman
                plt.figure(figsize=(8, 6))
                plt.scatter(mean_rel_pressure, diff_rel_pressure)
                plt.axhline(mean_diff, color='red', linestyle='--', label='Mean Difference')
                plt.axhline(loa_upper, color='blue', linestyle='--', label='Upper LOA (95% CI)')
                plt.axhline(loa_lower, color='blue', linestyle='--', label='Lower LOA (95% CI)')
                plt.title(f'Bland-Altman Plot: Time Segment {i + 1} vs {i + 2}')
                plt.xlabel('Mean Relative Pressure (mmHg)')
                plt.ylabel('Difference in Relative Pressure (mmHg)')

                # Add text annotations for 95% CI LOA
                plt.text(np.max(mean_rel_pressure), loa_upper, f'Upper LOA: {loa_upper:.2f}', color='blue', va='bottom', ha='left')
                plt.text(np.max(mean_rel_pressure), loa_lower, f'Lower LOA: {loa_lower:.2f}', color='blue', va='top', ha='left')

                # Save Bland-Altman plot
                bland_altman_filename = self.generate_filename(f'bland_altman_plot_relative_pressure_time_segment_{i + 1}_vs_{i + 2}', 'png')
                plt.savefig(bland_altman_filename)
                print(f"Bland-Altman plot for time segments {i + 1} vs {i + 2} saved as {bland_altman_filename}")
                plt.close()

        except KeyError as e:
            print(f"Key error: {e}. Check if the CSV file contains the correct columns.")
        except ValueError as e:
            print(f"Value error: {e}. Check the data for missing or incomplete entries.")
        except Exception as e:
            print(f"Unexpected error during Bland-Altman analysis: {e}.")


    def check_for_esc(self):
        """Checks for ESC key press to stop the experiment."""
        if keyboard.is_pressed('esc'):
            print("ESC key pressed. Stopping experiment...")
            self.experiment_running = False

    def configure_plot(self):
        self.ax1.set_title('Live Absolute Pressure Data')
        self.ax1.set_xlabel('Time (s)')
        # self.ax1.set_ylabel('Pressure (hPa)')
        self.ax1.set_ylabel('Pressure (mmHg)')
        # self.ax1.set_ylim(800, 1200)  # Absolute pressure range in hPa
        # self.ax1.set_ylim(953, 1073)  # Absolute pressure range in hPa
        # self.ax1.set_ylim(100, 2000)  # Absolute pressure range in hPa
        self.ax1.set_ylim(700, 820)  # Absolute pressure range in mmHg
        self.ax1.legend()

        self.ax2.set_title('Live Relative Pressure Data')
        self.ax2.set_xlabel('Time (s)')
        # self.ax2.set_ylabel('Pressure (hPa)')
        self.ax2.set_ylabel('Pressure (mmHg)')
        self.ax2.set_ylim(-5, 50)  # Relative pressure range
        self.ax2.legend()

        # self.ax3.set_title('Live Temperature Data')
        # self.ax3.set_xlabel('Time (s)')
        # self.ax3.set_ylabel('Temperature (°C)')
        # self.ax3.set_ylim(0, 50)
        # self.ax3.legend()

        self.ax2.axhline(y=12, color='g', linestyle='--', linewidth=0.5)
        self.ax2.text(0.02, 12, 'IAH Grade I', color='g', fontsize=8, va='bottom')
        self.ax2.axhline(y=16, color='y', linestyle='--', linewidth=0.5)
        self.ax2.text(0.02, 16, 'IAH Grade II', color='y', fontsize=8, va='bottom')
        self.ax2.axhline(y=21, color='orange', linestyle='--', linewidth=0.5)
        self.ax2.text(0.02, 21, 'IAH Grade III (ACS)', color='orange', fontsize=8, va='bottom')
        self.ax2.axhline(y=25, color='r', linestyle='--', linewidth=0.5)
        self.ax2.text(0.02, 25, 'IAH Grade IV (ACS)', color='r', fontsize=8, va='bottom')

        # self.add_iah_grade_lines()

    def init_plot(self):
        # self.ax1.set_xlim(0, 10)
        self.line1.set_data([], [])

        self.ax2.set_xlim(0, 10)
        self.line2.set_data([], [])

        # self.ax3.set_xlim(0, 10)
        # self.line3.set_data([], [])

        # return self.line1, self.line2, self.line3
        return self.line1, self.line2

    def update_plot(self, frame):
        # Introduce a delay to ignore rapid consecutive F4 key presses
        current_time = time.time() # motor turns at near 90, wiggle at 75
        if keyboard.is_pressed('up') and (current_time - self.last_f4_press_time) > 0.5:
            self.last_f4_press_time = current_time  # Update the last press time
            if (self.air_manual_control_enabled):
                self.air_speed = 100 if self.air_speed == 0 else self.air_speed
                self.air_speed = min(self.air_speed + 15, 250)
                self.ser.write(f"A_SPEED {self.air_speed}\n".encode())
                print(f"Motor speed increased to {self.air_speed}")
            else:
                self.ser.write(f"A_SPEED 0\n".encode())

        if keyboard.is_pressed('down') and (current_time - self.last_f4_press_time) > 0.5:
            self.last_f4_press_time = current_time  # Update the last press time
            if (self.air_manual_control_enabled):
                self.air_speed = 100 if self.air_speed == 0 else self.air_speed
                self.air_speed = max(self.air_speed - 15, 100)
                self.ser.write(f"A_SPEED {self.air_speed}\n".encode())
                print(f"Motor speed decreased to {self.air_speed}")
            else:
                self.ser.write(f"A_SPEED 0\n".encode())

        if keyboard.is_pressed('right') and (current_time - self.last_f4_press_time) > 0.5:
            self.last_f4_press_time = current_time  # Update the last press time
            if (self.suction_manual_control_enabled):
                self.suction_speed = 100 if self.suction_speed == 0 else self.suction_speed
                self.suction_speed = min(self.suction_speed + 15, 250)
                self.ser.write(f"S_SPEED {self.suction_speed}\n".encode())
                print(f"Motor speed increased to {self.suction_speed}")
            else:
                self.ser.write(f"S_SPEED 0\n".encode())

        if keyboard.is_pressed('left') and (current_time - self.last_f4_press_time) > 0.5:
            self.last_f4_press_time = current_time  # Update the last press time
            if (self.suction_manual_control_enabled):
                self.suction_speed = 100 if self.suction_speed == 0 else self.suction_speed
                self.suction_speed = max(self.suction_speed - 15, 100)
                self.ser.write(f"S_SPEED {self.suction_speed}\n".encode())
                print(f"Motor speed decreased to {self.suction_speed}")
            else:
                self.ser.write(f"S_SPEED 0\n".encode())

        if keyboard.is_pressed('esc') and (current_time - self.last_f4_press_time) > 0.5:
            self.last_f4_press_time = current_time  # Update the last press time
            print("Esc key pressed. Saving data and exiting...")
            self.air_speed = 0
            self.ser.write(f"A_SPEED {self.air_speed}\n".encode())
            time.sleep(0.5)
            self.ser.write(b"AIR_OFF\n")
            time.sleep(0.5)
            self.suction_speed = 0
            time.sleep(0.5)
            self.ser.write(f"S_SPEED {self.suction_speed}\n".encode())
            self.ser.write(b"SUCTION_OFF\n")
            time.sleep(0.5)
            self.save_data_to_csv()
            self.save_SS_data_to_csv()
            self.save_final_plot()
            self.save_final_stat_plot()
            self.perform_bland_altman_analysis() 
            self.close_program()

        if keyboard.is_pressed('f1') and (current_time - self.last_f4_press_time) > 0.5:
            self.last_f4_press_time = current_time  # Update the last press time
            print("F1 key pressed. Resetting relative pressure...")
            if self.pressure_data:
                self.initial_pressure = self.pressure_data[-1] / 0.750062  # Reset relative pressure to current absolute pressure

        if keyboard.is_pressed('f2') and (current_time - self.last_f4_press_time) > 0.5:
            self.last_f4_press_time = current_time  # Update the last press time
            print("F2 key pressed. Toggling pressure units...")
            self.toggle_pressure_units()

        if keyboard.is_pressed('f3') and (current_time - self.last_f4_press_time) > 0.5:
            self.last_f4_press_time = current_time  # Update the last press time
            print("F3 key pressed. Saving recent data...")
            self.save_recent_data()

        if keyboard.is_pressed('f4') and (current_time - self.last_f4_press_time) > 0.5:
            self.motor_control_enabled = not self.motor_control_enabled
            self.last_f4_press_time = current_time  # Update the last press time
            if self.motor_control_enabled:
                print("F4 key pressed. Motor control enabled.")
                self.ser.write(b"MOTOR_ON\n")
            else:
                print("F4 key pressed. Motor control disabled.")
                self.ser.write(b"MOTOR_OFF\n")

        if keyboard.is_pressed('f5') and (current_time - self.last_f4_press_time) > 0.5:
            self.air_manual_control_enabled = not self.air_manual_control_enabled
            self.last_f4_press_time = current_time  # Update the last press time
            if self.air_manual_control_enabled:
                print("F5 key pressed. Air manual control enabled.")
                self.ser.write(b"AIR_ON\n")
            else:
                self.air_speed = 0
                print("F5 key pressed. Air manual control disabled.")
                self.ser.write(b"AIR_OFF\n")

        if keyboard.is_pressed('f6') and (current_time - self.last_f4_press_time) > 0.5:
            self.suction_manual_control_enabled = not self.suction_manual_control_enabled
            self.last_f4_press_time = current_time  # Update the last press time
            if self.suction_manual_control_enabled:
                print("F6 key pressed. Suction manual control enabled.")
                self.ser.write(b"SUCTION_ON\n")
            else:
                self.suction_speed = 0
                print("F6 key pressed. Suction manual control disabled.")
                self.ser.write(b"SUCTION_OFF\n")
        if keyboard.is_pressed('f7') and (current_time - self.last_f4_press_time) > 0.5:
            self.air_manual_control_enabled = True
            self.ser.write(b"AIR_ON\n")
            self.motor_control_enabled = False
            self.ser.write(b"MOTOR_OFF\n")
            self.last_f4_press_time = current_time  # Update the last press time
            self.motor_experiment_find_speed_per_pressure_auto = not self.motor_experiment_find_speed_per_pressure_auto

        if keyboard.is_pressed('f8') and (current_time - self.last_f4_press_time) > 0.5:
            # self.suction_manual_control_enabled = not self.suction_manual_control_enabled
            self.last_f4_press_time = current_time  # Update the last press time
            self.motor_experiment_find_response_time_from_each_pressure = not self.motor_experiment_find_response_time_from_each_pressure
        
        if keyboard.is_pressed('f9') and (current_time - self.last_f4_press_time) > 0.5:
            # self.suction_manual_control_enabled = not self.suction_manual_control_enabled
            self.last_f4_press_time = current_time  # Update the last press time
            self.motor_experiment_find_response_time_from_each_pressure_manual = not self.motor_experiment_find_response_time_from_each_pressure_manual


        try:
            data = self.ser.readline().decode('utf-8', errors='ignore').strip()
            self.save_raw_data(data)
            if data and len(data.split()) == 4:  # Ensure we have both pressure and temperature
                pressure, temperature, air_speed, suction_speed = self.parse_data(data)
  
                if pressure is not None and temperature is not None:
                    biased_pressure, relative_pressure, elapsed_time = self.log_data(pressure, temperature, air_speed, suction_speed) # return in mmHg
                    # relative_pressure *= self.unit_conversion_factor
                    # biased_pressure *= self.unit_conversion_factor
                    self.update_lines()
                    self.adjust_axes()

                    #  # Calibrate relative pressure to 0 at 760 mmHg before starting the test
                    # if self.loopCount == 0 and self.motor_experiment_find_response_time_from_each_pressure:
                    #     # Check if the pressure data is available to set the reference
                    #     if self.pressure_data:
                    #         # Find the current absolute pressure in mmHg
                    #         current_absolute_pressure = self.pressure_data[-1] * self.unit_conversion_factor

                    #         # Check if the absolute pressure is close to 760 mmHg
                    #         if abs(current_absolute_pressure - 760) < 0.5:  # Allow a small tolerance
                    #             self.initial_pressure = current_absolute_pressure
                    #             self.relative_pressure_data.clear()  # Clear existing relative pressure data
                    #             print(f"Calibrated relative pressure to 0 at {current_absolute_pressure:.2f} mmHg.")
                    #         else:
                    #             print(f"Waiting to calibrate at 760 mmHg. Current absolute pressure: {current_absolute_pressure:.2f} mmHg.")
                    #             return  # Wait until the system stabilizes around 760 mmHg
                    if self.motor_experiment_increment_repeat_auto:
                        if self.repeat > 9: #9
                            self.save_data_to_csv()
                            self.save_final_plot()
                            self.save_final_stat_plot()
                            self.perform_bland_altman_analysis()  # Perform Bland-Altman analysis
                            self.close_program()

                        elif self.repeat == 0 and self.loopCount == 200:
                            # Set initial reference pressure
                            if self.pressure_data:
                                self.initial_pressure = biased_pressure
                        
                        self.loopCount += 1

                        # Make sure to append all necessary data and use default values if any are missing
                        data_entry = {
                            'Elapsed Time (s)': datetime.now().strftime('%Y-%m-%d %H:%M:%S')-self.system_start_time,
                            'Absolute Pressure (mmHg)': [biased_pressure] if biased_pressure is not None else [0],
                            'Relative Pressure (mmHg)': [relative_pressure] if relative_pressure is not None else [0],
                            'Temperature (°C)': [temperature] if temperature is not None else [0],
                            'System Start Time': self.system_start_time,
                            'System Exit Time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),  # Log exit time
                            'Air Speed': [air_speed] if air_speed is not None else [0],
                            'Suction Speed': [suction_speed] if suction_speed is not None else [0]
                            }

                        self.data_all.append(data_entry)
    
                        if self.loopCount == 4000: # 4000 = 20 seconds * 10 times
                            self.save_recent_data()
                            self.repeat += 1
                            self.loopCount = 0
                            self.air_speed = 0 
                            self.ser.write(f"A_SPEED 0\n".encode())
                            self.suction_speed = 0
                            self.ser.write(f"S_SPEED 0\n".encode())   

                        elif self.loopCount % 400 == 0: # 20 seconds
                            self.save_recent_data()
                            if self.suction_manual_control_enabled:
                                self.air_speed = 250
                                print(f"Setting speed to {self.air_speed}...")
                                self.ser.write(f"A_SPEED {self.air_speed}\n".encode())

                                self.suction_speed = 100 if self.suction_speed == 0 else self.suction_speed
                                self.suction_speed = min(self.suction_speed + 15, 250)
                                print(f"Setting speed to {self.suction_speed}...")
                                self.ser.write(f"S_SPEED {self.suction_speed}\n".encode())

                            elif self.air_manual_control_enabled:
                                self.air_speed = 100 if self.air_speed == 0 else self.air_speed
                                self.air_speed = min(self.air_speed + 15, 250)
                                print(f"Setting speed to {self.air_speed}...")
                                self.ser.write(f"A_SPEED {self.air_speed}\n".encode())

                    elif self.motor_experiment_find_speed_per_pressure_auto:
                        self.loopCount += 1
                        if self.loopCount == 60: # 3 sec
                            self.air_speed = 100
                            self.ser.write(f"A_SPEED {self.air_speed}\n".encode())
                        elif self.loopCount == 9300:
                            # Convert the data to a DataFrame for easy manipulation
                            df = pd.DataFrame({'Pump Speed': self.speedForPressure, 'Pressure (mmHg)': self.pressupressuresAtSteadyStateres})

                            # Plot the results
                            plt.figure(figsize=(10, 6))
                            plt.plot(df['Pressure (mmHg)'], df['Pump Speed'], marker='o', linestyle='-')
                            plt.xlabel('Relative Pressure (mmHg)')
                            plt.ylabel('Pump Speed')
                            plt.title('Pump Speed vs Relative Pressure')
                            plt.grid(True)

                            # Annotate the plot with pump speeds for each integer pressure
                            for target_pressure in range(1, int(df['Pressure (mmHg)'].max()) + 1):
                                closest_index = (df['Pressure (mmHg)'] - target_pressure).abs().idxmin()
                                pressure = df['Pressure (mmHg)'][closest_index]
                                speed = df['Pump Speed'][closest_index]
                                
                                # Annotate the plot at the closest pressure point
                                plt.annotate(f'Speed: {speed}', 
                                            (pressure, speed), 
                                            textcoords="offset points", 
                                            xytext=(5, 5), 
                                            ha='center', 
                                            fontsize=8, 
                                            color='blue')

                            plt.show()
                            self.motor_experiment_find_speed_per_pressure_auto = False
                            self.air_speed = 0
                            self.ser.write(f"A_SPEED {self.air_speed}\n".encode())
                        elif self.loopCount%60 == 0:
                            self.air_speed += 1
                            self.ser.write(f"A_SPEED {self.air_speed}\n".encode())
                            self.speedForPressure.append(self.air_speed)
                            self.pressuresAtSteadyState.append(relative_pressure)

                    elif self.motor_experiment_find_response_time_from_each_pressure:
                        # Ensure initial calibration is done
                        if self.initial_pressure is None:
                            if self.pressure_data:
                                self.initial_pressure = biased_pressure
                            return self.line1, self.line2  # Ensure consistent return
                        
                        if self.repeat > 4: # 9: #9
                            self.save_experiment_data_to_csv()
                            self.air_speed = 0
                            self.ser.write(f"A_SPEED {self.air_speed}\n".encode())
                            time.sleep(0.5)
                            self.ser.write(b"AIR_OFF\n")
                            time.sleep(0.5)
                            self.suction_speed = 0
                            time.sleep(0.5)
                            self.ser.write(f"S_SPEED {self.suction_speed}\n".encode())
                            self.ser.write(b"SUCTION_OFF\n")
                            time.sleep(0.5)
                            self.save_data_to_csv()
                            self.save_SS_data_to_csv()
                            self.save_final_plot()
                            self.save_final_stat_plot()
                            self.close_program()

                        # elif self.repeat == 0 and self.loopCount == 200:
                        #     # Set initial reference pressure
                        #     if self.pressure_data:
                        #         self.initial_pressure = biased_pressure

                        # Start the experiment for the current speed
                        current_air_speed = self.air_speed_for_each_relP[self.air_speed_for_each_relP_index]

                        if self.loopCount == 0:
                            # Enable manual control and set speed
                            self.air_manual_control_enabled = True
                            self.ser.write(b"AIR_ON\n")
                            self.air_speed = current_air_speed
                            self.ser.write(f"A_SPEED {self.air_speed}\n".encode())
                            print(f"Setting air speed to {self.air_speed} for stabilization.")
                            self.start_time_to_stabilize = time.time()
                            self.response_time_init = 0
                            self.response_time_final = 0
                            self.wait_for_stabilization = True

                        elif self.wait_for_stabilization:
                            # Wait 10 seconds to allow the relative pressure to stabilize
                            if time.time() - self.start_time_to_stabilize >= 10:
                                self.wait_for_stabilization = False
                                print(f"Starting relative pressure stabilized at speed {self.air_speed}.")
                                self.motor_control_enabled = True
                                self.air_manual_control_enabled = False
                                self.ser.write(b"AIR_OFF\n")
                                self.ser.write(b"MOTOR_ON\n")
                                print("Switched to automatic control for stabilization.")
                                self.start_time_to_measure = time.time()  # Start timing for stabilization

                        # Measure response time to stabilize at 10 mmHg
                        if self.motor_control_enabled and not self.wait_for_stabilization:
                            # Calculate the moving average of the last 60 relative pressure readings
                            if len(self.relative_pressure_data) >= 20:  # Use the last 20 data points
                                
                                # moving avg
                                self.moving_avg_init = np.mean(list(self.pressure_data_all)[-20:])-760
                                self.moving_avg_final = np.mean(list(self.pressure_data_all)[-100:])-760

                                # Debug print to track moving average
                                print(f"Moving average for speed {current_air_speed}: {self.moving_avg_init:.2f}")

                                # Check if moving average is within the steady state range
                                if 9.7 <= self.moving_avg_init <= 10.3 and self.response_time_init == 0:  # Adjusted range for more precision
                                    self.response_time_init = time.time() - self.start_time_to_measure
                                    print(f"Stabilized at 10 mmHg in {self.response_time_init:.2f} seconds at initial speed {current_air_speed}.")
                                if 9.95 <= self.moving_avg_final <= 10.05 and self.response_time_final == 0:  # Adjusted range for more precision
                                    self.response_time_final = time.time() - self.start_time_to_measure
                                    print(f"Stabilized at 10 mmHg in {self.response_time_final:.2f} seconds at initial speed {current_air_speed}.")

                                # Test for 10 minutes and move on to next speed
                                if time.time() - self.start_time_to_measure >= 600: # 600:  # 10 minutes
                                    if self.response_time_init == 0:
                                        self.record_experiment_data(biased_pressure, relative_pressure, temperature, current_air_speed, -1, -1, current_air_speed)
                                        print(f"Timeout: Could not stabilize at speed {current_air_speed} within 15 minutes. Skipping to next speed.")
                                    else:
                                        self.record_experiment_data(biased_pressure, relative_pressure, temperature, current_air_speed, self.response_time_init, self.response_time_final, current_air_speed)
                                    self.prepare_for_next_speed()

                        self.loopCount += 1

                    elif self.motor_experiment_find_response_time_from_each_pressure_manual:
                        # Ensure initial calibration is done
                        if self.initial_pressure is None:
                            if self.pressure_data:
                                self.initial_pressure = biased_pressure
                            return self.line1, self.line2  # Ensure consistent return
                        
                        if self.repeat > 4: # 9: #9
                            self.save_experiment_data_to_csv()
                            self.air_speed = 0
                            self.ser.write(f"A_SPEED {self.air_speed}\n".encode())
                            time.sleep(0.5)
                            self.ser.write(b"AIR_OFF\n")
                            time.sleep(0.5)
                            self.suction_speed = 0
                            time.sleep(0.5)
                            self.ser.write(f"S_SPEED {self.suction_speed}\n".encode())
                            self.ser.write(b"SUCTION_OFF\n")
                            time.sleep(0.5)
                            self.save_data_to_csv()
                            self.save_SS_data_to_csv()
                            self.save_final_plot()
                            self.save_final_stat_plot()
                            self.close_program()
                        # 5, 7, 9, 11, 13, 15, 17, 19, 21, 23
                        self.motor_control_enabled = True
                        self.ser.write(b"MOTOR_ON\n")
                            
                        if self.loopCount == 0:
                            
                            self.start_time_to_measure = time.time()
                            self.response_time_init = time.time()
                            self.start_time_to_stabilize = time.time()

    
                            # Calculate the moving average of the last 60 relative pressure readings
                        if len(self.relative_pressure_data) >= 20:  # Use the last 20 data points
                            
                            # moving avg
                            self.moving_avg_init = np.mean(list(self.pressure_data_all)[-20:])-760
                            self.moving_avg_final = np.mean(list(self.pressure_data_all)[-100:])-760

                            # Debug print to track moving average
                            # print(f"Moving average for speed {current_air_speed}: {self.moving_avg_init:.2f}")

                            # Check if moving average is within the steady state range
                            if 9.7 <= self.moving_avg_init <= 10.3 and self.response_time_init == 0:  # Adjusted range for more precision
                                self.response_time_init = time.time() - self.start_time_to_measure
                                # print(f"Stabilized at 10 mmHg in {self.response_time_init:.2f} seconds at initial speed {current_air_speed}.")
                            if 9.95 <= self.moving_avg_final <= 10.05 and self.response_time_final == 0:  # Adjusted range for more precision
                                self.response_time_final = time.time() - self.start_time_to_measure
                                # print(f"Stabilized at 10 mmHg in {self.response_time_final:.2f} seconds at initial speed {current_air_speed}.")

                        if self.response_time_init == 0:
                            self.record_experiment_data(biased_pressure, relative_pressure, temperature, 0, -1, -1, 0)
                        else:
                            self.record_experiment_data(biased_pressure, relative_pressure, temperature, 0, self.response_time_init, self.response_time_final, 0)

                        self.loopCount += 1
                        if self.loopCount >= 200: # 10 sec
                            self.loopCount = 0
                            self.moving_avg_init = 0
                            self.moving_avg_final = 0
                            self.response_time_init = 0
                            self.response_time_final = 0
                            self.motor_experiment_find_response_time_from_each_pressure_manual = False
                            self.motor_control_enabled = False
                            self.ser.write(b"MOTOR_OFF\n")
                        
                    print(f"Time: {elapsed_time:.2f}s, Raw: {data}, "
                        f"AP: {biased_pressure:.2f} {self.unit}, "
                        f"RP: {relative_pressure:.2f} {self.unit}, "
                        f"T: {temperature:.2f}°C, AS: {air_speed}, "
                        f"SS: {suction_speed}, M: {self.motor_control_enabled}, "
                        f"A: {self.air_manual_control_enabled}, S: {self.suction_manual_control_enabled}, ma: {round(self.moving_avg_init,2)}, "
                        f"resp:1 {round(self.response_time_init,2)}, resp2: {round(self.response_time_final,2)}, "
                        f"r-l-s: {self.repeat}-{self.loopCount}-", end='')
                    print(f"{self.air_speed}") if self.air_manual_control_enabled else print(f"{self.suction_speed}")

                else:
                    print(f"Data: {data} | Status: Skipped (Data format error)")
            else:
                print(f"Data format error. Skipping this line: {data}")
            # return pressure, temperature, air_speed, suction_speed
        except UnicodeDecodeError as e:
            print(f"Decoding error for data: {data if 'data' in locals() else 'N/A'} | Error: {e}")
        except Exception as e:
            print(f"Unexpected error: {e}. Skipping this line.")

        # return self.line1, self.line2, self.line3
        return self.line1, self.line2

    def record_experiment_data(self, biased_pressure, relative_pressure, temperature, speed, response_time_init, response_time_final, set_speed):
        """Record experiment data."""
        self.data_all_SS.append({
            'Elapsed Time (s)': time.time() - self.start_time,
            'Absolute Pressure (mmHg)': [biased_pressure] if biased_pressure is not None else [0],
            'Relative Pressure (mmHg)': [relative_pressure] if relative_pressure is not None else [0],
            'Temperature (°C)': [temperature] if temperature is not None else [0],
            'System Start Time': self.system_start_time,
            'System Exit Time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),  # Log exit time
            'Air Speed': [self.air_speed] if self.air_speed is not None else [0],
            'Suction Speed': [self.suction_speed] if self.suction_speed is not None else [0],
            'Initial Speed': speed,
            'Initial Relative Pressure': self.relative_pressure_data[0] if self.relative_pressure_data else 0,
            'Response Time to 10 mmHg in 1 sec(s)': response_time_init, 
            'Response Time to 10 mmHg in 5 sec(s)': response_time_final,
            'Suction Speed': [set_speed] if set_speed is not None else [0],
        })
        unit = 'hPa' if self.unit_conversion_factor == 1.0 else 'mmHg'

    def prepare_for_next_speed(self):
        """Prepare the system for the next speed test."""
        self.air_speed_for_each_relP_index += 1
        self.loopCount = -1  # Reset loop count for the next speed
        self.response_time_init = 0
        self.response_time_final = 0
        self.motor_control_enabled = False
        self.ser.write(b"MOTOR_OFF\n")
        self.ser.write(f"A_SPEED 0\n".encode())
        self.air_manual_control_enabled = True
        self.ser.write(b"AIR_ON\n")

        # Check if all speeds have been tested for single cycle and reset to 0
        if self.air_speed_for_each_relP_index >= len(self.air_speed_for_each_relP):
            print(f"Experiment completed for all speeds for cycle {self.repeat}.")
            self.repeat += 1
            self.air_speed_for_each_relP_index = 0  # Reset speed index for next repeat
            self.loopCount = 0
            self.air_speed = 0 
            self.ser.write(f"A_SPEED 0\n".encode())
            self.suction_speed = 0
            self.ser.write(f"S_SPEED 0\n".encode())   

    def save_recent_data(self):
        # Define the number of recent data points to save
        num_recent_points = 20

        # Find the minimum length among the lists to ensure all are the same length
        min_length = min(len(self.time_data), len(self.pressure_data), 
                        len(self.relative_pressure_data), len(self.temperature_data))

        # Determine the current motor speed based on which control is enabled
        speed_data = [self.air_speed if self.air_manual_control_enabled else self.suction_speed] * min(min_length, num_recent_points)
                  

        # Slice all lists to the minimum length or last 20 entries to ensure they are equal
        recent_data = pd.DataFrame({
            'Elapsed Time (s)': list(self.time_data)[-num_recent_points:],
            'Absolute Pressure': list(self.pressure_data)[-num_recent_points:],
            'Relative Pressure': list(self.relative_pressure_data)[-num_recent_points:],
            'Temperature (°C)': list(self.temperature_data)[-num_recent_points:],
            'Speed': speed_data
        })

        # Define the filename with a fixed name

        # Check if the file exists
        file_exists = os.path.isfile(self.recent_filename)

        # Save or append the recent data to CSV
        with open(self.recent_filename, 'a' if file_exists else 'w', newline='') as f:
            recent_data.to_csv(f, index=False, header=not file_exists)
        
        print(f"Recent 20 data points saved to {self.recent_filename}")

    @staticmethod
    def parse_data(data):
        try:
            pressure, temperature, air_speed, suction_speed = map(float, data.split())
            return pressure, temperature, air_speed, suction_speed
        except ValueError as e:
            print(f"Data format error. Skipping this line: {data} | Error: {e}")
        return None, None, None, None

    def log_data(self, pressure, temperature, air_speed, suction_speed):
        biased_pressure = pressure + self.pressure_bias 

        if self.initial_pressure is None:
            self.initial_pressure = 1013.25  # [hPa] Set initial pressure as reference for relative pressure

        elapsed_time = time.time() - self.start_time
        relative_pressure = biased_pressure - self.initial_pressure
        relative_pressure *= self.unit_conversion_factor
        biased_pressure *= self.unit_conversion_factor
        self.pressure_data.append(biased_pressure)
        self.relative_pressure_data.append(relative_pressure)
        self.temperature_data.append(temperature)
        self.time_data.append(elapsed_time)

        # Append full data to lists (for saving)
        self.pressure_data_all.append(biased_pressure)
        self.relative_pressure_data_all.append(relative_pressure)
        self.temperature_data_all.append(temperature)
        self.time_data_all.append(elapsed_time)
        self.air_speed_data_all.append(air_speed)
        self.suction_speed_data_all.append(suction_speed)

        return biased_pressure, relative_pressure, elapsed_time

    def update_lines(self):
        adjusted_pressure_data = [p * self.unit_conversion_factor for p in self.pressure_data]
        adjusted_relative_pressure_data = [rp * self.unit_conversion_factor for rp in self.relative_pressure_data]

        self.line1.set_data(self.time_data, adjusted_pressure_data)
        self.line2.set_data(self.time_data, adjusted_relative_pressure_data)
        # self.line3.set_data(self.time_data, self.temperature_data)

    def adjust_axes(self):
        if self.time_data:
            elapsed_time = self.time_data[-1]
            self.ax1.set_xlim(max(0, elapsed_time - self.max_points * 0.05), elapsed_time)
            self.ax2.set_xlim(max(0, elapsed_time - self.max_points * 0.05), elapsed_time)
            # self.ax3.set_xlim(max(0, elapsed_time - self.max_points * 0.05), elapsed_time)

            """
            # Redraw canvas to apply changes
            self.ax1.figure.canvas.draw()
            self.ax2.figure.canvas.draw()
            self.ax3.figure.canvas.draw()
            """

    def toggle_pressure_units(self):
        if self.unit == "hPa":
            self.unit_conversion_factor = 0.750062
            self.unit = "mmHg"
            self.ax1.set_ylabel('Pressure (mmHg)')
            self.ax2.set_ylabel('Pressure (mmHg)')
            self.ax1.set_ylim(700, 820)
            print("Pressure units converted to mmHg.")
        else:
            self.unit_conversion_factor = 1.0
            self.unit = "hPa"
            self.ax1.set_ylabel('Pressure (hPa)')
            self.ax2.set_ylabel('Pressure (hPa)')
            self.ax1.set_ylim(800, 1200)
            print("Pressure units converted to hPa.")

        self.update_iah_grade_lines()
        # self.adjust_axes()

    def add_iah_grade_lines(self):
        iah_grades_human = {
            "IAH Grade I": 12, # 15.99864
            "IAH Grade II": 16, #  21.33152
            "IAH Grade III (ACS)": 21, # 27.99762
            "IAH Grade IV (ACS)": 25 # 33.305, 66.661
        }

        iah_grades_small_animal = {
            "Mild IAH": 7.4,
            "Moderate IAH": 14.7,
            "Severe IAH": 25.7
        }

        # Add lines for human IAH grades
        for grade, value in iah_grades_human.items():
            line = self.ax2.axhline(y=value, color='orange', linestyle='--', label=grade, linewidth=0.5)
            self.ax2.text(0.02, value, grade, color='orange', va='bottom')
            self.iah_lines.append((line, value))

        # Add black dotted lines for small animal IAH grades
        for grade, value in iah_grades_small_animal.items():
            line = self.ax2.axhline(y=value, color='blue', linestyle='-', label=grade, linewidth=0.5)
            self.ax2.text(0.98, value, grade, color='blue', va='bottom', ha='right', transform=self.ax2.get_yaxis_transform())
            self.iah_lines.append((line, value))

    def update_iah_grade_lines(self):
        for line, value in self.iah_lines:
            line.set_ydata([value * self.unit_conversion_factor] * 2)

    def save_data_to_csv(self):
        adjusted_pressure_data_all = self.pressure_data_all # mmHg
        adjusted_relative_pressure_data_all = self.relative_pressure_data_all # mmHg

        df = pd.DataFrame({
            'Elapsed Time (s)': self.time_data_all,
            'Absolute Pressure (mmHg)': adjusted_pressure_data_all,
            'Relative Pressure (mmHg)': adjusted_relative_pressure_data_all,
            'Temperature (°C)': self.temperature_data_all,
            'System Start Time': self.system_start_time,
            'System Exit Time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),  # Log exit time
            'Air Speed': self.air_speed_data_all,
            'Suction Speed': self.suction_speed_data_all,
        })
        unit = 'hPa' if self.unit_conversion_factor == 1.0 else 'mmHg'

        # filename = self.generate_filename(f'pressure_data_{unit}', 'xlsx')
        # df.to_excel(filename, index=False)
        filename = self.generate_filename(f'pressure_data_{unit}', 'csv')
        df.to_csv(filename, index=False)
        
        print(f"Data saved to {filename}")
    
    def save_SS_data_to_csv(self):
        df = pd.DataFrame(self.data_all_SS)
        unit = 'hPa' if self.unit_conversion_factor == 1.0 else 'mmHg'

        # filename = self.generate_filename(f'pressure_data_{unit}', 'xlsx')
        # df.to_excel(filename, index=False)
        filename = self.generate_filename(f'SS_pressure_data_{unit}', 'csv')
        df.to_csv(filename, index=False)
        
        print(f"Data saved to {filename}")

    def save_final_plot(self):
        # Create a new figure for the entire dataset
        fig_all, (ax1_all, ax2_all) = plt.subplots(2, 1, figsize=(10, 8))

        # Plot all data in the final plot, not just the last 100 points
        adjusted_pressure_data_all = [p * self.unit_conversion_factor for p in self.pressure_data_all]
        adjusted_relative_pressure_data_all = [rp * self.unit_conversion_factor for rp in self.relative_pressure_data_all]

        # Plot the entire dataset
        ax1_all.plot(self.time_data_all, adjusted_pressure_data_all, 'g-', label='Absolute Pressure (mmHg)')
        ax1_all.set_title('Absolute Pressure - Entire Dataset')
        ax1_all.set_xlabel('Time (s)')
        ax1_all.set_ylabel(f'Pressure ({self.unit})')
        ax1_all.set_ylim(700, 820)
        ax1_all.legend()

        ax2_all.plot(self.time_data_all, adjusted_relative_pressure_data_all, 'b-', label='Relative Pressure (mmHg)')
        ax2_all.set_title('Relative Pressure - Entire Dataset')
        ax2_all.set_xlabel('Time (s)')
        ax2_all.set_ylabel(f'Pressure ({self.unit})')
        ax2_all.set_ylim(-5, 50)
        ax2_all.legend()

        # Adjust spacing between subplots for the entire data plot
        fig_all.subplots_adjust(hspace=0.4)  # Increase hspace to avoid overlap

        # Add IAH human grade lines to the relative pressure plot for the entire dataset
        human_iah_grades = {
            "IAH Grade I": 12,
            "IAH Grade II": 16,
            "IAH Grade III (ACS)": 21,
            "IAH Grade IV (ACS)": 25
        }

        # for grade, value in human_iah_grades.items():
        ax2_all.axhline(y=12, color='g', linestyle='--', linewidth=0.5)
        ax2_all.text(min(self.time_data_all), 12, 'IAH Grade I', color='g', fontsize=8, ha='left', va='bottom')
        ax2_all.axhline(y=16, color='y', linestyle='--', linewidth=0.5)
        ax2_all.text(min(self.time_data_all), 16, 'IAH Grade II', color='y', fontsize=8, ha='left', va='bottom')
        ax2_all.axhline(y=21, color='orange', linestyle='--', linewidth=0.5)
        ax2_all.text(min(self.time_data_all), 21, 'IAH Grade III (ACS)', color='orange', fontsize=8, ha='left', va='bottom')
        ax2_all.axhline(y=25, color='r', linestyle='--', linewidth=0.5)
        ax2_all.text(min(self.time_data_all), 25, 'IAH Grade IV (ACS)', color='r', fontsize=8, ha='left', va='bottom')

        # Set x-axis limits to the full time range
        ax1_all.set_xlim(min(self.time_data_all), max(self.time_data_all))
        ax2_all.set_xlim(min(self.time_data_all), max(self.time_data_all))

        # Annotate system start and exit time at the x-axis limits
        ax1_all.text(min(self.time_data_all), ax1_all.get_ylim()[0], f'Start: {self.system_start_time}', va='bottom', ha='left', color='black', fontsize=6)
        ax1_all.text(max(self.time_data_all), ax1_all.get_ylim()[0], f'Exit: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', va='bottom', ha='right', color='black', fontsize=6)

        ax2_all.text(min(self.time_data_all), ax2_all.get_ylim()[0], f'Start: {self.system_start_time}', va='bottom', ha='left', color='black', fontsize=6)
        ax2_all.text(max(self.time_data_all), ax2_all.get_ylim()[0], f'Exit: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', va='bottom', ha='right', color='black', fontsize=6)
 
        # Save the final plot of the entire dataset
        filename = self.generate_filename('final_entire_dataset_plot', 'png')
        fig_all.savefig(filename)
        # filename = self.generate_filename('final_entire_dataset_plot', 'svg')
        # fig_all.savefig(filename)
        print(f"Entire dataset plot saved as {filename}")
        plt.close(fig_all)  # Close figure to free memory
        # # Save the last visible plot as the final plot
        # filename = self.generate_filename('final_plot', 'png')
        # self.fig.savefig(filename)
        # print(f"Final plot saved as {filename}")
        """
        self.fig1.savefig(self.generate_filename('absolute_pressure_plot', 'png'))
        self.fig2.savefig(self.generate_filename('relative_pressure_plot', 'png'))
        self.fig3.savefig(self.generate_filename('temperature_plot', 'png'))
        print("Final plots saved.")
        """
    
    def save_final_stat_plot(self):
        # Define the filename for recent data

        # Check if the file exists before proceeding
        if not os.path.exists(self.recent_filename):
            print(f"File {self.recent_filename} does not exist. Cannot generate final stat plot.")
            return

        # Read the recent data from CSV file
        df = pd.read_csv(self.recent_filename , encoding='cp949')

        # Ensure that the DataFrame has enough data to be divided into 10 repeats
        total_rows = len(df)
        # if total_rows < 2000: # 4000~~~~~~~~~~~~~~~~~``
        #     print(f"Not enough data in {filename}. Found {total_rows} rows, but need at least 2000.")
        #     return
    
        plt.figure(figsize=(10, 8))

        # Use the 'tab10' colormap which provides 10 distinct colors
        colors = plt.colormaps['tab10'](np.linspace(0, 1, 10))  # Generate 10 distinct colors for the repeats

        # Define fixed axis limits for y-axis
        min_relative_pressure = df['Relative Pressure'].min() if 'Relative Pressure' in df.columns else -10
        max_relative_pressure = df['Relative Pressure'].max() if 'Relative Pressure' in df.columns else 30

        # Divide the DataFrame into 10 chunks of 400 rows each
        chunk_size = total_rows // 2  # 10~~~~~~~~~~~~~~~~Compute chunk size dynamically based on total rows

        plot_exists = False  # Flag to check if any valid plots are created

        for repeat in range(10):  # Adjusted for 2 repeats, change to 10 for the full dataset
            start_index = repeat * chunk_size
            end_index = start_index + chunk_size
            df_chunk = df.iloc[start_index:end_index]  # Extract chunk for the current repeat

            # Check if the DataFrame chunk has valid data
            if not df_chunk.empty and 'Elapsed Time (s)' in df_chunk.columns and 'Relative Pressure' in df_chunk.columns:
                time_data = df_chunk['Elapsed Time (s)'].dropna()
                pressure_data = df_chunk['Relative Pressure'].dropna()

                if not time_data.empty and not pressure_data.empty:
                    # Normalize the elapsed time for each repeat to start at 0
                    normalized_time_data = time_data - time_data.iloc[0]

                    plt.plot(
                        normalized_time_data,
                        pressure_data,
                        label=f'Repeat {repeat + 1}',
                        color=colors[repeat % 10]  # Use distinct colors for each repeat
                    )
                    plot_exists = True
                else:
                    print(f"No data to plot for repeat {repeat + 1}. Skipping.")
            else:
                print(f"No valid data found for chunk {repeat + 1}. Skipping.")

        if plot_exists:
            # Set dynamic axis limits
            plt.xlim(0, max(normalized_time_data))  # X-axis from 0 to max normalized time
            plt.ylim(min_relative_pressure, max_relative_pressure)

            plt.xlabel('Elapsed Time (s)')
            plt.ylabel('Relative Pressure (mmHg)')
            plt.title('Relative Pressure vs Elapsed Time for Multiple Repeats')
            plt.legend()
            plt.grid(True)

            # Save the plot with the current system time in the filename
            filename_png = self.generate_filename('final_Stat_dataset_plot', 'png')
            filename_svg = self.generate_filename('final_Stat_dataset_plot', 'svg')
            plt.savefig(filename_png)
            plt.savefig(filename_svg)
            plt.close()

            print(f"Final plot saved as {filename_png} and {filename_svg}")
        else:
            print("No valid data found for plotting. No plot was created.")
            

    @staticmethod
    def generate_filename(prefix, extension):
        current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        return f"{current_time}_{prefix}.{extension}"

    def close_program(self):
        self.air_speed = 0
        self.ser.write(f"A_SPEED {self.air_speed}\n".encode())
        time.sleep(0.5)
        self.ser.write(b"AIR_OFF\n")
        time.sleep(0.5)
        self.suction_speed = 0
        time.sleep(0.5)
        self.ser.write(f"S_SPEED {self.suction_speed}\n".encode())
        self.ser.write(b"SUCTION_OFF\n")
        time.sleep(0.5)
        plt.close('all')
        self.ser.close()
        sys.exit()

    def run(self):
        ani = animation.FuncAnimation(self.fig, self.update_plot, init_func=self.init_plot, blit=True, interval=0.1)
        """
        ani1 = animation.FuncAnimation(self.fig1, self.update_plot, init_func=self.init_plot, blit=True, interval=1)
        ani2 = animation.FuncAnimation(self.fig2, self.update_plot, init_func=self.init_plot, blit=True, interval=1)
        ani3 = animation.FuncAnimation(self.fig3, self.update_plot, init_func=self.init_plot, blit=True, interval=1)
        """
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    logger = ArduinoDataLogger()
    # while logger.experiment_running:
    #         logger.check_for_esc()  # Check for ESC key press
    #         logger.run_experiment()
    logger.run()
    logger.save_data_to_csv()
    # logger.perform_anova_analysis()  # Perform ANOVA analysis
    logger.perform_bland_altman_analysis()  # Perform Bland-Altman analysis
    logger.save_final_stat_plot()

    # Save separate plots
    # Toggle unit and Y-axis
    # X axis
