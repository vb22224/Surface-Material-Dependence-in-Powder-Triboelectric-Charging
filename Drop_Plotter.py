# -*- coding: utf-8 -*-
"""
Code for Producing a Figure from Voltage Drop Data for multiple drops
Will plot Charge or Voltage
Can plot the trace, boxplot, or both
Separated into functions

Created: 27/09/23
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import statistics as stats
import math as m
import os
import json
from scipy.signal import find_peaks, peak_prominences, iirnotch, filtfilt
import numpy as np
import re
import random
import csv



def get_dif(d, change_type):
    
    """ Calculates the specified difference in a charge or voltage trace """
    
    if change_type == "mm":
        dif = max(d) - min(d)
        
    elif change_type == "sf":
        dif = d[-1] - d[0]
        
    elif change_type == "mmsf":
        dif = max(d) - min(d) - abs(d[-1] - d[0])
        
    else:
        raise ValueError("Invalid change type, should be start to finish (sf), min and max (mm), or mm minus sf (mmsf)")
    
    return dif



def decimal_place(number):
    
    """Returns the number of decimal places needed to represent the most significant digit of a positive number."""
    
    if number == 0:
        decimals_to_return =  0
    else:
        decimals_to_return = 1 + max(0, -int(m.floor(m.log10(abs(number)))))
    
    return decimals_to_return



def get_times(sample_rates, data):
    
    """ Gets the time step, array of times, and total time required for the trace data """
    
    if len(sample_rates) == 0:
        raise ValueError('The smple_rates is empty, this is probably due to no matching files being found. Check path/file names.')
    time_step = 1 / sample_rates[0]
    times = [i * time_step for i in range(len(data[0]))]
    time = len(data[0]) * time_step
    
    return time_step, times, time



def get_axis_title(p, adjust_by_mass=False):
    
    """ Looks up the axis title from parameter """
    
    if p == "V" or p == "RV":
        axis = "Voltage / V"
        
    elif p == "Q" or p == "HQ" or p == "VHQ":
        axis = "Charge / pC"
        
    if adjust_by_mass:
        axis = "Specific " + axis + " g$^{-1}$"
    
    return axis



def notch_filter(data, fs=1000, freq=50, Q=30):
    
    """Applies a notch filter to remove a specific frequency (e.g., 50 Hz mains hum)."""
    
    # Calculate the notch filter coefficients
    b, a = iirnotch(freq, Q, fs)
    filtered_data = filtfilt(b, a, data)  # Apply the filter (zero-phase filtering)
    
    return filtered_data



def calculate_opacity(file_names, counter2, lol_structure, counter):
    
    "Calculates how opaque each line in the same category should be"
    
    try:
        # Get the number from filename
        current_num = float(file_names[counter2].split('.')[0].split('_')[-1])
        
        # Calculate max value safely
        first_num = int(file_names[0].split('.')[0].split('_')[-1])
        max_val = max(lol_structure[counter], first_num)
        
        # Calculate opacity and clamp between 0 and 1
        opacity = 1 - 0.5 * ((current_num - 1) / max_val)
        opacity = max(0, min(1, opacity))  # Clamp between 0 and 1
        
        return opacity
        
    except (IndexError, ValueError, ZeroDivisionError) as e:
        print(f"Error: {e}")
        return 1  # Default opacity



def save_to_csv(times, data, file_path, labels=None, append_columns=False):
    
    """
    Saves the given times and data to a CSV file. Can either write a new file 
    or append new columns to an existing file.

    Parameters:
    - times: List of time values.
    - data: List of data values (single list or list of lists for multiple traces).
    - file_path: Path to the CSV file where the data will be saved.
    - labels: Optional list of labels for each dataset.
    - append_columns: If True, new data will be appended as columns to an existing file.
    """

    # Ensure data is a list of lists for consistent processing
    if not isinstance(data[0], list):
        data = [data]  # Convert single dataset into a list of lists
    if labels is None:
        labels = [f"Data_{i}" for i in range(len(data))]  # Auto-generate labels if missing

    if append_columns and os.path.exists(file_path):
        with open(file_path, mode='r', newline='') as file:
            reader = list(csv.reader(file))

        header = reader[0]
        existing_data = [row[:] for row in reader[1:]]
        new_columns, new_labels = [], []
        
        for i, dataset in enumerate(data): # Identify existing columns (avoid duplicates)
            label = labels[i] if i < len(labels) else f"Data_{i}"
            if label not in header:  # Prevent duplicate column names
                new_labels.append(label)
                new_columns.append(dataset)
            else:
                print(f"Skipping duplicate column: {label}")

        while len(dataset) < len(existing_data):
            dataset.append("") # or np.nan for missing data
            
        while len(existing_data) < len(dataset):
            existing_data.append([""] * len(existing_data[0]))  # Fill with empty strings

        for row_idx in range(len(existing_data)): # Append new columns
            for dataset in new_columns:
                existing_data[row_idx].append(str(dataset[row_idx]))

        header.extend(new_labels) # Update header
        
        with open(file_path, mode='w', newline='') as file: # Write updated CSV
            writer = csv.writer(file)
            writer.writerow(header)
            writer.writerows(existing_data)

    else:
        with open(file_path, mode='w', newline='') as file: # Create new CSV file
            writer = csv.writer(file)
            writer.writerow(['Time (s)'] + labels)
            for i, time in enumerate(times):
                row = [time] + [dataset[i] for dataset in data]
                writer.writerow(row)



def custom_sort(file_names):
    
    """ Sorts the file names such that they will be in a logical order in the plot legend """
    
    def sorting_key(name):
        parts = re.split(r'(-?\d+)', name)
        parts = [-int(part) if part.lstrip('-').isdigit() else part for part in parts]
        return parts
    
    sorted_names = sorted(file_names, key=sorting_key)
    
    return sorted_names



def bootstrap_statistic(data, num_samples=1000, stat_func=stats.mean):
    
    """ Generate bootstrap samples and calculate the statistic for each sample. """
    
    boot_samples = []
    
    for _ in range(num_samples):
        resample = [random.choice(data) for _ in range(len(data))]
        boot_samples.append(stat_func(resample))
    
    return boot_samples



def average_data(data, times, start_index=0, end_index=None, error_lines=False, av_option='mean', drop_stats=False):
    
    """
    Averages data from the data list of traces in specified range start_index to end_index.
    Clips this data to the shortest trace in the range and also returns the times required to plot this average.
    Will also return the error bounds if error_lines = True.
    """
    
    if end_index is None:
        end_index = len(data[0])
        
    valid_data = data[start_index:end_index]
    
    data_lengths, averaged_data, average_minus_error, average_plus_error = [], [], [], []
    
    bootstrap_samples = 1000
    
    for d in valid_data:
        data_lengths.append(len([x for x in d if not m.isnan(x)]))
        
    if drop_stats:  # Gets errors on the charge drops
        diff_lis = []
        
        for i, vd in enumerate(valid_data):
            diff = vd[data_lengths[i]-1] - vd[0]
            diff_lis.append(diff)
        
        print(f'Mean drop: {np.mean(diff_lis)}, std dev: {np.std(diff_lis)}, N: {len(diff_lis)}')
    
    for n in range(min(data_lengths)):
        n_data = []
        
        for d in valid_data:
            n_data.append(d[n])
            
        if av_option == 'mean':
            av = stats.mean(n_data)
            
        elif av_option == 'median':
            av = stats.median(n_data)
            
        elif av_option == 'bootstrap':
            boot_means = bootstrap_statistic(n_data, num_samples=bootstrap_samples, stat_func=stats.mean)
            av = stats.mean(boot_means)
            
        else:
            raise ValueError(f"av_option should be 'mean', 'median', or 'bootstrap', instead: {av_option}")
        
        averaged_data.append(av)
        
        if error_lines:
            if len(n_data) > 1:
                SEM = stats.stdev(n_data) / stats.sqrt(len(n_data))
                average_minus_error.append(av - SEM)
                average_plus_error.append(av + SEM)
                
            else:
                average_minus_error, average_plus_error = 'NaN', 'NaN'
                
    av_times = times[:min(data_lengths)]
    
    return averaged_data, av_times, average_minus_error, average_plus_error



def calibration_constants(step_heights, peak_heights):
    
    """ Takes array of Faraday cup steps and ring probe peaks, returning the corresponding calibration constants and associated errors """
    
    if len(step_heights) > 0 and (len(peak_heights) == 1 or all(len(peak_heights[i]) == len(step_heights) for i in range(len(peak_heights)))):
        
        peak_heights = np.array(peak_heights)
        
        for ring, peaks in enumerate(peak_heights):
            if len(peaks) == len(step_heights):
                ratios = step_heights / peaks
                ratio_mean = stats.mean(ratios)
                ratio_sem = stats.stdev(ratios) / np.sqrt(len(peaks))
                print(f"The calibration constant of ring {ring + 1} is: {round(ratio_mean, 2)} ± {round(ratio_sem, 2)}")
                
            else:
                print(f"Warning: The number of peaks for ring {ring + 1} does not match the number of Faraday cup steps.")
                
    else:
        raise ValueError(f"Mismatch in lengths: Faraday cup steps [{len(step_heights)}] and ring peaks {[len(peak_heights[i]) for i in range(len(peak_heights))]}.")

    return



def get_steps(d, time_step, step_background_time, step_threshold, step_width):
    
    """ Takes a Faraday cup calibration trace and finds the steps produced by each droplet """
    
    background_stdev = stats.stdev(d[:int(step_background_time / time_step)])
    average_level = stats.mean(d[:int(step_background_time / time_step)])
    threshold = step_threshold * background_stdev
    step_width_index = int(step_width / time_step)
    
    step_indices, step_times, step_middles, step_heights = [], [], [], []
    
    for n, d_point in enumerate(d[:len(d) - step_width_index]):  # Scan over trace
        stdev = stats.stdev(d[n:n + step_width_index])
        
        if stdev > threshold and all(abs(n - i) > step_width_index for i in step_indices):
            step_indices.append(n)
    
    for n in step_indices:  # Going through each step working out their properties
        average_after = stats.mean(d[int(n + step_width_index):int(n + step_width_index * 2)])
        step_height = average_after - average_level
        step_middle = average_level + step_height / 2
        step_time = time_step * (n + np.argmin(np.abs(np.array(d[int(n):int(n + step_width_index)]) - step_middle)))
        
        step_heights.append(step_height)  # Appending values
        step_times.append(step_time)
        average_level = average_after  # Updating new height
        
    step_times, step_middles, step_heights = np.array(step_times), np.array(step_middles), np.array(step_heights)
    
    return step_heights, step_times, step_middles



def get_peaks(d, peak_std_dev_threshold, include_negatives=False):
    
    """ Takes a ring probe calibration trace and finds the peaks produced by each droplet """
    
    peak_threshold = np.std(d) * peak_std_dev_threshold
    peaks, properties = find_peaks(d)
    prominences = peak_prominences(d, peaks)[0]
    
    peak_height_row = []
    peak_indices = []
    
    if include_negatives:
        neg_peaks, neg_properties = find_peaks(-np.array(d))
        prominences = np.concatenate([peak_prominences(d, peaks)[0], peak_prominences(-np.array(d), neg_peaks)[0]])
        peaks = np.concatenate([peaks, neg_peaks])
        
    for peak_num, p in enumerate(peaks):
        if prominences[peak_num] > peak_threshold:
            peak_height_row.append(prominences[peak_num])
            peak_indices.append(p)
    
    return peak_height_row, peak_indices



def average_multiple(collated_data, collated_times, filtered_names, parameter_lis=["Q", "RV", "RV"]):
    
    """ Takes data from multiple runs and averages them to the required format """
    
    averaged_names, lol_data, averaged_times, averaged_data = [], [], [], [[]]
    
    data_sets = int(len(collated_data) / len(parameter_lis))
    
    if data_sets != len(filtered_names):
        raise ValueError(f"Error with the number of files! Data sets: {data_sets}, filtered names: {filtered_names}")
    
    for i1, p in enumerate(parameter_lis):
        
        for i2, n in enumerate(filtered_names):
            name = n.split('_')[0] + '_' + p + '_' + str(parameter_lis[:i1].count(p) + 1)
            index = i1 * data_sets + i2
            
            if name not in averaged_names:
                averaged_names.append(name)
                lol_data.append([collated_data[index]])
                averaged_times.append(collated_times[index])
                
            else:
                lol_data[averaged_names.index(name)].append(collated_data[index])
    
    for i, lol in enumerate(lol_data):
        
        for d in range(len(lol[0])):
            total = 0
            
            for l in lol:
                total += l[d]
                
            av = total / len(lol)
            
            if d == 0:
                averaged_data.append([])
                
            averaged_data[i].append(av)
    
    return averaged_data[:-1], averaged_times, averaged_names



def print_charge_drop(data, d, name, T=0, RH=0, datetime="29/04/2025 13:38:39.040658", print_drops='N'):
    
    """ Function that prints the chagre drop for each dataset using the index (d), its name (name) and the given calculation method (print_drops) """
    
    time_steps_to_average = 10  # Number of timesteps to average in charge drop calculation
    column = pd.Series(pd.to_numeric(data[d], errors='coerce'))
    name = '.'.join(name.split('.')[:-1]) # Removing file extension from name
    
    last_non_nan_index = column.last_valid_index()
    start = np.nanmean(column[:time_steps_to_average])
    end = np.nanmean(column[last_non_nan_index - time_steps_to_average:last_non_nan_index])
    dmax = np.nanmax(column[:last_non_nan_index])
    dmin = np.nanmin(column[:last_non_nan_index])
    dif = end - start
    
    if end < start:
        dif *= -1
        
    if end < 0:
        extreme = dmax
    else:
        extreme = dmin
    
    if print_drops == 'SE':
        print(f'{name}, {end - start}', end='')  # End - Start
    elif print_drops == 'MM':
        print(f'{name}, {dmax - dmin}', end='') # Max - Min (sign altered to be the same as that of net change)
    elif print_drops == 'EE':
        print(f'{name}, {end - extreme}', end='')  # End - Extreme
    elif print_drops == 'ALL':
        print(f'{name}, {end - start}, {dmax - dmin}, {end - extreme}', end='')  # All the above options
    else:
        raise ValueError(f"The value of 'print_drops' should be: start and end (SE), maximum and minimum (MM), end minus the opposite extreme (EE), all (ALL) or none (N), but instead given: {print_drops}")
    
    if T != 0 and RH != 0:
        print(f', {T}, {RH}, {datetime}', end='')
    
    print('')
    
    return



def get_file_names(input_type, path, file_names_lis, file_names_lol, remove_files, lol_labels, spec_cat=[]):
    
    """ Retrieves file names based on input type and filters them """
    
    lol_structure, file_names = [], []
    
    if input_type == "dir" or input_type == "cat":
        target_directory = os.path.join(os.getcwd(), path)
        
        if not os.path.exists(target_directory):
            raise ValueError(f"Directory not found: {target_directory}")
        
        all_file_names = os.listdir(target_directory)
        
    elif input_type == "lis":
        all_file_names = file_names_lis
        
    elif input_type == "lol":
        all_file_names = []
        
        for lis in file_names_lol:
            lol_structure.append(0)
            
            for l in lis:
                all_file_names.append(l)
                lol_structure[-1] += 1
    
    else:
        raise ValueError("Invalid input_type, requires directory (dir), categorised (cat), list (lis), or list of lists (lol)")
    
    all_file_names = [file for file in all_file_names if file.endswith(".txt")]
    
    for remove_file in remove_files:
        all_file_names = [file for file in all_file_names if remove_file not in file]
        
    for file in all_file_names:
        if "README" not in file:  # Ignores the README file
            try:
                read = open(os.path.join(path, file), "r", encoding="utf-8").read()
                
                if "Keithley" in read:
                    file_names.append(file)
                    
            except FileNotFoundError:
                print(f"File not found: {file}")
                
            except UnicodeDecodeError as e:
                raise ValueError(f"Error reading {file}: {e}")
    
    if input_type == "cat":
        if spec_cat:
            file_names = [f for f in file_names if f.split("_")[0] in spec_cat]
            
        # Sort files by category
        file_names = custom_sort(file_names)
        
        # Separate PP files and other files
        pp_files = [f for f in file_names if f.split("_")[0] == 'PP']
        other_files = [f for f in file_names if f.split("_")[0] != 'PP']
        
        # Combine them with PP files first
        file_names = pp_files + other_files
        
        lol_labels, lol_structure = [], []
        category_map = {'PP': 'Polypropylene'}  # Add mappings here
        
        for f in file_names:
            category = f.split("_")[0]
            display_name = category_map.get(category, category)  # Use mapped name if exists
            
            if display_name in lol_labels:
                lol_structure[-1] += 1
            else:
                lol_labels.append(display_name)
                lol_structure.append(1)
    
    return file_names, lol_structure, lol_labels



def read_and_convert_data(file_names, target_directory, plot_parameter, ignore_len_errors, base_time=1, tolerance_amp=2, tolerance_len=100,
                          trim=False, manual_trim={}, store_dict=False, read_dict=False, data_col=0, adjust_by_mass=False, adjust_for_decay=False, masses_name='Masses.csv', time_const=310.5, remove_noise=False):
    
    """ Reads and converts data from files into the required format """
    
    data, sample_rates, temperatures, humidities, datetimes, filtered_names = [], [], [], [], [], []
    
    trim_file = os.path.join(target_directory, "manual_trim_dictionary")
    
    if adjust_by_mass:
        target_mass_dir = f'{target_directory}\{masses_name}'
        
        if os.path.exists(target_mass_dir):
            mass_df = pd.read_csv(target_mass_dir)
        else:
            raise FileNotFoundError(f"The file {target_mass_dir} does not exist.")
    
    if os.path.exists(trim_file):
        if read_dict:
            with open(trim_file, 'r') as json_file:
                manual_trim = json.load(json_file)
        else:
            print(f"{trim_file} does not exist")
    
    if store_dict:
        with open(trim_file, 'w') as json_file:
            json.dump(manual_trim, json_file)
    
    for file in file_names:
        with open(os.path.join(target_directory, file), 'r') as f:
            for line_number, line in enumerate(f):
                if "Keithley" in line:
                    detected_header = line_number
        
        dat = pd.read_csv(os.path.join(target_directory, file), header=detected_header).values.tolist()
        converted_data = []
        
        if len(dat[0]) < data_col + 1:
            print(f"No column {data_col} in file: {file}")
        
        else:
            filtered_names.append(file)
            
            if adjust_by_mass:
                if file.split('.')[0] in mass_df['files'].values:
                    indices = np.where(mass_df['files'].values == file.split('.')[0])[0][0]
                    mass = mass_df['masses'][indices]
                else:
                    raise FileNotFoundError(f"{target_mass_dir} does not contain {file}.")
            else:
                mass = 1
            
            metadata = open(os.path.join(target_directory, file), "r").read().split("Keithley")[0]
            sample_rate = float(metadata.split(",")[1])
            sample_rates.append(sample_rate)
            last_d, total_offset = 0, 0  # no previous datapoint or offset
            
            if "Temperature" in metadata and "Humidity" in metadata:
                temperature = float(metadata.split(",")[-2].split(" ")[-2])
                humidity = float(metadata.split(",")[-1].split(" ")[-3])
                datetime = str(metadata[:26])
                temperatures.append(temperature)
                humidities.append(humidity)
                datetimes.append(datetime)
                
            for d in dat:
                if plot_parameter == "V":
                    converted_data.append(d[data_col] * 10 / mass)
                elif plot_parameter == "RV":
                    converted_data.append(d[data_col] / mass) 
                elif plot_parameter == "Q":
                    if "Carbon Black" in file: d[data_col] -= 0.35
                    converted_data.append(d[data_col] * 10 * 130 / mass)
                elif plot_parameter == "HQ":
                    converted_data.append(d[data_col] * 100 * 130 / mass)
                elif plot_parameter == "VHQ":
                    converted_data.append(d[data_col] * 100 * 130  * 1.18 / mass) # Factor for having flexy cable
                else:
                    raise ValueError("Invalid plot_parameter, requires charge (Q), voltage (V), raw voltage (RV), high charage range (HQ), or very high charage range (VHQ)")
                
                if adjust_for_decay or file.split("_")[0] == "Grimsvotn":  # Adjusting for the decay due to the conductivity of air
                    time = 1 / sample_rate
                    decay = last_d * (1 - np.exp(-time / time_const))  # Standard exponential decay from the last timestep
                    total_offset += decay
                    last_d = converted_data[-1]  # Storing datapoint before the adjustment
                    converted_data[-1] = converted_data[-1] + total_offset  # The data adjusted for the decay
            
            if remove_noise: # Applying a notch filter at 50 Hz
                filtered_data = notch_filter(converted_data, Q=30, freq=50)  # First pass: clean 50 Hz
                filtered_data = notch_filter(filtered_data, Q=10, freq=50)   # Second pass: deeper 50 Hz
                filtered_data = notch_filter(filtered_data, Q=30, freq=100)  # Third pass: remove 100 Hz harmonic
                filtered_data = notch_filter(filtered_data, Q=30, freq=150)  # Third pass: remove 100 Hz harmonic
                
                # converted_data = np.array(converted_data)
                # filtered_data = np.array(filtered_data)     
                # fs = 1000  # e.g., 1000 Hz sampling rate
                # t = np.arange(len(converted_data)) / fs
                # mask = (t >= 1.0) & (t <= 1.1)
                # t_short = t[mask]
                # converted_short = converted_data[mask]
                # filtered_short = filtered_data[mask]
                # plt.figure(figsize=(12, 6))
                # plt.subplot(2, 1, 1)
                # plt.plot(t_short, converted_short)
                # plt.title("Original Signal with 50 Hz Hum")
                # plt.grid(True)
                # plt.subplot(2, 1, 2)
                # plt.plot(t_short, filtered_short)
                # plt.title("Signal after Notch Filter (50 Hz removed)")
                # plt.grid(True)
                # plt.tight_layout()
                # plt.show()
                
                converted_data = filtered_data.tolist()
            
            data.append(converted_data)
        
        if len(sample_rates) == 0:
            raise ValueError(f'No output files found in directory: {target_directory}')
        
        elif len(set(sample_rates)) != 1:
            raise ValueError(f'Sample rates do not match: {sample_rates}')
    
    if trim:  # Trimming off the start of each trace
        for count, f in enumerate(file_names):  # Doing the manual trim
            if f in manual_trim:
                trim_time = manual_trim[f]
                trim_steps = trim_time * sample_rates[count]
                data[count] = data[count][int(trim_steps):]
        
        for count, dat in enumerate(data):
            samples = round(base_time * sample_rates[count])
            amp_s = max(dat[:samples]) - min(dat[:samples])
            
            for c, d in enumerate(dat):
                if len(dat) > c + tolerance_len:
                    if abs(d - dat[c + tolerance_len]) > amp_s * tolerance_amp:
                        data[count] = data[count][c:]
                        break
                
                else:
                    print(f"No significant trace detected in {file_names[count]} at this significance level")
                    break
    
    if ignore_len_errors != "Ignore":
        for count_1, d_1 in enumerate(data):
            for count_2, d_2 in enumerate(data):
                if len(d_2) != len(d_1):
                    sr = sample_rates[count_1]
                    t_1 = len(d_1) / sr
                    t_2 = len(d_2) / sr
                    
                    if ignore_len_errors == "Error":
                        print(f'Sample durations do not match: {t_1} s and {t_2} s in {file_names[count_1]} and {file_names[count_2]}.')
                        
                    elif ignore_len_errors == "Extend":
                        if t_2 < t_1:
                            while len(data[count_2]) < len(data[count_1]):
                                data[count_2].append(float('nan'))
                            
                        elif t_2 > t_1:
                            while len(data[count_2]) > len(data[count_1]):
                                data[count_1].append(float('nan'))
                            
                    elif ignore_len_errors == "Crop":
                        if t_2 < t_1:
                            while len(data[count_2]) < len(data[count_1]):
                                del data[count_1][-1]
                            
                        elif t_2 > t_1:
                            while len(data[count_2]) > len(data[count_1]):
                                del data[count_2][-1]
                            
                    else:
                        raise ValueError('ignore_len_error should be "Ignore", "Crop", "Extend", or "Error".')
    
    return data, sample_rates, temperatures, humidities, datetimes, filtered_names



def plot_figure(data, times, time, input_type, plot_parameter, file_names, target_directory, lol_structure, lol_labels, change_type, plot_option,
                temperatures=[], humidities=[], datetimes=[], plot_average=True, legend_option=True, show=True, get_av_data=False, Show_T_RH=False, simultaneous=False, error_lines='N',
                get_error=False, calibration_args=[False], adjust_by_mass=False, av_option='mean', label_sep_runs=False, print_drops='N', drop_stats='N', save_csv=False, dpi=200):
    
    """ Plots data with many options, for more information see the comments by each option in __main__ """
    
    color_list = ["#1f77b4", "#ff7f0e", "#2ca02c", "c", "m", "y", "darkgoldenrod", "yellowgreen", "indigo", "r", "g", "b"]  # For colourblind accessibility
    loosely_dashed = (15, 15)

    if calibration_args[0]:
        calibration, step_times, step_middles, peak_indices, peak_heights, time_step = calibration_args
    
    if get_error:
        error_lines = 'A'

    if plot_option != "Box" and plot_option != "Trace" and plot_option != "Both":
        raise ValueError('plot_option should be "Box" for box plot, "Trace" for just the trace, or "Both" for both plots.')
    
    if show:
        fig, ax1 = plt.subplots(dpi=dpi)  # Will fix at regular dimensions: figsize=(6, 4)
        
        if Show_T_RH and len(temperatures) > 0 and len(humidities) > 0:
            av_temp, av_RH = stats.mean(temperatures), stats.mean(humidities)
            
            if len(temperatures) > 1 and len(humidities) > 1:
                err_temp = stats.stdev(temperatures) # SD
                err_RH = stats.stdev(humidities) # SD
                # err_temp = stats.stdev(temperatures) / np.sqrt(len(temperatures)) # SEM
                # err_RH = stats.stdev(humidities) / np.sqrt(len(humidities)) # SEM
                temp_round, RH_round = decimal_place(err_temp), decimal_place(err_RH)
                text_to_display = f"T = {round(av_temp, temp_round)} ± {round(err_temp, temp_round)} °C\n RH = {round(av_RH, RH_round)} ± {round(err_RH, RH_round)}%"
            else:
                text_to_display = f"T = {round(av_temp, 2)} °C\n RH = {round(av_RH, 2)}%"
            
            plt.text(0.95, 0.9, text_to_display, horizontalalignment='right', verticalalignment='top', transform=plt.gca().transAxes)
        
        if plot_option == "Both":
            gs = GridSpec(1, 2, width_ratios=[3, 1])
            plt.subplot(gs[0, 0])
        
        if plot_option != "Box":
                plt.plot([0, time], [0, 0], '--', linewidth=1.0, color='black')
    
    box_data = []
    
    if input_type == "dir" or input_type == "lis":
        check_ax2 = None
        
        for d in range(len(data)):
            # print(f'{file_names[d]}, {np.mean(data[d][last_non_nan_index-time_steps_to_average:last_non_nan_index]) - np.mean(data[d][:time_steps_to_average])}') # End - Start
            
            if plot_option != "Box" and not plot_average and show:
                if simultaneous:
                    if d == 0:
                        check_ax1 = file_names[d].split('_')[1]
                        ax1.set_ylabel(get_axis_title(check_ax1, adjust_by_mass))
                        ax1.plot(times[d], data[d], '-', linewidth=1.0, label=file_names[d], color=color_list[d % len(color_list)])
                        
                        if print_drops != 'N': # Printing individual drop changes
                            print_charge_drop(data, d, file_names[d], temperatures[d], humidities[d], datetimes[d], print_drops=print_drops)
                        
                    elif file_names[d].split('_')[1] == check_ax1:
                        ax1.plot(times[d], data[d], '-', linewidth=1.0, label=file_names[d], color=color_list[d % len(color_list)])
                        
                        if print_drops != 'N': # Printing individual drop changes
                            print_charge_drop(data, d, file_names[d], temperatures[d], humidities[d], datetimes[d], print_drops=print_drops)
                        
                    elif check_ax2 is None:
                        if calibration_args[0]:
                            for s, step in enumerate(step_heights):
                                ax1.plot([step_times[s], step_times[s]], [step_middles[s] - step / 2, step_middles[s] + step / 2], '-', color='black')
                                
                        ax2 = ax1.twinx()
                        check_ax2 = file_names[d].split('_')[1]
                        ax2.set_ylabel(get_axis_title(check_ax2, adjust_by_mass))
                        ax2.plot(times[d], data[d], '-', linewidth=1.0, label=file_names[d], color=color_list[d % len(color_list)])
                        
                        if print_drops != 'N':
                            print_charge_drop(data, d, file_names[d], temperatures[d], humidities[d], datetimes[d], print_drops=print_drops)
                        
                    elif file_names[d].split('_')[1] == check_ax2:
                        ax2.plot(times[d], data[d], '-', linewidth=1.0, label=file_names[d], color=color_list[d % len(color_list)])
                        
                        if print_drops != 'N': # Printing individual drop changes
                            print_charge_drop(data, d, file_names[d], temperatures[d], humidities[d], datetimes[d], print_drops=print_drops)
                        
                    else:
                        print(f"WARNING! {file_names[d]} not plotted as it doesn't match either axis of {check_ax1} or {check_ax2}")
                else:
                    plt.plot(times, data[d], '-', linewidth=1.0, label=file_names[d])
                    
                    if print_drops != 'N': # Printing individual drop changes
                        print_charge_drop(data, d, file_names[d], temperatures[d], humidities[d], datetimes[d], print_drops=print_drops)
            
            box_data.append(get_dif(data[d], change_type))
        
        if calibration_args[0] and show:
            for row_num in range(len(peak_indices)):
                if all(len(peak_heights[i]) == len(peak_heights[0]) for i in range(len(peak_heights))):
                    peak_indices = np.array(peak_indices)
                    ax2.plot(peak_indices[row_num] * time_step, [data[row_num + 1][i] for i in peak_indices[row_num]], 'x', color='black')
                    
                else:
                    raise ValueError(f"The number of peaks picked out for each ring do not match: {[len(peak_heights[i]) for i in range(len(peak_heights))]}")
        
        if plot_average:
            averaged_data, av_times, average_minus_error, average_plus_error = average_data(data, times, error_lines=error_lines, av_option=av_option)
            
            if show:
                plt.plot(av_times, averaged_data, '-', linewidth=1.0, label="Average Trace", color=color_list[0])
                
                if print_drops != 'N': # Printing individual drop changes
                            print_charge_drop([averaged_data], 0, 'Average', print_drops=print_drops)
                
                if error_lines == 'A':
                    plt.fill_between(av_times, average_minus_error, average_plus_error, color=color_list[0], alpha=0.3)
                    
                elif error_lines == 'L':
                    plt.plot(av_times, average_minus_error, '--', dashes=loosely_dashed, linewidth=0.5, label="Uncertainty", color=color_list[0])
                    plt.plot(av_times, average_plus_error, '--', dashes=loosely_dashed, linewidth=0.5, color=color_list[0])
                    
                elif error_lines != 'N':
                    raise ValueError(f'error_lines should be area (A), line (L), or none (N) but instead: {error_lines}')
            
            if save_csv: # Save averaged data to CSV if requested
                csv_file_name = 'drop_plotter_output_data.csv'
                save_to_csv(av_times, averaged_data, os.path.join(target_directory, csv_file_name), labels=["Averaged Data"])
        
        else:  # Save individual traces to CSV if requested
            if save_csv:
                csv_file_name = 'drop_plotter_output_data.csv'
                save_to_csv(times, data, os.path.join(target_directory, csv_file_name), labels=file_names)
                
    elif input_type == "lol" or input_type == "cat":
        count, counter = 0, 0
        
        for counter2, d in enumerate(data):
            dif = get_dif(d, change_type)
            sep_runs_opacity = calculate_opacity(file_names, counter2, lol_structure, counter)
            # sep_runs_opacity = 1 - 0.5 * ((float(file_names[counter2].split('.')[0].split('_')[-1]) - 1) / max([lol_structure[counter], int(file_names[0].split('.')[0].split('_')[-1])]))
            
            if count == 0:
                if plot_option != "Box":
                    if show:
                        plt.plot([0, time], [0, 0], '--', linewidth=1.0, color='black')
                    
                    if plot_average:
                        start_index = sum(lol_structure[:counter])
                        end_index = start_index + lol_structure[counter]
                        averaged_data, av_times, average_minus_error, average_plus_error = average_data(data, times, start_index, end_index, error_lines=error_lines, av_option=av_option, drop_stats=drop_stats)
                        
                        if show:
                            plt.plot(av_times, averaged_data, '-', linewidth=1.0, label=lol_labels[counter], color=color_list[counter % len(color_list)])
                            
                            if save_csv : # Save averages
                                csv_file_name = 'drop_plotter_output_data.csv'
                                save_to_csv(av_times, averaged_data, os.path.join(target_directory, csv_file_name), labels=[lol_labels[counter]], append_columns=True)
                            
                            if print_drops != 'N': # Printing individual drop changes
                                print_charge_drop([averaged_data], 0, name=lol_labels[counter], print_drops=print_drops)
                            
                            if drop_stats:
                                print(f'{lol_labels[counter]} -> first: {averaged_data[0]}, last: {averaged_data[-1]}')  # Printing the beginning and end of each averaged trace
                            
                            if error_lines == 'A':
                                plt.fill_between(av_times, average_minus_error, average_plus_error, color=color_list[counter], alpha=0.3)
                                
                            elif error_lines == 'L':
                                plt.plot(av_times, average_minus_error, '--', dashes=loosely_dashed, linewidth=0.5, color=color_list[counter % len(color_list)])
                                plt.plot(av_times, average_plus_error, '--', dashes=loosely_dashed, linewidth=0.5, color=color_list[counter % len(color_list)])
                                
                            elif error_lines != 'N':
                                raise ValueError(f'error_lines should be area (A), line (L), or none (N) but instead: {error_lines}')
                    
                    elif show:
                        if plot_average or not label_sep_runs:
                            plt.plot(times, d, '-', linewidth=1.0, color=color_list[counter % len(color_list)], label=lol_labels[counter])
                            if print_drops != 'N':
                                print_charge_drop([d], 0, name=lol_labels[counter], print_drops=print_drops)
                        else:
                            plt.plot(times, d, '-', linewidth=1.0, color=color_list[counter % len(color_list)], label=file_names[counter2], alpha=sep_runs_opacity)
                            if print_drops != 'N':
                                print_charge_drop([d], 0, name=file_names[counter2], print_drops=print_drops)
                
                box_data.append([dif])
            
            else:
                if plot_option != "Box" and not plot_average and show:
                    if label_sep_runs:
                        plt.plot(times, d, '-', linewidth=1.0, color=color_list[counter % len(color_list)], label=file_names[counter2], alpha=sep_runs_opacity)
                        if print_drops != 'N':
                            print_charge_drop([d], 0, name=file_names[counter2], print_drops=print_drops)
                    else:
                        plt.plot(times, d, '-', linewidth=1.0, color=color_list[counter % len(color_list)])
                        if print_drops != 'N':
                            print_charge_drop([d], 0, name=file_names[counter2], print_drops=print_drops)
                
                box_data[-1].append(dif)
                
            count += 1
            
            if count >= lol_structure[counter]:
                count = 0
                counter += 1
    
        if save_csv and not plot_average: # Save individual traces to CSV if requested
            csv_file_name = 'drop_plotter_output_data.csv'
            save_to_csv(times, data, os.path.join(target_directory, csv_file_name), labels=file_names)    
    
    if show:
        if plot_option != "Box":
            plt.xlabel("Time / s")
            
            if simultaneous:
                if legend_option:
                    ax1.legend(frameon=True, loc="upper right")
                    ax2.legend(frameon=True, loc="lower right")
                
            else:
                plt.ylabel(get_axis_title(plot_parameter, adjust_by_mass))
                if legend_option:
                    plt.legend(frameon=True, bbox_to_anchor=(1, 0.6))  # Option: loc="lower left" or loc="upper right", bbox_to_anchor=(1, 0.9) (shifts it down a little)
                    
            ax = plt.gca()
            ax.set_xlim([0, 5])
            left_subplot_ylim = plt.gca().get_ylim()  # Extracting the y axis limits
        
        if plot_option == "Both":
            plt.subplot(gs[0, 1])
        
        if not simultaneous:
            plt.ylabel(get_axis_title(plot_parameter, adjust_by_mass))

        if plot_option != "Trace":
            box = plt.boxplot(box_data, showfliers=True)
            plt.xticks(range(1, len(lol_labels) + 1), lol_labels, rotation=90)
        
        plt.subplots_adjust(wspace=0.7)
        save_name_png = os.path.join(target_directory, target_directory.split("\\")[-1] + "_" + plot_parameter + "_" + plot_option + ".png")
        save_name_pdf = os.path.join(target_directory, target_directory.split("\\")[-1] + "_" + plot_parameter + "_" + plot_option + ".pdf")
        plt.savefig(fname=save_name_png, format='png', bbox_inches='tight', dpi=dpi)  # Save as PNG
        plt.savefig(fname=save_name_pdf, format='pdf', bbox_inches='tight', dpi=dpi)  # Save as PDF
        plt.show()
    
    if get_av_data:
        if plot_average:
            if get_error:
                return av_times, averaged_data, average_minus_error, average_plus_error
                
            return av_times, averaged_data
            
        else:
            print(f"No Averaged data as plot_average is: {plot_average}")
    
    if plot_average:
        time_out = av_times
    else:
        time_out = times
        
    if get_error:
        return data, time_out, average_minus_error, average_plus_error
    
    return data, time_out



if __name__ == "__main__":
    
    # path = "..\\Drops\\Fume_Hood_Tests"
    # path = "..\\Drops\\Fume_Hood_Tests"
    # path = "..\\Data for Faraday cup paper\\Labradorite_Scaling\\Charge_drops"
    path = "..\\Data for IEEE Paper\\Charging Drops Rig Material Investigation"
    # path = "C:\\Users\\vb22224\\OneDrive - University of Bristol\\Desktop\MAIN\\Data for Electrostatics Paper\\Triboelectric Charging Drops"
    input_type = "cat" # Directory (dir), categorised (cat), list (lis), or list of lists (lol).
    # Note: use input_type = "dir"and plot_average = False to get seperate T and RH for each drop
    # Note: the categorised (cat) input type requires filenames formatted such that the type is followed by underscore then index e.g. Atitlan_1.txt 
    plot_parameter = "Q" # Charge (Q), voltage (V), raw voltage (RV), high charage range (HQ), or very high charage range (VHQ)
    plot_option = "Trace" # Box plot (Box), The trace (Trace), or both (Both)
    change_type = "mm" # How the boxplot is measured: by the change from the start to finish (sf), by the difference between min and max (mm), or by mm - abs(sf) => (mmsf)
    ignore_len_errors = "Extend" # Should be "Crop" or "Extend" if you want to shorten data to shortest series or extend to the longest "Error" returns error. Note: if "Crop" selected this will affect the data plotted in the boxplot too
    file_names_lis = ['Thiovit_52.txt', 'Thiovit_52.txt', 'Thiovit_53.txt', 'Thiovit_54.txt', 'Thiovit_55.txt', 'Thiovit_56.txt', 'Thiovit_57.txt']   
    file_names_lol = [['PP_01.txt', 'PP_10.txt','PP_12.txt','PP_15.txt','PP_16.txt'], ['Flour_01.txt']]
    remove_files = ['PP_02.txt', 'Flour_37.txt', 'Carbon Black_01.txt', 'Carbon Black_02.txt', 'CB_01.txt', 'CB_02.txt', 'CB_03.txt', 'Thiovit_51.txt',]
    remove_files += ['PP_18.txt', 'Fuego_3.txt']
    lol_labels = ["Polypropylene", "Zerostat", "Rested"]
    spec_cat = [] # specify categories to include if input type is categorised (file names before the underscore), leave empty to include all
    # 'MGS-1', 'MGS-1C', 'JSC-1A', 'LMS-1', 'LSP-2', 'CM-E', 'Ano', 'Bas', 'Bro', 'Ilm', 'Oli'
    # '63-125', '63-75', '75-125', '80-125', '63-80'
    # 'Cont', 'Neg', 'Pos', 'Neut'
    
    trim = True # If True removes the beginning of the trace such that all traces start at the same time (required to calculate average trace)
    plot_average = True # If True plots the average trace instead of all individual traces
    legend_option = True # If true plots a legend
    av_option = 'mean' # How the average is calculated, should be 'mean', 'median', or 'bootstrap,
    label_sep_runs = True # If input_type = "cat" and plot_average = False, this will make the runs different hues and labelled, not available in simultaneous
    error_lines = 'A' # Shows the error when averaged, should be area (A), line (L), or none (N)
    manual_trim = {}
    # manual_trim.update({'Lab2Small_1.txt':0.5, 'JSC-1A_2.txt':3.8, 'MGS-1C_6.txt':3, 'MGS-1C_9.txt':4, '+1p5_3.txt':2, '+1p5_2.txt':3}) # Dictionary of file names and the time (in seconds) that you want to maually trim from the start
    # manual_trim.update({'JSC-1A_6.txt':1, 'MGS-1_1.txt':3.6, 'MGS-1_2.txt':3.2, 'MGS-1_4.txt':3.1, 'MGS-1_10.txt':3.2, 'MGS-1_12.txt':3.0, 'MGS-1_15.txt':3.0})
    manual_trim.update({'Lab2Small_1.txt':0.5, 'JSC-1A_2.txt':3.8, 'MGS-1C_6.txt':3, 'MGS-1C_9.txt':4, '+1p5_3.txt':2, '+1p5_2.txt':3})
    manual_trim.update({'Ano_3.txt':2.93, 'Ano_10.txt':3.25, 'Bas_10.txt':3.15, 'Oli_9.txt':2.6, 'Flour_02.txt':3.0, 'Flour_07.txt':2.0, 'PP_13.txt':3.4, 'Cellulose_01.txt':2.0, 'B01_06.txt':1.5})
    manual_trim.update({'B03_01.txt':1.0, 'B03_02.txt':1.0, 'B03_03.txt':1.0, 'B03_04.txt':1.0, 'B03_05.txt':1.0, 'B03_06.txt':1.0, 'B06_07.txt':2.0, 'B06_04.txt':2.0})
    store_dict, read_dict = False, False # Options to store or read from file the manual trim dictionary
    Show_T_RH = False # Whether or not to show Temperature and humidity on the plot (if multiple traces plotted than it is an average)
    print_drops = 'ALL' # Whether to print the change in each drop, can be calculated from the start and end (SE), the maximum and minimum (MM), the end minus the opposite extreme (EE), all (ALL), or not at all (N)
    drop_stats = False # Whether to print stats about the average drops
    adjust_by_mass = True
    adjust_for_decay, time_const = True, 310.5 # in [s] and comes from the conductivity of air
    base_time, tolerance_amp, tolerance_len = 1.0, 1.5, 300 # Number of seconds to use as the baseline, How many times greater the trace needs to be than baseline noise to register start over how many steps
    remove_noise = True # If true applis a notch filter at 50 Hz to remove "mains hum"
    
    data_col = 0 # Selects which column of data file to read (default is the first column = 0)
    simultaneous_plot = False # True plots both ring and cup together
    calibration = False # Picks the peaks for calibration, requires that: parameter_lis = ["Q", "RV", "RV"] and that it is of one dataset (ie input_type = "lis" and plot_average = False) 
    parameter_lis = ["Q", "RV", "RV"] # Charge (Q), voltage (V), or raw voltage (RV), require simultaneous_plot = True
    peak_std_dev_threshold = 2
    step_background_time, step_threshold, step_width = 1, 1.6, 0.2 # The time (/ s) to take a standard deviation of at the start of the trace, the threshold number of standard deviations to count as a step, time width of a step (/ s)
    save_csv = False # Weather to save the output data plotted to a csv
    dpi = 600

    ###########################################################################
    # Code to Make Single Plot
    
    if not simultaneous_plot:
        file_names, lol_structure, lol_labels = get_file_names(input_type, path, file_names_lis, file_names_lol, remove_files, lol_labels, spec_cat=spec_cat)
        data, sample_rates, temperatures, humidities, datetimes, filtered_names = read_and_convert_data(file_names, path, plot_parameter, ignore_len_errors, base_time, tolerance_amp, tolerance_len, trim, manual_trim, store_dict, read_dict, data_col, adjust_by_mass=adjust_by_mass, adjust_for_decay=adjust_for_decay, time_const=time_const, remove_noise=remove_noise)
        time_step, times, time = get_times(sample_rates, data)
        plot_figure(data, times, time, input_type, plot_parameter, filtered_names, path, lol_structure, lol_labels, change_type, plot_option, temperatures, humidities, datetimes, plot_average=plot_average, legend_option=legend_option, Show_T_RH=Show_T_RH, error_lines=error_lines, adjust_by_mass=adjust_by_mass, av_option=av_option, label_sep_runs=label_sep_runs, drop_stats=drop_stats, print_drops=print_drops, save_csv=save_csv, dpi=dpi)
    
    ###########################################################################
    # Code to plot cup and ring data together and calibrate
    
    if simultaneous_plot:
        peak_heights, peak_indices, collated_data, collated_times = [], [], [], []
        file_names, lol_structure, lol_labels = get_file_names(input_type, path, file_names_lis, file_names_lol, remove_files, lol_labels, spec_cat=spec_cat)
    
        for count, parameter in enumerate(parameter_lis):
            data_col = count
            data, sample_rates, temperatures, humidities, datetimes, filtered_names = read_and_convert_data(file_names, path, parameter, ignore_len_errors, base_time, tolerance_amp, tolerance_len, trim, manual_trim, store_dict, read_dict, data_col, adjust_by_mass=adjust_by_mass, adjust_for_decay=adjust_for_decay, time_const=time_const, remove_noise=remove_noise)
            time_step, times, time = get_times(sample_rates, data)
            data_out, times_out = plot_figure(data, times, time, input_type, parameter, filtered_names, path, lol_structure, lol_labels, change_type, plot_option, temperatures, humidities, plot_average=plot_average, legend_option=legend_option, Show_T_RH=Show_T_RH, show=False, dpi=dpi)
    
            for index, d in enumerate(data_out):
                times_out = np.arange(0, len(d) * time_step, time_step)
                collated_times.append(times_out)
                collated_data.append(d)
    
                if calibration:
                    if count == 0:
                        step_heights, step_times, step_middles = get_steps(d, time_step, step_background_time, step_threshold, step_width)
                    else:
                        peak_height_row, peak_index_row = get_peaks(d, peak_std_dev_threshold)
                        peak_heights.append(peak_height_row)
                        peak_indices.append(peak_index_row)
    
        if calibration:
            calibration_args = calibration, step_times, step_middles, peak_indices, peak_heights, time_step
        else:
            calibration_args = [False]
    
        # Possibly need to change this if you don't want to plot average but do want to plot multiple
        collated_data, collated_times, filtered_names = average_multiple(collated_data, collated_times, filtered_names, parameter_lis)
    
        plot_figure(collated_data, collated_times, time, "lis", "Q", filtered_names, path, lol_structure, lol_labels, change_type, plot_option, temperatures, humidities, plot_average=False, Show_T_RH=Show_T_RH, show=True, simultaneous=True, calibration_args=calibration_args, av_option=av_option, save_csv=save_csv, dpi=dpi)
    
        if calibration:
            calibration_constants(step_heights, peak_heights)