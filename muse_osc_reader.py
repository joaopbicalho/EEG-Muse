# ------------------------------------------------------------------------------
# Author: Joao Pedro Bicalho
# Date: 2024-04-25
# Description: Code to open UDP port and read and process OSC output from MUSE 
# headband. The code filters the data, computes the power spectral density (PSD) using 
# Welch's method, and calculates band powers for theta, alpha, beta, and gamma frequency bands.
# ------------------------------------------------------------------------------

import socket
import numpy as np
import mne
from pythonosc import osc_message
from pythonosc.osc_message import ParseError
from mne.filter import create_filter
from mne.time_frequency import psd_array_welch

# Initialize lists to store the raw data
tp9_data = []
af7_data = []
af8_data = []
tp10_data = []

# Epoch parameters
sfreq = 256  # Sampling frequency
epoch_length = 3 * sfreq  # 3 seconds
overlap = epoch_length // 2  # 50% overlap

# Function to handle parsed OSC messages
def osc_handler(address, *args):
    if address == "/eeg":
        try:
            tp9, af7, af8, tp10 = args[0], args[1], args[2], args[3]
            tp9_data.append(tp9)
            af7_data.append(af7)
            af8_data.append(af8)
            tp10_data.append(tp10)
            # print(f"Received data: {tp9}, {af7}, {af8}, {tp10}")
        except Exception as e:
            print(f"Error parsing arguments: {e}")

# Set up the UDP socket
UDP_IP = "0.0.0.0"
UDP_PORT = 5000
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

print(f"Listening on {UDP_IP}:{UDP_PORT}")

# Disable MNE verbose output
mne.set_log_level('WARNING')

# Main loop to receive data and parse as OSC messages
while True:
    data, addr = sock.recvfrom(1024)
    try:
        msg = osc_message.OscMessage(data)
        osc_handler(msg.address, *msg.params)
    except ParseError as e:
        print(f"Failed to parse OSC message: {e}")

   # Process data when enough samples are collected for one epoch
    if len(tp9_data) >= epoch_length:
        # Extract the epoch data
        tp9_epoch = tp9_data[:epoch_length]
        af7_epoch = af7_data[:epoch_length]
        af8_epoch = af8_data[:epoch_length]
        tp10_epoch = tp10_data[:epoch_length]

        # Convert data to numpy array
        tp9_array = np.array(tp9_epoch)
        af7_array = np.array(af7_epoch)
        af8_array = np.array(af8_epoch)
        tp10_array = np.array(tp10_epoch)

    
        # Create filter
        l_freq = 0.1
        h_freq = 15
        tp9_filtered = mne.filter.filter_data(tp9_array, sfreq, l_freq, h_freq, method='iir')
        af7_filtered = mne.filter.filter_data(af7_array, sfreq, l_freq, h_freq, method='iir')
        af8_filtered = mne.filter.filter_data(af8_array, sfreq, l_freq, h_freq, method='iir')
        tp10_filtered = mne.filter.filter_data(tp10_array, sfreq, l_freq, h_freq, method='iir')

         # Apply notch filter (IIR)
        notch_freq = 60
        tp9_filtered = mne.filter.notch_filter(tp9_filtered, sfreq, notch_freq, method='iir')
        af7_filtered = mne.filter.notch_filter(af7_filtered, sfreq, notch_freq, method='iir')
        af8_filtered = mne.filter.notch_filter(af8_filtered, sfreq, notch_freq, method='iir')
        tp10_filtered = mne.filter.notch_filter(tp10_filtered, sfreq, notch_freq, method='iir')

        # print("Filtered data:")
        # print(tp9_filtered[:10], af7_filtered[:10], af8_filtered[:10], tp10_filtered[:10])

         # Average the filtered data from all channels
        avg_filtered = (tp9_filtered + af7_filtered + af8_filtered + tp10_filtered) / 4

        # Frequency band ranges
        frequency_bands = {
            'theta': (4, 8),
            'alpha': (8, 12),
            'beta': (12, 15),
            'gamma': (30, 40)
        }

        # Parameters for psd_array_welch
        fmin, fmax = 1, 40  # Adjust according to your frequency bands
        n_fft = epoch_length  # Number of points for FFT (adjust as needed)
        n_per_seg = epoch_length  # Number of points per segment (adjust as needed)


        # Compute PSD using psd_array_welch for the averaged signal
        psd, freqs = psd_array_welch(avg_filtered, sfreq, fmin=fmin, fmax=fmax, n_fft=n_fft, n_per_seg=n_per_seg)

        band_powers = {}

        # Compute band powers
        for band, (fmin_band, fmax_band) in frequency_bands.items():
            band_indices = np.where((freqs >= fmin_band) & (freqs < fmax_band))[0]
            band_powers[band] = np.mean(psd[band_indices])


        #FOR INDIVIDUAL ELECTRODE PSD OUTPUT
        # Initialize dictionaries to store PSD and band powers
        #psd_dict = {}
       
        # band_powers = {band: np.zeros(4) for band in frequency_bands}
        # # Compute PSD using psd_array_welch for each channel
        # channels = ['tp9', 'af7', 'af8', 'tp10']
        # for i, channel in enumerate(channels):
        #     filtered_data = eval(f"{channel}_filtered")  # Assuming tp9_filtered, af7_filtered, etc. are variables
        #     psd, freqs = psd_array_welch(filtered_data, sfreq, fmin=fmin, fmax=fmax, n_fft=n_fft, n_per_seg=n_per_seg)
        #     psd_dict[channel] = psd

        # # Compute band powers
        # for band, (fmin_band, fmax_band) in frequency_bands.items():
        #     band_indices = np.where((freqs >= fmin_band) & (freqs < fmax_band))[0]
        #     for i, channel in enumerate(channels):
        #         band_powers[band][i] = np.mean(psd_dict[channel][band_indices]) 
                
        # Print frequency band power
        print("Frequency band power (uV^2/Hz):")
        print("Theta (4-8 Hz):", band_powers['theta'])
        print("Alpha (8-12 Hz):", band_powers['alpha'])
        print("Beta (12-15 Hz):", band_powers['beta'])
        print("Gamma (30-40 Hz):", band_powers['gamma'])

        # Clear the lists to prepare for the next batch of data
        tp9_data = tp9_data[overlap:]
        af7_data = af7_data[overlap:]
        af8_data = af8_data[overlap:]
        tp10_data = tp10_data[overlap:]

        # Main loop will keep running and filtering data as it is received
