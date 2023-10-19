import numpy as np
from scipy.optimize import minimize
import sys
import wave
from scipy.io import wavfile
from scipy.signal import spectrogram
from scipy.signal import hilbert
import scipy.signal as signal
import matplotlib.pyplot as plt
from scipy.signal import correlate
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from scipy.optimize import minimize
from scipy.signal import butter, lfilter
from scipy.signal import fftconvolve
from pydub import AudioSegment
from art import *





# Heading

print("===================Triangulation Simulation=========================")
print("This program performs acoustic triangulation using four microphones")
print("m1, m2, m3, and m4, located on an A1 paper. A sound source,")
print("located at [sound_source_x, sound_source_y, sound_source_z], ")
print("emits a signal which is recorded by the microphones at the time of ")
print("arrival. The program uses the time difference of arrival between ")
print("microphones to triangulate the location of the source source ")
print("with m1 as the reference microphone.")
print("====================================================================\n")

# Set microphone m1 as default microphone
m1 = [-0.05, -0.05, 0]
m2 = []
m3 = []
m4 = []

initial_source_position = np.array([0.1,0.1,0.1])

# A1 dimensions 
# Dimensions of the rectangular area
length = 0.841  # Length of the area in meters
width = 0.594   # Width of the area in meters

microphone_positions = np.array([[], [], [], []])
temperature = 25  # Default temperature in Celsius
speed_of_sound = 331.4 * np.sqrt(1 + (temperature / 273.15))
source_coord = [1,1,1]
source_position = np.array(source_coord)  # Default position (1m above origin)
default = ""

while default == "":
    default = input("Use default settings? y/n:\n")
    
    if default == "y":
        m2 = [0.05, -0.05, 0]
        m3 = [0.05, 0.05, 0]
        m4 = [-0.05, 0.05, 0]
        microphone_positions = np.array([m1, m2, m3, m4])
        print("===================Temperature & Speed of Sound=====================")
        print(f"Default ambient {temperature}.")  # Default temperature in Celsius
        print_speed = "{:.2f}".format(speed_of_sound)
        print(f"Default speed of sound at {temperature} degrees Celsius is "+ print_speed + ".")
        print("====================================================================")
        print()
        print("==================Default Microphone Positions======================")
        print("Microphone m1 is positioned at", m1)
        print("Microphone m2 is positioned at", m2)
        print("Microphone m3 is positioned at", m3)
        print("Microphone m4 is positioned at", m4)
        print("====================================================================")
        print("Loading...")
        print()
    elif default == "n":

        print("===================Temperature & Speed of Sound=====================")
        temperature = float(input("Enter ambient temperature (degree Celsius):\n"))
        speed_of_sound = 331.4 * np.sqrt(1 + (temperature / 273.15))
        print_speed = "{:.2f}".format(speed_of_sound)
        print(f"The speed of sound at {temperature} degrees Celsius is "+ print_speed+".")
        print("====================================================================\n")

        print("=====================Microphone Positions===========================")
        mic_dist_str = ""
        while mic_dist_str == '':
            mic_dist_str = input("Enter the distance between microphones (m):\n")
            if mic_dist_str != '':
                mic_dist = float(mic_dist_str)
                if mic_dist !=0:
                    m1 = [-mic_dist/2, -mic_dist/2, 0]
                    m2 = [mic_dist/2, -mic_dist/2, 0]
                    m3 = [mic_dist/2, mic_dist/2, 0]
                    m4 = [-mic_dist/2, mic_dist, 0]
            else:
                print("Distance between microphones cannot be zero.")
        microphone_positions = np.array([m1, m2, m3, m4])

        print()
        print("Microphone m1 is positioned at", m1,".")
        print("Microphone m2 is positioned at", m2, ".")
        print("Microphone m3 is positioned at", m3, ".")
        print("Microphone m4 is positioned at", m4, ".")
        print()
        print("====================================================================\n")
        print("Loading...")
        print(1)
    else:
        default = ""

# Calculate the center of each microphone pair
center_m2_m3 = ((m2[0] + m3[0]) / 2, (m2[1] + m3[1]) / 2, (m2[2] + m3[2]) / 2)
center_m1_m4 = ((m1[0] + m4[0]) / 2, (m1[1] + m4[1]) / 2, (m1[2] + m4[2]) / 2)

# Calculate the distance between the centers of the two pairs
microphone_distance = np.sqrt(
    (center_m2_m3[0] - center_m1_m4[0])**2 +
    (center_m2_m3[1] - center_m1_m4[1])**2 +
    (center_m2_m3[2] - center_m1_m4[2])**2
)

# Print the distance
# print(f"Distance from center of m1-m2 pair to center of m3-m4 pair: {microphone_distance} meters")
# Initialize parameters

# Load the two WAV files
file_path1 = '/home/bonga/Documents/EEE3097S_Project/EEE3097S_Assignment_04_Third_Progress_Report/5_1-output.wav'
file_path2 = '/home/bonga/Documents/EEE3097S_Project/EEE3097S_Assignment_04_Third_Progress_Report/5_2-output.wav'
file_path3 = '/home/bonga/Documents/EEE3097S_Project/EEE3097S_Assignment_04_Third_Progress_Report/6-2.wav'
file_path4 = '/home/bonga/Documents/EEE3097S_Project/EEE3097S_Assignment_04_Third_Progress_Report/6-1.wav'

def butter_bandpass(lowcut, highcut, fs, order=6):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=6):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

# trim audio

# Load the audio file
audio3 = AudioSegment.from_file(file_path3, format="wav")
audio4 = AudioSegment.from_file(file_path4, format="wav")

# Define the duration to trim (in milliseconds)
trim_duration = 2000  # 1000 milliseconds = 1 second

# Trim the audio file
trimmed_audio3 = audio3[trim_duration:]
trimmed_audio4 = audio4[trim_duration:]

# Export the trimmed audio
trimmed_audio3.export(out_f = "trimmed_audio3.wav", format="wav")
trimmed_audio4.export(out_f = "trimmed_audio4.wav", format="wav")

file_path1 = '/home/bonga/Documents/EEE3097S_Project/Acoustic-Triangulation-System/trimmed_audio3.wav'
file_path2 = '/home/bonga/Documents/EEE3097S_Project/Acoustic-Triangulation-System/trimmed_audio4.wav'

# Read the two audio files
fs, audio1 = wavfile.read(file_path1)
fs, audio2 = wavfile.read(file_path2)

# Define the time axis for plotting
time1 = np.arange(len(audio1)) / fs
time2 = np.arange(len(audio2)) / fs

lowcut = 10
highcut = 3000

# Filter and keep only the specified frequency range
filtered_audio1 = butter_bandpass_filter(audio1, lowcut, highcut, fs*0.5)
filtered_audio2 = butter_bandpass_filter(audio2, lowcut, highcut, fs*0.5)

# Save the filtered audio signals to new WAV files
wavfile.write('filtered_audio1.wav', fs, filtered_audio1.astype(np.int16))
wavfile.write('filtered_audio2.wav', fs, filtered_audio2.astype(np.int16))

# Read the filtered audio signal from a WAV file
fs, filtered_audio12 = wavfile.read('filtered_audio1.wav')
fs, filtered_audio22 = wavfile.read('filtered_audio2.wav')
# Create a time axis for plotting
time1 = np.arange(len(filtered_audio12)) / fs
time2 = np.arange(len(filtered_audio22)) / fs

min_ln = min(len(time1), len(time2))
filtered_audio12 = filtered_audio12[:min_ln]
filtered_audio22 = filtered_audio22[:min_ln]

def gcc_phat(audio12, audio22):
    # Calculate the FFT of the signals
    fft_filtered_audio12 = np.fft.fft(audio12)
    fft_filtered_audio22 = np.fft.fft(audio22)
    
    min_fft = min(len(fft_filtered_audio12), len(fft_filtered_audio22))

    fft_filtered_audio12 = fft_filtered_audio12[:min_fft]
    fft_filtered_audio22 = fft_filtered_audio22[:min_fft]
    # Calculate the cross-correlation in the frequency domain
    cross_correlation = np.multiply(fft_filtered_audio12, np.conj(fft_filtered_audio22))
    # cross_correlation = correlate(fft_filtered_audio12, fft_filtered_audio22, mode='full')  / np.max(np.abs(correlate(filtered_audio12, filtered_audio22, mode='full')))
    cross_correlation /= np.abs(cross_correlation)

    # Calculate the inverse FFT to get the time-domain result
    gcc_phat_result = np.fft.ifft(cross_correlation)
    
    return gcc_phat_result

# Compute GCC-PHAT cross-correlation
gcc_phat_result = gcc_phat(filtered_audio22, filtered_audio12)

# Find the time delay (index of maximum value)
time_delay_index = np.argmax(np.abs(gcc_phat_result))

# Convert the time delay index to seconds
time_delay = time_delay_index / fs
print(f"Time delay Preprocessed: {time_delay}")
if time_delay > 0.01:
    time_delay = time_delay/100000

# Print the time delay
print(time_delay)
print(f"Time Delay: {time_delay} seconds")

# Calculate the angle of arrival in radians
ratio12 = (time_delay * speed_of_sound)/microphone_distance
aoa = np.arccos(ratio12)

# Convert radians to degrees
aoa_degrees = np.degrees(aoa)

# Calculate the direction vector from the first microphone to the source
direction_vector = np.array([0.1*np.cos(aoa), 0.1*np.sin(aoa), 0])
print(f"Angle of Arrival (degrees): {aoa_degrees}")
print(f"Direction vector: {direction_vector}")

# Repeat the process for the width
distance_width = (width / np.tan(np.radians(90 - aoa_degrees))) / 10

if 0 <= distance_width <= width:
    print(f"Estimated distance to the source (y-coordinate): {distance_width:.4f} meters")
else:
    print("Source is not within the restricted area (width).")

# Estimate the distance to the source
distance = direction_vector[1] * 1.606

# Ensure the source is within the restricted area
if 0 <= distance <= length:
    print(f"Estimated distance to the source (x-coordinate): {distance:.4f} meters")
else:
    print("Source is not within the restricted area.")

time1 = np.arange(len(filtered_audio12)) / fs
time2 = np.arange(len(filtered_audio22)) / fs

# # Plot the filtered audio signal
plt.figure(figsize=(10, 6))
plt.plot(time1, filtered_audio12, lw=0.5)
plt.title('Filtered Audio Signal 1')
plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude')
plt.grid(True)
plt.show()

# Plot the filtered audio signal
plt.figure(figsize=(10, 6))
plt.plot(time2, filtered_audio22, lw=0.5)
plt.title('Filtered Audio Signal 2')
plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude')
plt.grid(True)
plt.show()

# # Plot the filtered audio signals
# plt.figure(figsize=(12, 6))
# plt.subplot(2, 1, 1)
# plt.plot(time1, audio1, 'b')
# plt.title('Filtered Audio Signal 1')
# plt.xlabel('Time (s)')
# plt.ylabel('Amplitude')

# plt.subplot(2, 1, 2)
# plt.plot(time2, audio2, 'r')
# plt.title('Filtered Audio Signal 2')
# plt.xlabel('Time (s)')
# plt.ylabel('Amplitude')

# plt.tight_layout()
# plt.show()

# Make sure both signals are of the same length
min_length = min(len(filtered_audio12), len(filtered_audio22))
filtered_audio12 = filtered_audio12[:min_length]
filtered_audio22 = filtered_audio22[:min_length]

# Perform FFT on both signals
fft_filtered_audio12 = np.fft.fftshift(filtered_audio12)
fft_filtered_audio22 = np.fft.fftshift(filtered_audio22)
frequencies = np.fft.fftfreq(min_length, 1 / fs)

# Plot the magnitude spectrum of both signals
# plt.figure(figsize=(12, 6))
# plt.subplot(2, 1, 1)
# plt.title('Frequency Domain - Signal 1')
# plt.plot(frequencies, np.abs(fft_filtered_audio12), lw=0.5)
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('Magnitude')
# plt.grid(True)

# plt.subplot(2, 1, 2)
# plt.title('Frequency Domain - Signal 2')
# plt.plot(frequencies, np.abs(fft_filtered_audio22), lw=0.5)
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('Magnitude')
# plt.grid(True)

# plt.tight_layout()
# plt.show()

# # Define the low-pass filter parameters
# low_cutoff = 10  # Lower frequency cutoff (Hz)
# high_cutoff = 800  # Higher frequency cutoff (Hz)
# nyquist = 0.5 * fs
# low = low_cutoff / nyquist
# high = high_cutoff / nyquist

# # Design the low-pass Butterworth filter
# order = 4  # Filter order
# b, a = signal.butter(order, [low, high], btype='band')

# # Apply the filter to the audio signal
# filtered_signal = signal.lfilter(b, a, filtered_audio12)

# # Write the filtered audio to a new WAV file
# wavfile.write('filtered_audio.wav', fs, filtered_signal.astype(np.int16))

# # Plot the original and filtered signals in the time domain
# plt.figure(figsize=(12, 6))
# plt.plot(filtered_audio12, label='Original Signal', lw=0.5)
# plt.plot(filtered_signal, label='Filtered Signal', lw=0.5)
# plt.title('Original vs Filtered Signal')
# plt.xlabel('Sample')
# plt.ylabel('Amplitude')
# plt.legend()
# plt.grid(True)
# plt.show()

# min_ln = min(len(filtered_audio12), len(filtered_audio22))

# filtered_audio12 = filtered_audio12[:min_ln]
# filtered_audio22 = filtered_audio22[:min_ln]

# corr = correlate(filtered_audio12, filtered_audio22, mode='full') / np.max(np.abs(correlate(filtered_audio12, filtered_audio22, mode='full')))

# # Calculate the time delays
# time_delays = np.linspace(-min_ln / fs, min_ln / fs, 2 * min_ln - 1)

# # Find the delay with the highest correlation
# delay = time_delays[np.argmax(corr)]

# print("Estimated Time Delay:", delay, "seconds")

# # Plot the cross-correlation result
# plt.figure(figsize=(10, 6))
# plt.plot(time_delays, corr, lw=0.5)
# plt.title('Cross-Correlation')
# plt.xlabel('Time Delay (s)')
# plt.ylabel('Normalized Correlation')
# plt.grid(True)
# plt.show()

# Compute the GCC-PHAT cross-correlation
# def gcc_phat(signal1, signal2):
#     fft_signal1 = np.fft.fft(signal1)
#     fft_signal2 = np.fft.fft(signal2)
#     cross_correlation = fft_signal1 * np.conj(fft_signal2)
#     cross_correlation /= np.abs(cross_correlation)
#     inverse_correlation = np.fft.ifft(cross_correlation)
#     gcc_phat_result = np.abs(inverse_correlation)
#     return gcc_phat_result

# gcc_max = max(len(filtered_audio12), len(filtered_audio22))
# pad_12 = 0
# pad_22 = 0
# if gcc_max == len(filtered_audio22):
#     pad_12 = len(filtered_audio22) - len(filtered_audio12)
#     filtered_audio12 = np.pad(filtered_audio12, (0,pad_12), 'constant')
# else:
#     pad_22 = len(filtered_audio12) - len(filtered_audio22)
#     filtered_audio12 = np.pad(filtered_audio22, (0,pad_22), 'constant')
    
# gcc_phat_result = gcc_phat(filtered_audio12, filtered_audio22)

# # Find the time delay with the highest cross-correlation
# time_delay2 = np.argmax(gcc_phat_result) - len(filtered_audio12)

# # Calculate the corresponding time delay in seconds
# time_delay_seconds = time_delay2 / 

# print(f"Time Delay: {time_delay2} samples")
# print(f"Time Delay (seconds): {time_delay_seconds} s")


# Create ASCII art text for microphone and source symbols
# microphone = text2art("M")
# source = text2art("S")

# # Create an empty grid
# grid_size = 20  # Define the size of the grid
# grid = [[" " for _ in range(grid_size)] for _ in range(grid_size)]

# # Place microphones and source on the grid
# grid[0][0] = microphone  # Microphone M1 at (0, 0)
# grid[0][-1] = microphone  # Microphone M2 at (0, 19)
# grid[-1][-1] = microphone  # Microphone M3 at (19, 19)
# grid[-1][0] = microphone  # Microphone M4 at (19, 0)
# grid[9][9] = source  # Sound source S at (9, 9)

# # Print the grid
# for row in grid:
#     print("".join(row))

# Extract X and Y coordinates for microphones and source
mic_x1 = m1[0]
mic_y1 = m1[1]

mic_x2 = m2[0] 
mic_y2 = m2[1]

mic_x3 = m3[0] 
mic_y3 = m3[1]

mic_x4 = m4[0] 
mic_y4 = m4[1]

mic_y = zip(*microphone_positions)
source_x = distance
source_y = distance_width

# Create a scatter plot
plt.scatter(mic_x1, mic_y1, label="m1", marker="o", color="blue", s=100)
plt.scatter(mic_x2, mic_y2, label="m2", marker="o", color="green", s=100)
plt.scatter(mic_x3, mic_y3, label="m3", marker="o", color="red", s=100)
plt.scatter(mic_x4, mic_y4, label="m4", marker="o", color="purple", s=100)
plt.scatter(source_x, source_y, label="Sound Source", marker="x", color="red", s=200)

# Add labels for microphones and source
for i, txt in enumerate(range(1, len(microphone_positions) + 1)):
    plt.annotate(txt, (0, 0), fontsize=12, ha="center", va="center", color="white")

plt.annotate("S", (source_x, source_y), fontsize=12, ha="center", va="center", color="white")

# Customize the plot
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.title("Sound Source and Microphone Locations")
plt.axhline(0, color="black", linewidth=1)
plt.axvline(0, color="black", linewidth=1)
plt.grid(True)

# Define the dimensions of the rectangle
width = 0.8
height = 0.5

# Calculate the coordinates of the rectangle's corners
half_width = width / 2
half_height = height / 2

x = [-half_width, half_width, half_width, -half_width, -half_width]
y = [-half_height, -half_height, half_height, half_height, -half_height]

# Create a plot
plt.plot(x, y, color='blue')

# Draw rectangle
plt.axhline(0, color='black', linewidth=1)
plt.axvline(0, color='black', linewidth=1)
plt.grid(True)

# Set the aspect ratio to be equal
plt.gca().set_aspect('equal', adjustable='box')

# Display the plot
plt.legend()
plt.show()