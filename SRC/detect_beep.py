import numpy as np
import matplotlib.pyplot as plt
import wave

# Load the two WAV files
file_path1 = '/home/bonga/Documents/EEE3097S_Project/EEE3097S_Assignment_04_Third_Progress_Report/5_2-output.wav'
file_path2 = '/home/bonga/Documents/EEE3097S_Project/EEE3097S_Assignment_04_Third_Progress_Report/5_1-output.wav'

# Open the WAV files
wav_file1 = wave.open(file_path1, 'rb')
wav_file2 = wave.open(file_path2, 'rb')

# Read the audio data
signal1 = np.frombuffer(wav_file1.readframes(-1), dtype=np.int16)
signal2 = np.frombuffer(wav_file2.readframes(-1), dtype=np.int16)

# Calculate the GCC-PHAT cross-correlation
def gcc_phat(signal1, signal2):
    # Calculate the FFT of the signals
    fft_signal1 = np.fft.fft(signal1)
    fft_signal2 = np.fft.fft(signal2)

    # Calculate the cross-correlation in the frequency domain
    cross_correlation = np.multiply(fft_signal1, np.conj(fft_signal2))
    cross_correlation /= np.abs(cross_correlation)

    # Calculate the inverse FFT to get the time-domain result
    gcc_phat_result = np.fft.ifft(cross_correlation)
    
    return gcc_phat_result

# Compute GCC-PHAT cross-correlation
gcc_phat_result = gcc_phat(signal1, signal2)

# Create a time array for the x-axis
frame_rate = wav_file1.getframerate()
time = np.linspace(0, len(gcc_phat_result) / frame_rate, len(gcc_phat_result))

# Find the time delay (index of maximum value)
time_delay_index = np.argmax(np.abs(gcc_phat_result))
print(time_delay_index)
# Convert the time delay index to seconds
time_delay = time_delay_index / frame_rate

# Print the time delay
print(f"Time Delay: {time_delay} seconds")
# Plot the original signals
plt.figure(figsize=(12, 4))
plt.subplot(2, 1, 1)
plt.plot(signal1, color='b')
plt.title('Audio Signal 1')
plt.xlabel('Sample')
plt.ylabel('Amplitude')
plt.grid()

plt.subplot(2, 1, 2)
plt.plot(signal2, color='g')
plt.title('Audio Signal 2')
plt.xlabel('Sample')
plt.ylabel('Amplitude')
plt.grid()
plt.tight_layout()

# Plot the GCC-PHAT cross-correlation
plt.figure(figsize=(12, 4))
plt.plot(time, gcc_phat_result, color='r')
plt.title('GCC-PHAT Cross-Correlation of Audio Signals')
plt.xlabel('Time (s)')
plt.ylabel('Correlation')
plt.grid()
plt.tight_layout()
plt.show()