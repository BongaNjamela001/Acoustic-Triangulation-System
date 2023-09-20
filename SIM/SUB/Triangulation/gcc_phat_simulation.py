import numpy as np
import matplotlib.pyplot as plt

# Define parameters
sampling_rate = 44100  # Sampling rate in Hz
duration = 1.0  # Duration of the signal in seconds
num_samples = int(sampling_rate * duration)
time = np.linspace(0, duration, num_samples, endpoint=False)  # Time array

# Create a reference signal (e.g., a sine wave)
reference_frequency = 1000.0  # Frequency in Hz
reference_signal = np.sin(2 * np.pi * reference_frequency * time)

# Simulate propagation to two microphones with time delays
microphone_positions = [0.05, 0.0], [-0.05, 0.0]  # Microphone positions (x-coordinates)
microphone_delays = [0.0, 0.01]  # Delays in seconds

microphone_signals = []
for delay in microphone_delays:
    delayed_signal = np.roll(reference_signal, int(delay * sampling_rate))
    microphone_signals.append(delayed_signal[:num_samples])

# Compute the GCC-PHAT cross-correlation
def gcc_phat(signal1, signal2):
    fft_signal1 = np.fft.fft(signal1)
    fft_signal2 = np.fft.fft(signal2)
    cross_correlation = fft_signal1 * np.conj(fft_signal2)
    cross_correlation /= np.abs(cross_correlation)
    inverse_correlation = np.fft.ifft(cross_correlation)
    gcc_phat_result = np.abs(inverse_correlation)
    return gcc_phat_result

gcc_phat_result = gcc_phat(microphone_signals[0], microphone_signals[1])

# Find the time delay corresponding to the peak
peak_index = np.argmax(gcc_phat_result)
estimated_delay = time[peak_index]

# Plot the GCC-PHAT result
plt.figure(figsize=(8, 4))
plt.plot(time, gcc_phat_result)
plt.xlabel("Time Delay (s)")
plt.ylabel("GCC-PHAT Value")
plt.title("GCC-PHAT Cross-Correlation")
plt.grid(True)
plt.show()

print("Estimated Time Delay:", estimated_delay, "seconds")