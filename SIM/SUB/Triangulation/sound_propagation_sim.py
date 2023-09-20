import numpy as np
import matplotlib.pyplot as plt

# Define parameters
sampling_rate = 44100  # Sampling rate in Hz
duration = 0.05  # Duration of the signal in seconds
num_samples = int(sampling_rate * duration)
time = np.linspace(0, duration, num_samples, endpoint=False)  # Time array

# Define frequencies and their amplitudes
frequencies = [440.0, 880.0, 1320.0, 1760.0]  # Frequencies in Hz
amplitudes = [0.8, 0.6, 0.4, 0.2]  # Corresponding amplitudes

# Generate the mixed signal
mixed_signal = np.zeros(num_samples)
for freq, amp in zip(frequencies, amplitudes):
    mixed_signal += amp * np.sin(2 * np.pi * freq * time)

# Simulate propagation to the four microphones (add delays and attenuations)
microphone_positions = [0.05, 0.05, 0], [-0.05, 0.05, 0], [-0.05, -0.05, 0], [0.05, -0.05, 0]
microphone_delays = [0.0, 0.01, 0.02, 0.03]  # Delays in seconds
microphone_attenuations = [1.0, 0.8, 0.6, 0.4]  # Attenuations

microphone_signals = []
for delay, attenuation in zip(microphone_delays, microphone_attenuations):
    delayed_signal = np.roll(mixed_signal, int(delay * sampling_rate))
    microphone_signal = attenuation * delayed_signal[:num_samples]
    microphone_signals.append(microphone_signal)

# Plot the signals at the microphones
plt.figure(figsize=(12, 6))
for i, mic_signal in enumerate(microphone_signals):
    plt.subplot(2, 2, i + 1)
    plt.plot(time, mic_signal)
    plt.title(f"Microphone {i + 1}")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)

plt.tight_layout()
plt.show()



