import numpy as np
import scipy.signal as signal

# Constants
SPEED_OF_SOUND = 350  # Speed of sound in m/s (room temperature)

# Function to calculate time differences of arrival (TDoA)
def calculate_time_delay(audio_signal1, audio_signal2, sampling_rate):
    # Calculate cross-correlation using FFT
    cross_correlation = signal.correlate(audio_signal1, audio_signal2, mode='full', method='fft')

    # Find the time shift (lag) with the highest cross-correlation value
    time_shift = np.argmax(cross_correlation) - len(audio_signal1) + 1

    # Calculate time delay in seconds
    time_delay = time_shift / sampling_rate

    return time_delay

# Function to interface with the triangulation subsystem
def interface_with_triangulation(time_delay_measurement):
    # Replace this with your code to send time_delay_measurement to the triangulation subsystem
    # Example: Use inter-process communication or other suitable methods

# Main function
if __name__ == "__main__":
    # Simulated audio data for two microphones (replace with your actual audio data)
    audio_signal1 = np.array([...])
    audio_signal2 = np.array([...])

    # Sampling rate (replace with your actual sampling rate)
    sampling_rate = 44100  # Example: 44.1 kHz

    # Calculate TDoA between audio signals
    time_delay = calculate_time_delay(audio_signal1, audio_signal2, sampling_rate)

    # Interface with the triangulation subsystem
    interface_with_triangulation(time_delay)

    # Print the calculated TDoA (optional)
    print("Time Delay: {:.6f} seconds".format(time_delay))
