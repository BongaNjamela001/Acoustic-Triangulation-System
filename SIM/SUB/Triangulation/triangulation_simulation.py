import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt1
from scipy.signal import correlate
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Initialize parameters
microphone_positions = np.array([[0.05, 0, 0], [0, 0.05, 0], [-0.05, 0, 0], [0, -0.05, 0]])
source_position = np.array([1, 1, 1])  # Default position (1m above origin)
temperature = 25  # Default temperature in Celsius

# Calculate speed of sound
speed_of_sound = 331.4 * np.sqrt(1 + (temperature / 273.15))
source_amplitude = 1.0

# Generate a sound signal (sine wave)
sample_rate = 44100  # Sample rate in Hz
duration = 0.5  # Duration in seconds
num_samples = int(duration * sample_rate)
frequency = 1000  # Frequency in Hz
t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)


# Generate the sound signal (sine wave)
sound_signal = source_amplitude * np.sin(2 * np.pi * frequency * t)

# Visualization
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(microphone_positions[:, 0], microphone_positions[:, 1], microphone_positions[:, 2], c='r', marker='o', label='Microphones')
ax.scatter(source_position[0], source_position[1], source_position[2], c='b', marker='*', label='Sound Source')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
#plt.show()
#plt.close()

# Sound propagation simulation
def simulate_sound_propagation(sound_signal, source_position, microphone_position, speed_of_sound, sample_rate):
    # Calculate the distance from the source to the microphone
    distance = np.linalg.norm(source_position - microphone_position)
    # Calculate the time it takes for sound to travel from the source to the microphone
    travel_time = distance / speed_of_sound
    # Calculate the corresponding sample delay
    sample_delay = int(travel_time * sample_rate)

    # Create a delayed version of the sound signal
    delayed_sound_signal = np.zeros_like(sound_signal)

    
    if sample_delay < len(sound_signal):
        delayed_sound_signal[sample_delay:] = sound_signal[:-sample_delay]
    
    return delayed_sound_signal

microphone_signals = []
for mic_position in microphone_positions:
    microphone_signal = simulate_sound_propagation(sound_signal, source_position, mic_position, speed_of_sound, sample_rate)
    microphone_signals.append(microphone_signal)

# Plot the sound signals received at each microphone
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for i, mic_signal in enumerate(microphone_signals):
    ax.plot(t, mic_signal, zs=mic_position[2], label=f'Microphone {i+1}')

ax.set_xlabel('Time (s)')
ax.set_ylabel('X-axis')
ax.set_zlabel('Amplitude')
ax.legend()

#plt.show()

# Apply microphone reception and signal processing

# Signal processing: Cross-correlation
def process_microphone_signals(microphone_signals):
    # Calculate the cross-correlation of each microphone's signal with the original signal
    correlations = []
    for mic_signal in microphone_signals:
        correlation = correlate(mic_signal, sound_signal, mode='full')
        correlations.append(correlation*1000)
    return correlations

# Process the microphone signals
correlations = process_microphone_signals(microphone_signals)

# Plot the cross-correlation results for each microphone
for i, correlation in enumerate(correlations):
    plt1.figure()
    plt1.plot(correlation)
    plt1.title(f'Cross-Correlation with Microphone {i+1}')
    plt1.xlabel('Lag (samples)')
    plt1.ylabel('Correlation')
    plt1.grid(True)

#plt1.show()

# Implement GCC-PHAT and angle estimation

# Simulated time delay estimation
def gcc_phat(signal_1, signal_2, sample_rate):
    # Compute the cross-correlation with phase transform (GCC-PHAT)
    cross_correlation = signal.correlate(signal_1, signal_2, mode='full', method='fft')
    # Compute the phase transform
    phase_transform = np.angle(np.fft.fftshift(np.fft.fft(cross_correlation)))

    # Calculate time delay estimation (in samples)
    max_index = np.argmax(phase_transform)

    time_delay_samples = abs(max_index - (len(signal_1) + 1))

    # Calculate time delay in seconds
    time_delay_seconds = time_delay_samples / sample_rate
    return time_delay_seconds

# Estimate the time delays for each microphone pair
time_delays = []

# Calculate time delays for each microphone pair
for i in range(len(microphone_positions) - 1):
    for j in range(i + 1, len(microphone_positions)):
        if (j!=i):
            microphone_1_signal = microphone_signals[i]
            microphone_2_signal = microphone_signals[j]

            # Calculate the time delay between microphone pairs
            time_delay = gcc_phat(microphone_1_signal, microphone_2_signal, sample_rate)
            # print("========GCC-Phat Time Delay========")
            # print(time_delay)
            # print("================")
            time_delays.append(time_delay)
#print(time_delays)
def calculate_angle_of_arrival(mic_pair_positions, time_delays, speed_of_sound):
    # Calculate the direction vector between the microphones in the pair
    direction_vector = mic_pair_positions[1] - mic_pair_positions[0]

    # Calculate the distance between the microphones in the pair
    distance = np.linalg.norm(direction_vector)
    
    # Calculate the time difference of arrival (TDoA) between the microphones
    tdoa = time_delays[1] - time_delays[0]
 #   print(tdoa)
 #   print(tdoa)
    angle_t = tdoa * speed_of_sound / distance
 #   print(angle_t)
    angle_t = np.clip(angle_t, -1, 1)
    # Calculate the angle of arrival using the TDoA and speed of sound
    angle_of_arrival = np.arcsin(angle_t)

    return angle_of_arrival


mic_pair1_positions = microphone_positions[[0, 2]]
mic_pair2_positions = microphone_positions[[1, 3]]

# Calculate the angles of arrival for the two pairs
angle_of_arrival1 = calculate_angle_of_arrival(mic_pair1_positions, time_delays[:2], speed_of_sound)
# print(angle_of_arrival1)
angle_of_arrival2 = calculate_angle_of_arrival(mic_pair2_positions, time_delays[2:], speed_of_sound)
# print(angle_of_arrival2)

# Triangulation based on angles of arrival
def triangulate_sound_source(angle1, angle2, mic_positions, speed_of_sound):
    # Calculate direction vectors from angles
    dir1 = np.array([np.cos(angle1), np.sin(angle1), 0])
    dir2 = np.array([np.cos(angle2), np.sin(angle2), 0])

    # Calculate the normal vector of the plane defined by the microphone pairs
    normal_vector = np.cross(dir1, dir2)

    # Calculate distances between microphone pairs
    dist1 = speed_of_sound * time_delays[0]
    dist2 = speed_of_sound * time_delays[2]

    if np.linalg.norm(normal_vector) != 0.0:
        # Calculate the position of the sound source
        source_position = mic_positions[0] + (dist1 / np.linalg.norm(normal_vector)) * dir1

    return source_position

# Perform triangulation
estimated_source_position = triangulate_sound_source(angle_of_arrival1, angle_of_arrival2, mic_pair1_positions, speed_of_sound)

# Output the estimated sound source location
print(f"Estimated Sound Source Location: X = {estimated_source_position[0]:.3f} meters, "
      f"Y = {estimated_source_position[1]:.3f} meters, "
      f"Z = {estimated_source_position[2]:.3f} meters")
