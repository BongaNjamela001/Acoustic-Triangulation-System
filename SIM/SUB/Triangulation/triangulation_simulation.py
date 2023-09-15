import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt1
from scipy.signal import correlate
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Initialize parameters
microphone_positions = np.array([[0.05, 0.05, 0], [-0.05, 0.05, 0], [-0.05, -0.05, 0], [0.05, -0.05, 0]])
source_position = np.array([0, 0, 1])  # Default position (1m above origin)
temperature = 25  # Default temperature in Celsius

# Calculate speed of sound
speed_of_sound = 331.4 * np.sqrt(1 + (temperature / 273.15))
speed_of_sound = int(speed_of_sound)
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

plt.show()

# Apply microphone reception and signal processing

# Signal processing: Cross-correlation
def process_microphone_signals(microphone_signals):
    # Calculate the cross-correlation of each microphone's signal with the original signal
    correlations = []
    for mic_signal in microphone_signals:
        correlation = correlate(mic_signal, sound_signal, mode='full')
        correlations.append(correlation)
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

plt1.show()

# Implement GCC-PHAT and angle estimation

# Simulated time delay estimation
def gcc_phat(signal_1, signal_2, sample_rate):
    # Compute the cross-correlation with phase transform (GCC-PHAT)
    cross_correlation = signal.correlate(signal_1, signal_2, mode='full', method='fft')

    # Compute the phase transform
    phase_transform = np.angle(np.fft.fftshift(np.fft.fft(cross_correlation)))

    # Calculate time delay estimation (in samples)
    max_index = np.argmax(phase_transform)
    time_delay_samples = max_index - len(signal_1) + 1

    # Calculate time delay in seconds
    time_delay_seconds = time_delay_samples / sample_rate

    return time_delay_seconds

# Estimate the time delays for each microphone pair
sample_rate = 44100  # Adjust this to match your actual sample rate
time_delays = []

# Calculate time delays for each microphone pair
for i in range(len(microphone_positions) - 1):
    for j in range(i + 1, len(microphone_positions)):
        if (j!=i):
            microphone_1_signal = microphone_signals[i]
            microphone_2_signal = microphone_signals[j]

            # Calculate the time delay between microphone pairs
            time_delay = gcc_phat(microphone_1_signal, microphone_2_signal, sample_rate)
            time_delays.append((i, j, time_delay))

# Calculate angles of arrival based on time delays
angles_of_arrival = []

for i in range(len(microphone_positions) - 1):
    for j in range(i + 1, len(microphone_positions)):
        # Calculate the direction vector from microphone i to microphone j
            direction_vector = microphone_positions[j] - microphone_positions[i]

            # Calculate the distance between microphone pairs
            distance = np.linalg.norm(direction_vector)

            if distance != 0:
                # Calculate the angle of arrival using the known speed of sound and time delay
                angle_of_arrival = np.arcsin(time_delays[i] * speed_of_sound / distance)
                angles_of_arrival.append(angle_of_arrival)

# Convert angles from radians to degrees
angles_of_arrival_degrees = [np.degrees(angle) for angle in angles_of_arrival]

# Print the estimated angles of arrival
for i, angle_deg in enumerate(angles_of_arrival_degrees):
    print(f"Microphone Pair {i + 1}: Angle of Arrival = {angle_deg} degrees")

# Implement triangulation

# Known time delays (replace with your actual measurements)
time_delays = np.array([0.001, 0.002, 0.003])  # Time delays in seconds

# Function to perform sound source triangulation
def triangulate_sound_source(microphone_positions, angles_of_arrival, speed_of_sound, time_delays):
    # Initialize variables for triangulation
    x_sum = 0.0
    y_sum = 0.0
    z_sum = 0.0

    # Calculate triangulated position
    for i in range(len(angles_of_arrival)):
        for j in range(i + 1, len(angles_of_arrival)):
            # Calculate the direction vectors
            direction_vector_i = np.array([np.cos(angles_of_arrival[i]), np.sin(angles_of_arrival[i]), 0])
            direction_vector_j = np.array([np.cos(angles_of_arrival[j]), np.sin(angles_of_arrival[j]), 0])

            # Calculate the normal vector of the plane defined by the microphones
            normal_vector = np.cross(direction_vector_i, direction_vector_j)

            # Calculate the time differences and distance between microphone pairs
            time_diff = time_delays[j] - time_delays[i]
            if np.linalg.norm(normal_vector) != 0.0:

                distance = speed_of_sound * time_diff

                # Calculate the position components in 3D space
                position_component = distance * normal_vector / np.linalg.norm(normal_vector)

                # Add the position components to the sums
                x_sum += position_component[0]
                y_sum += position_component[1]
                z_sum += position_component[2]

    # Calculate the average positions
    avg_x = x_sum / (len(angles_of_arrival) * (len(angles_of_arrival) - 1) / 2)
    avg_y = y_sum / (len(angles_of_arrival) * (len(angles_of_arrival) - 1) / 2)
    avg_z = z_sum / (len(angles_of_arrival) * (len(angles_of_arrival) - 1) / 2)

    return np.array([avg_x, avg_y, avg_z])

# Perform sound source triangulation
estimated_source_position = triangulate_sound_source(
    microphone_positions, angles_of_arrival, speed_of_sound, time_delays
)

# Output the estimated sound source location
print(f"Estimated Sound Source Location: X = {estimated_source_position[0]:.3f} meters, "
      f"Y = {estimated_source_position[1]:.3f} meters, "
      f"Z = {estimated_source_position[2]:.3f} meters")