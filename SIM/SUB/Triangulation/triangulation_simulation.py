import numpy as np
from scipy.signal import hilbert
import scipy.signal as signal
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt1
from scipy.signal import correlate
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Initialize parameters
microphone_positions = np.array([[0, 0, 0], [0.1, 0, 0], [0.1, 0.1, 0], [0, 0.1, 0]])
source_position = np.array([1, 1, 1])  # Default position (1m above origin)
temperature = 25  # Default temperature in Celsius

# Calculate speed of sound
speed_of_sound = 331.4 * np.sqrt(1 + (temperature / 273.15))
source_amplitude = 1.0

# Generate a sound signal (sine wave)
sample_rate = 44100  # Sample rate in Hz
duration = 3  # Duration in seconds
num_samples = int(duration * sample_rate)
frequency = 440  # Frequency in Hz
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
plt.show()
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

#     # Calculate the cross-correlation of each microphone's signal with the original signal
#     correlations = []
#     for mic_signal in microphone_signals:
#         correlation = correlate(mic_signal, sound_signal, mode='full')
#         correlations.append(correlation)
#     return correlations

# # Process the microphone signals
# correlations = process_microphone_signals(microphone_signals)
# print(correlations)

# # Plot the cross-correlation results for each microphone
# for i, correlation in enumerate(correlations):
#     plt1.figure()
#     plt1.plot(correlation)
#     plt1.title(f'Cross-Correlation with Microphone {i+1}')
#     plt1.xlabel('Lag (samples)')
#     plt1.ylabel('Correlation')
#     plt1.grid(True)

#plt1.show()

# Implement GCC-PHAT and angle estimation

# Simulated time delay estimation
# def gcc_phat(signal_1, signal_2, sample_rate):
#     # Compute the cross-correlation with phase transform (GCC-PHAT)
#     cross_correlation = signal.correlate(signal_1, signal_2, mode='full', method='fft')
   
#     # Compute the phase transform
#     phase_transform = np.angle(np.fft.fftshift(np.fft.fft(cross_correlation)))

#     # Calculate time delay estimation (in samples)
#     max_index = np.argmax(phase_transform)

#     time_delay_samples = abs(max_index - (len(signal_1) + 1))

#     # Calculate time delay in seconds
#     time_delay_seconds = time_delay_samples / sample_rate
#     return time_delay_seconds

# # Estimate the time delays for each microphone pair
time_delays = []


def gcc_phat(signal1, signal2):
    # Compute the cross-correlation of the two signals
    cross_correlation = correlate(signal1, signal2, mode='full')
    
    # Calculate the phase of the cross-correlation using the Hilbert transform
    hilbert_transformed = hilbert(cross_correlation)
    phase = np.angle(hilbert_transformed)
    
    # Compute the GCC-PHAT
    gcc_phat_result = np.exp(1j * phase)
    
    # Find the time delay that maximizes the GCC-PHAT
    optimal_time_delay = np.argmax(np.abs(gcc_phat_result))
    
    return optimal_time_delay / sample_rate

# def process_microphone_signals(microphone_signals):
#     # Split the microphone signals into pairs (m1-m2 and m3-m4)
#     m1_m2_signals = microphone_signals[0:2]
#     m3_m4_signals = microphone_signals[2:4]

#     # Calculate the optimal time delay for each pair using GCC-PHAT
#     optimal_time_delay_m1_m2 = abs(gcc_phat(m1_m2_signals[0], m1_m2_signals[1]))
#     optimal_time_delay_m3_m4 = abs(gcc_phat(m3_m4_signals[0], m3_m4_signals[1]))

#     return optimal_time_delay_m1_m2, optimal_time_delay_m3_m4

def process_microphone_signals(microphone_signals):
    # Split the microphone signals into pairs (m1-m2 and m3-m4)
    m1_m2_signals = microphone_signals[0:2]
    m3_m4_signals = microphone_signals[2:4]
    m1_m4_signals = [microphone_signals[0], microphone_signals[3]]
    m2_m3_signals = [microphone_signals[1], microphone_signals[2]]

    # Calculate the optimal time delay for each pair using GCC-PHAT
    optimal_time_delay_m1_m2 = gcc_phat(m1_m2_signals[0], m1_m2_signals[1])
    # print(optimal_time_delay_m1_m2)
    optimal_time_delay_m3_m4 = gcc_phat(m3_m4_signals[0], m3_m4_signals[1])
    # print(optimal_time_delay_m3_m4)
    optimal_time_delay_m1_m4 = gcc_phat(m1_m4_signals[0], m1_m4_signals[1])
    # print(optimal_time_delay_m1_m4)
    optimal_time_delay_m2_m3 = gcc_phat(m2_m3_signals[0], m2_m3_signals[1])
    # print(optimal_time_delay_m2_m3)
    # Put the optimal time delays in an array
    optimum_time_delays = np.array([optimal_time_delay_m1_m2, optimal_time_delay_m3_m4, optimal_time_delay_m1_m4, optimal_time_delay_m2_m3])

    return optimum_time_delays

# Calculate distances from the source to each microphone
distances = np.linalg.norm(microphone_positions - source_position, axis=1)

# Calculate TOA for each microphone
TOA = distances / speed_of_sound


# Output the calculated TOAs
print("Time of Arrival (TOA) at Each Microphone:")
for i, toa in enumerate(TOA):
    print(f"Microphone {i+1}: {toa:.6f} seconds")

# Calculate time delays for each microphone pair
# for i in range(len(microphone_positions) - 1):
#     for j in range(i + 1, len(microphone_positions)):
#         microphone_1_signal = microphone_signals[i]
#         microphone_2_signal = microphone_signals[j]

#         # Calculate the time delay between microphone pairs
#         time_delay = gcc_phat(microphone_1_signal, microphone_2_signal, sample_rate)

#         time_delays.append(time_delay)

# #print(time_delays)

# # ANGLE OF ARRIVAL
# angles_of_arrival = []
# for mic1, mic2, time_delay in time_delays:
#     # Calculate the distance between microphone pairs
#     distance = np.linalg.norm(microphone_positions[mic1] - microphone_positions[mic2])
    
#     # Calculate the angle of arrival using the speed of sound and time delay
#     angle_of_arrival = np.arctan2(distance, speed_of_sound * time_delay)
    
#     angles_of_arrival.append(angle_of_arrival)

# # Convert angles from radians to degrees
# angles_of_arrival_degrees = [np.degrees(angle) for angle in angles_of_arrival]

# # Print the estimated angles of arrival
# for i, angle_deg in enumerate(angles_of_arrival_degrees):
#     print(f"Microphone Pair {i + 1}: Angle of Arrival = {angle_deg} degrees")


def triangulate_sound_source(optimum_time_delays, microphone_positions, speed_of_sound):
    # Calculate the distances between microphone pairs
    distance_between_m1_m2 = np.linalg.norm(microphone_positions[1] - microphone_positions[0])
    distance_between_m3_m4 = np.linalg.norm(microphone_positions[3] - microphone_positions[2])
    distance_between_m1_m4 = np.linalg.norm(microphone_positions[3] - microphone_positions[0])

    distance_to_source_1 = optimum_time_delays[0] * speed_of_sound
    
    distance_to_source_2 = optimum_time_delays[1] * speed_of_sound
    

    #Calculate distance using time of arrival
    dist_m1 = TOA[0] * speed_of_sound
    dist_m2 = TOA[1] * speed_of_sound
    dist_m3 = TOA[2] * speed_of_sound
    dist_m4 = TOA[3] * speed_of_sound

    #Calculate angle between x-axis and path followed to microphone m1
    arrival_angle_m1_m2 = np.arccos((distance_between_m1_m2**2 + dist_m1**2-dist_m2**2)/(2*dist_m1*distance_between_m1_m2))
    arrival_angle_m2_m1 = np.arccos((distance_between_m1_m2**2 + dist_m2**2 - dist_m1**2)/(2*distance_between_m1_m2*dist_m2))
    arrival_angle_m1_m4 = np.arccos((distance_between_m1_m4**2 + dist_m1**2 - dist_m4**2)/(2*dist_m1*distance_between_m1_m4))
    # arrival_angle_m2 = np.arccos(()/())


    sound_source_z = dist_m2*np.sin(arrival_angle_m2_m1)
    sound_source_x = np.sqrt(dist_m1**2 - sound_source_z**2)
    sound_source_y = 0

    
    if arrival_angle_m1_m2 > np.pi/2:
        if dist_m1 < dist_m2:
            sound_source_x = -sound_source_x
            # arrival_angle_m1_m2 
    
    if arrival_angle_m1_m2 > np.pi:
        sound_source_z = -sound_source_z

    if arrival_angle_m1_m4 < np.pi/2:
        sound_source_y = sound_source_x/np.tan(arrival_angle_m1_m4)
    # if dist_m4 > dist_m1 :
    #     sound_source_y = -sound_source_y
    # else if arrival
    # if (abs(sound_source_x) < distance_between_m1_m2):
    #     sound_y = 
    # else:

    # print(np.degrees(arrival_angle_m1))

    # Calculate the angles of arrival for m1-m2 and m3-m4 pairs
    # angle_arg_m1_m2 = distance_between_m1_m2/((optimum_time_delays[0] * speed_of_sound ))
    # angles_m1_m2 = np.arcsin(angle_arg_m1_m2)
    # # print(np.degrees(angles_m1_m2))
    # angle_arg_m3_m4 = distance_between_m3_m4/(optimum_time_delays[1] * speed_of_sound)
    # angles_m3_m4 = np.arcsin(angle_arg_m3_m4)
    # print(np.degrees(angles_m3_m4))

    # Triangulate the sound source position
    sound_source_position = np.array([sound_source_x, sound_source_y, sound_source_z])

    return sound_source_position

# def triangulate_sound_source(optimum_time_delays, microphone_positions, speed_of_sound):
     # Calculate the distances between microphone pairs
    #  direction_vector_m1_m2 = microphone_positions[1] - microphone_positions[0]
    #  direction_vector_m3_m4 = microphone_positions[3] - microphone_positions[2]
    
     # print(direction_vector_m1_m2)
     # print(direction_vector_m3_m4)
    

    #  distance_between_m1_m2 = np.linalg.norm(microphone_positions[1] - microphone_positions[0])
    #  distance_between_m3_m4 = np.linalg.norm(microphone_positions[3] - microphone_positions[2])
    
    #  midpoint_m1_m2 = (microphone_positions[1] + microphone_positions[0]) / 2
    #  midpoint_m3_m4 = (microphone_positions[3] + microphone_positions[2]) / 2

    #  distance_between_arrays = np.linalg.norm(midpoint_m3_m4 - midpoint_m1_m2)
     # print(midpoint_m1_m2)
     # print(midpoint_m3_m4)
    #  mid_to_mid = midpoint_m3_m4 - midpoint_m1_m2
    
     # print(mid_to_mid)
     # vec = np.cross(midpoint_m1_m2,midpoint_m3_m4)
    
     # Calculate the angles of arrival for m1-m2 and m3-m4 pairs
    #  angles_m1_m2 = np.arcsin(optimum_time_delays[0] * speed_of_sound / 2*distance_between_m1_m2)
    #  print(angles_m1_m2)
#     angles_m3_m4 = np.arcsin(optimum_time_delays[1] * speed_of_sound / 2*distance_between_m3_m4)
#     print(angles_m3_m4)

#     # Triangulate the sound source position in 3D
#     # sound_source_x = distance_between_m1_m2 * np.tan(angles_m1_m2)
#     sound_source_y = distance_between_m3_m4 /(np.tan(angles_m3_m4) + np.tan(angles_m1_m2))
#     sound_source_x = sound_source_y*np.tan(angles_m1_m2)
#     sound_source_z = distance_between_m1_m2*np.tan(angles_m1_m2)  # Assuming source is at (0, 0, 1)

#     sound_source_position = np.array([sound_source_x, sound_source_y, sound_source_z])

#     return sound_source_position
# #     num_microphones = len(microphone_positions)
# #     A = []
# #     b = []

# #     # Create a system of equations for each pair of microphones (i, j)
# #     for i in range(num_microphones):
# #         for j in range(i + 1, num_microphones):
# #             x_ij = microphone_positions[j] - microphone_positions[i]
# #             c = speed_of_sound
# #             equation = u.dot(x_ij) - c * optimum_time_delays[i]  # u dot x_ij = c * optimum_time_delay_ij
# #             A.append(x_ij)
# #             b.append(equation)

# #     # Convert A and b to NumPy arrays
# #     A = np.array(A)
# #     b = np.array(b)

# #     # Solve the system of linear equations
# #     solution = np.linalg.lstsq(A, b, rcond=None)[0]

# #     # The solution contains the x, y, and z coordinates of the sound source
# #     return solution

# Process microphone signals to get optimal time delays
optimum_time_delays = process_microphone_signals(microphone_signals)
print(optimum_time_delays)
# Triangulate the sound source position
estimated_sound_source_position = triangulate_sound_source(optimum_time_delays, microphone_positions, speed_of_sound)

# # Output the estimated sound source location
print(f"Estimated Sound Source Location: X = {estimated_sound_source_position[0]:.3f} meters, "
      f"Y = {estimated_sound_source_position[1]:.3f} meters, "
      f"Z = {estimated_sound_source_position[2]:.3f} meters")
# # def calculate_angle_of_arrival(mic_pair_positions, time_delays, speed_of_sound):
# #     # Calculate the direction vector between the microphones in the pair
# #     direction_vector = mic_pair_positions[1] - mic_pair_positions[0]

# #     # Calculate the distance between the microphones in the pair
# #     distance = np.linalg.norm(direction_vector)
    
# #     # Calculate the time difference of arrival (TDoA) between the microphones
# #     tdoa = time_delays[1] - time_delays[0]

# #     angle_t = tdoa * speed_of_sound / distance
# #     angle_t = np.clip(angle_t, -1, 1)
# #     # Calculate the angle of arrival using the TDoA and speed of sound
# #     angle_of_arrival = np.arcsin(angle_t)

# #     return angle_of_arrival


# # mic_pair1_positions = microphone_positions[[0, 2]]
# # mic_pair2_positions = microphone_positions[[1, 3]]

# # # Calculate the angles of arrival for the two pairs
# # angle_of_arrival1 = calculate_angle_of_arrival(mic_pair1_positions, time_delays[:2], speed_of_sound)
# # # print(angle_of_arrival1)
# # angle_of_arrival2 = calculate_angle_of_arrival(mic_pair2_positions, time_delays[2:], speed_of_sound)
# # # print(angle_of_arrival2)

# # # Triangulation based on angles of arrival
# # def triangulate_sound_source(angle1, angle2, mic_positions, speed_of_sound):
# #     # Calculate direction vectors from angles
# #     dir1 = np.array([np.cos(angle1), np.sin(angle1), 0])
# #     print(dir1)
# #     dir2 = np.array([np.cos(angle2), np.sin(angle2), 0])
# #     print(dir2)
# #     # Calculate the normal vector of the plane defined by the microphone pairs
# #     normal_vector = np.cross(dir1, dir2)

# #     # Calculate distances between microphone pairs
# #     dist1 = speed_of_sound * time_delays[0]
# #     dist2 = speed_of_sound * time_delays[2]

# #     if np.linalg.norm(normal_vector) != 0.0:
# #         # Calculate the position of the sound source
# #         source_position = mic_positions[0] + (dist1 / np.linalg.norm(normal_vector)) * dir1

# #     return source_position

# # Perform triangulation
# # estimated_source_position = triangulate_sound_source(angle_of_arrival1, angle_of_arrival2, mic_pair1_positions, speed_of_sound)

# # # Output the estimated sound source location
# # print(f"Estimated Sound Source Location: X = {estimated_source_position[0]:.3f} meters, "
# #       f"Y = {estimated_source_position[1]:.3f} meters, "
# #       f"Z = {estimated_source_position[2]:.3f} meters")

# Calculate possible source locations (intersection points of spheres)
# source_locations = []

# Initialize the coefficient matrix A and the right-hand side vector b
# num_microphones = len(microphone_positions)
# A = np.zeros((num_microphones - 1, 3))
# b = np.zeros(num_microphones - 1)

# # Create a system of equations based on pairwise differences
# for i in range(num_microphones - 1):
#     A[i, :] = 2 * (microphone_positions[i + 1] - microphone_positions[0])
#     b[i] = distances[0]**2 - distances[i + 1]**2 + np.dot(microphone_positions[i + 1], microphone_positions[i + 1]) - np.dot(microphone_positions[0], microphone_positions[0])
# # Solve the system of equations
# intersection_x = np.linalg.lstsq(A, b, rcond=None)[0]

# # Output the estimated 3D intersection point
# print("Estimated 3D Intersection Point:")
# print(f"X = {intersection_x[0]:.3f} meters")
# print(f"Y = {intersection_x[1]:.3f} meters")
# print(f"Z = {intersection_x[2]:.3f} meters")
# for i in range(len(microphone_positions)):
#     for j in range(i + 1, len(microphone_positions)):
#         for k in range(j + 1, len(microphone_positions)):
#             d1 = distances[i]
#             d2 = distances[j]
#             d3 = distances[k]
            
#             p1 = microphone_positions[i]
#             p2 = microphone_positions[j]
#             p3 = microphone_positions[k]
            
#             # Calculate intersection point between spheres
#             # This involves solving a system of equations
            
#             # Example: Solve for x-coordinate
#             A = 2 * np.array([p2 - p1, p3 - p1]).T
#             print(A)
#             b = np.array([d1**2 - d2**2 + np.dot(p2, p2) - np.dot(p1, p1),
#                           d1**2 - d3**2 + np.dot(p3, p3) - np.dot(p1, p1)])
            
#             print(b)
#             intersection_x = np.linalg.solve(A, b)
            
#             # Calculate y and z coordinates similarly
            
#             source_locations.append(intersection_x)

# # Output the estimated 3D source locations
# print("Estimated 3D Sound Source Locations:")
# for location in source_locations:
#     print(f"X = {location[0]:.3f} meters, Y = {location[1]:.3f} meters, Z = {location[2]:.3f} meters")