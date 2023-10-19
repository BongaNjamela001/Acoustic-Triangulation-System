import numpy as np
from scipy.optimize import minimize
import sys
from scipy.signal import hilbert
import scipy.signal as signal
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt1
from scipy.signal import correlate
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation


# Heading

print("===================Triangulation Simulation=========================")
print("This program simulates an acoustic triangulation system of four")
print("microphones m1, m2, m3, and m4, in 3D. A sound source, located at")
print("[sound_source_x, sound_source_y, sound_source_z], emits a signal")
print("which is recorded by the microphones at the time of arrival. The")
print("The program uses the time difference of arrival between microphones")
print("to triangulate the location of the source source with m1 as the")
print("reference microphone.")
print("====================================================================\n")

# Set microphone m1 as default microphone
m1 = [0, 0, 0]
m2 = []
m3 = []
m4 = []
microphone_positions = np.array([[], [], [], []])
temperature = 25  # Default temperature in Celsius
speed_of_sound = 331.4 * np.sqrt(1 + (temperature / 273.15))
source_coord = [1,1,1]
source_position = np.array(source_coord)  # Default position (1m above origin)
default = ""

while default == "":
    default = input("Use default settings? y/n:\n")
    
    if default == "y":
        m2 = [0.1, 0, 0]
        m3 = [0.1, 0.1, 0]
        m4 = [0, 0.1, 0]
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
        print()
        print("=====================Default Source Position========================")
        print("Sound source emits signal from coordinate", source_coord, " with")
        print("respect to microphone m1.")
        print("====================================================================")
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
                    m2 = [mic_dist, 0, 0]
                    m3 = [mic_dist, mic_dist, 0]
                    m4 = [0, mic_dist, 0]
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
        
        print("======================User Source Position==========================")
        sound_source_x_str = ""
        sound_source_y_str = ""
        sound_source_z_str = ""
        while sound_source_x_str == "" or sound_source_y_str == "" or sound_source_z_str == "":
            sound_source_x_str, sound_source_y_str, sound_source_z_str = [float(i) for i in input("Enter a comma-separated sound source 3D coordinate .\n").split(", ")]
            if sound_source_x_str != "" and sound_source_y_str != "" and sound_source_z_str != "":
                u_source_x = float(sound_source_x_str)
                u_source_y = float(sound_source_y_str)
                u_source_z = float(sound_source_z_str)
                source_coord = [u_source_x,u_source_y,u_source_z]
                source_position = np.array(source_coord)
        print("Source coordinate ({},{},{}) in meters.".format(u_source_x, u_source_y, u_source_z))
        print("====================================================================\n")
        
    else:
        default = ""

# Initialize parameters

# Calculate speed of sound
# source_amplitude = 1.0

# Generate a sound signal (sine wave)
sample_rate = 44100  # Sample rate in Hz
duration = 0.2 # Duration in seconds
num_samples = int(duration * sample_rate)
frequencies = [440.0, 880.0, 1320.0, 1760.0]  # Frequencies in Hz
t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

amplitudes = [0.8, 0.6, 0.4, 0.2]  # Corresponding amplitudes

# Generate the mixed signal
mixed_signal = np.zeros(num_samples)
for freq, amp in zip(frequencies, amplitudes):
    mixed_signal += amp * np.sin(2 * np.pi * freq * t)
# Generate the sound signal (sine wave)
sound_signal = mixed_signal

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

# Plot the signals at the microphones
plt.figure(figsize=(12, 6))
for i, mic_signal in enumerate(microphone_signals):
    plt.subplot(2, 2, i + 1)
    plt.plot(t, mic_signal)
    plt.title(f"Microphone {i + 1}")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)

plt.tight_layout()
plt.show()

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

# Signal processing: Cross-correlation
def correlate_microphone_signals(microphone_signals):
    # Calculate the cross-correlation of each microphone's signal with the original signal
    correlations = []
    for mic_signal in microphone_signals:
        correlation = correlate(mic_signal, sound_signal, mode='full')
        correlations.append(correlation)
    return correlations

# Process the microphone signals
correlations = correlate_microphone_signals(microphone_signals)

# Plot the cross-correlation results for each microphone
for i, correlation in enumerate(correlations):
    plt.figure()
    plt.plot(correlation)
    plt.title(f'Cross-Correlation with Microphone {i+1}')
    plt.xlabel('Lag (samples)')
    plt.ylabel('Correlation')
    plt.grid(True)

plt.show()

# time_delays = []

# def gcc_phat(signal_1, signal_2, sample_rate):
#     # Compute the cross-correlation with phase transform (GCC-PHAT)
#     cross_correlation = signal.correlate(signal_1, signal_2, mode='full', method='fft')
    
#     # Compute the phase transform
#     phase_transform = np.angle(np.fft.fftshift(np.fft.fft(cross_correlation)))
    
#     # Calculate time delay estimation (in samples)
#     max_index = np.argmax(phase_transform)
#     time_delay_samples = max_index - len(signal_1) + 1
    
#     # Calculate time delay in seconds
#     time_delay_seconds = time_delay_samples / sample_rate
    
#     return time_delay_seconds

# Calculate time delays for each microphone pair
# for i in range(len(microphone_positions) - 1):
#     for j in range(i + 1, len(microphone_positions)):
#         microphone_1_signal = microphone_signals[i]  
#         microphone_2_signal = microphone_signals[j] 
        
#         # Calculate the time delay between microphone pairs
#         time_delay = gcc_phat(microphone_1_signal, microphone_2_signal, sample_rate)
#         print("Time delay:",i, j, time_delay, "s")
#         time_delays.append((i, j, time_delay))

# Calculate distances from the source to each microphone
distances = np.linalg.norm(microphone_positions - source_position, axis=1)

# Simulate TOA for each microphone
TOA = distances / speed_of_sound
print(TOA)

def tdoa(toas, num_microphones):
    TDoAs = []
    for i in range(num_microphones - 1):
        for j in range(i + 1, num_microphones):
            tdoa = toas[j] - toas[i]
            print("Time difference of arrival for mic ",i,"and",j,":",tdoa, "s")
            TDoAs.append((i,j,tdoa))
    return TDoAs

tdoaMeasurements = tdoa(TOA, num_microphones=len(microphone_positions))
# Calculate angles of arrival based on time delays
pair_distances = []
dcos_aoas = []

# Calculate angles of arrival based on time delays
angles_of_arrival = []

for mic1, mic2, tdoas in tdoaMeasurements:
    # Calculate the distance between microphone pairs
    distance = np.linalg.norm(microphone_positions[mic1] - microphone_positions[mic2])
    
    # Calculate the angle of arrival using the speed of sound and time delay
    aoa = np.arccos((speed_of_sound*tdoas)/distance)
    
    angles_of_arrival.append(aoa)

# Convert angles from radians to degrees
angles_of_arrival_degrees = [np.degrees(angle) for angle in angles_of_arrival]

# Print the estimated angles of arrival
for i, angle_deg in enumerate(angles_of_arrival_degrees):
    print(f"Microphone Pair {i + 1}: Angle of Arrival = {angle_deg} degrees")

# def tdoa_aoa(tdoas, mic_positions):
#     # Calculate distance dcos(AoA) corresponding to speed_of_sound*TDoA
#     for i in range(len(tdoas)):
#         dcos_aoa = np.abs(tdoas[i])*speed_of_sound
#         dcos_aoas.append(dcos_aoa)
    
#     # Calculate distance between microphone pairs
#     for i in range(len(mic_positions)):
#         for j in range(i+1, len(mic_positions)):
#             distance = np.linalg.norm(mic_positions[j] - mic_positions[i])
#             pair_distances.append(distance)

#     # Calculate angles of arrival
#     for i in range(len(dcos_aoas)):
#         aoa_longest = np.arccos(dcos_aoas[i]/pair_distances[i]) #Angle subtended by the longest distance to
#         # print(np.degrees(aoa_longest))
#         angles_of_arrival.append(aoa_longest)
#         # aoa_shortest = np.pi - np.pi/2 - aoa_longest
    
#     return angles_of_arrival

# print(tdoaMeasurements)
# aoa_longest_sides = tdoa_aoa(tdoaMeasurements, microphone_positions)

# print(np.degrees(aoa_longest_sides))
# Function to calculate the unit vector from an angle in xy-plane
# est_x = 0
# est_y = 0
# est_z = 0

# diag_removed = [0 for i in range(4)]
# entry = 0
# for i in range(len(dcos_aoas)):
#     if i != 1 or i != 4:
#         entry = dcos_aoas[i]
#         diag_removed.append(entry)

# print(diag_removed)

# short_dist = np.array(diag_removed)
# # Triangulate the 3D source location
# est_source_location = np.linalg.lstsq(microphone_positions, short_dist ** 2, rcond=None)[0]
# source_location = np.sqrt(est_source_location)

# # Output the estimated 3D source location
# print("Estimated 3D Sound Source Location:")
# print(f"X = {source_location[0]:.3f} meters")
# print(f"Y = {source_location[1]:.3f} meters")
# print(f"Z = {source_location[2]:.3f} meters")
# def check_x_y_quadrant(aoas, tdoas):
#     xy_quad = 0

#     for i in range(len(aoas)):
#         shared_distance = 
#     if aoas[0] == np.pi/2 - aoas[2]:
#         xy_quad = 1
#     elif aoas[3] == np.pi/2 - aoas[0]:
#         xy_quad = 2
#     elif aoas[5] == np.pi/2 - aoas[3]:
#         xy_quad = 3
#     elif aoas[2] == np.pi/2 - aoas[5]:
#         xy_quad = 4
#     else:
#         xy_quad = 5
#     return xy_quad

# quadrant = check_x_y_quadrant(aoa_longest_sides, tdoaMeasurements)
# print("Quadrant =", quadrant)


# Output the calculated TOAs
# print("=============Time of Arrival (TOA) at Each Microphone:==============")
# for i, toa in enumerate(TOA):
#     print(f"Microphone {i+1}: {toa:.6f} seconds")
# print("====================================================================\n")

# # def find_source_position(aoas):

# #     if (aoas[0] == np.pi - aoas[2]):
#         # source_x = 

# def triangulate_sound_source(optimum_time_delays, microphone_positions, speed_of_sound):
#     # Calculate the distances between microphone pairs
#     distance_between_m1_m2 = np.linalg.norm(microphone_positions[1] - microphone_positions[0])
#     distance_between_m3_m4 = np.linalg.norm(microphone_positions[3] - microphone_positions[2])
#     distance_between_m1_m4 = np.linalg.norm(microphone_positions[3] - microphone_positions[0])

#     distance_to_source_1 = optimum_time_delays[0] * speed_of_sound
    
#     distance_to_source_2 = optimum_time_delays[1] * speed_of_sound
    

#     #Calculate distance using time of arrival
#     dist_m1 = TOA[0] * speed_of_sound
#     dist_m2 = TOA[1] * speed_of_sound
#     dist_m3 = TOA[2] * speed_of_sound
#     dist_m4 = TOA[3] * speed_of_sound

#     #Calculate angle between x-axis and path followed to microphone m1
#     arrival_angle_m1_m2 = np.arccos((distance_between_m1_m2**2 + dist_m1**2-dist_m2**2)/(2*dist_m1*distance_between_m1_m2))
#     print(np.degrees(arrival_angle_m1_m2))
#     arrival_angle_m2_m1 = np.arccos((distance_between_m1_m2**2 + dist_m2**2 - dist_m1**2)/(2*distance_between_m1_m2*dist_m2))
#     arrival_angle_m1_m4 = np.pi/2 - np.arccos((distance_between_m1_m4**2 + dist_m1**2 - dist_m4**2)/(2*dist_m1*distance_between_m1_m4))
#     # arrival_angle_m2 = np.arccos(()/())


#     temp = dist_m2*np.sin(arrival_angle_m2_m1)
#     sound_source_x = np.sqrt(dist_m1**2 - temp**2)
#     sound_source_y = np.arctan(sound_source_x/dist_m1)
#     sound_source_z = 0

    
#     if arrival_angle_m1_m2 > np.pi/2:
#         if dist_m1 < dist_m2:
#             sound_source_x = -sound_source_x
#             # arrival_angle_m1_m2 
    
#     if arrival_angle_m1_m2 > np.pi:
#         sound_source_y = -sound_source_y

#     if arrival_angle_m1_m4 < np.pi/2:
#         sound_source_y = sound_source_x*np.sin(arrival_angle_m1_m4)

#     # Triangulate the sound source position
#     sound_source_position = np.array([sound_source_x, sound_source_y, sound_source_z])

#     return sound_source_position

# # Process microphone signals to get optimal time delays
# optimum_time_delays = process_microphone_signals(microphone_signals)
# # print(optimum_time_delays)

# # Triangulate the sound source position
# estimated_sound_source_position = triangulate_sound_source(optimum_time_delays, microphone_positions, speed_of_sound)

# # # Output the estimated sound source location
# print("=========================Simulation Result==========================")
# print(f"Estimated Sound Source Location:\n X = {estimated_sound_source_position[0]:.3f} meters,\n "
#       f"Y = {estimated_sound_source_position[1]:.3f} meters,\n "
#       f"Z = {estimated_sound_source_position[2]:.3f} meters")
# print("====================================================================\n")
