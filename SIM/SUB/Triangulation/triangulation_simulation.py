import numpy as np
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
print("====================================================================")
print()

# Set microphone m1 as default microphone
m1 = [0, 0, 0]
m2 = []
m3 = []
m4 = []
microphone_positions = np.array([[], [], [], []])
temperature = 25  # Default temperature in Celsius
speed_of_sound = 331.4 * np.sqrt(1 + (temperature / 273.15))
source_coord = [0.05,1,1]
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

# plt.show()

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
print("=============Time of Arrival (TOA) at Each Microphone:==============")
for i, toa in enumerate(TOA):
    print(f"Microphone {i+1}: {toa:.6f} seconds")
print("====================================================================\n")

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
    arrival_angle_m1_m4 = np.pi/2 - np.arccos((distance_between_m1_m4**2 + dist_m1**2 - dist_m4**2)/(2*dist_m1*distance_between_m1_m4))
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
        sound_source_y = sound_source_x*np.sin(arrival_angle_m1_m4)

    # Triangulate the sound source position
    sound_source_position = np.array([sound_source_x, sound_source_y, sound_source_z])

    return sound_source_position

# Process microphone signals to get optimal time delays
optimum_time_delays = process_microphone_signals(microphone_signals)
# print(optimum_time_delays)

# Triangulate the sound source position
estimated_sound_source_position = triangulate_sound_source(optimum_time_delays, microphone_positions, speed_of_sound)

# # Output the estimated sound source location
print("=========================Simulation Resul===t=======================")
print(f"Estimated Sound Source Location: X = {estimated_sound_source_position[0]:.3f} meters, "
      f"Y = {estimated_sound_source_position[1]:.3f} meters, "
      f"Z = {estimated_sound_source_position[2]:.3f} meters")
print("====================================================================\n")
