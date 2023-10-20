
from tkinter import *
import subprocess
import time
import numpy as np
import matplotlib.pyplot as plt
import wave

#SSH To Pis and record
def btn1Clc():
    lbl.config(text = "Stutus: Connecting");
    window.update();
    k = 0;
    start_time = time.time()
    while (k<1000):
        elapsed_time = time.time() - start_time 
        if elapsed_time > 30:
            lbl.config(text="Status: Connection Timed Out")
            window.update();
            break
        subprocess.Popen(f"echo raspberry | ssh pi@raspberrypi12.local arecord -D plughw:0 -c2 -r 48000 -f S32_LE -t wav -V stereo -v file_stereo.wav", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()
        subprocess.Popen(f"echo raspberry | ssh pi@raspberrypi13.local arecord -D plughw:0 -c2 -r 48000 -f S32_LE -t wav -V stereo -v file_stereo.wav", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()
        lbl.config(text = "Stutus: Recording");
        window.update();        
        time.sleep(10);
        subprocess.Popen(f"echo raspberry | ssh pi@raspberrypi12.local ctrl+x", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()
        subprocess.Popen(f"echo raspberry | ssh pi@raspberrypi13.local ctrl+x", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()
        k = k+1
        window.update();
    lbl.config(text = "Stutus: Connection Failed");
    window.update();

#SCP From Pis
def btn2Clc():
    lbl.config(text = "Stutus: SCP Starting");
    window.update();
    subprocess.Popen(f"echo raspberry | scp pi@raspberrypi12:file_stereo.wav file1.wav", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()
    subprocess.Popen(f"echo raspberry | scp pi@raspberrypi13:file_stereo.wav file2.wav", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()
    lbl.config(text = "Stutus: SCP Successful");
    window.update(); 

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

#Calculate
def btn3Clc():
    lbl.config(text = "Stutus: Calculating");
    window.update();


    # Load the two WAV files
    file_path1 = '/home/portm/file2.wav'
    file_path2 = '/home/portm/file1.wav'

    # Open the WAV files
    wav_file1 = wave.open(file_path1, 'rb')
    wav_file2 = wave.open(file_path2, 'rb')

    # Read the audio data
    signal1 = np.frombuffer(wav_file1.readframes(-1), dtype=np.int16)
    signal2 = np.frombuffer(wav_file2.readframes(-1), dtype=np.int16)

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
    txt = "Time Delay: {td} seconds"
    lbl2.config(text = txt.format(rd = time_delay));
    window.update();

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
  

#UI Design
window = Tk()

btn1=Button(window, text="SSH Pis", fg='black')
btn1.config(command=lambda: btn1Clc())
btn1.place(x=80, y=100)

btn2=Button(window, text="SCP From Pis", fg='black')
btn2.config(command=lambda: btn2Clc())
btn2.place(x=180, y=100)

btn3=Button(window, text="Calculate", fg='black')
btn3.config(command=lambda: btn3Clc())
btn3.place(x=280, y=100)

lbl=Label(window, text="Status:Idle", fg='black', font=("Helvetica", 10))
lbl.place(x=180, y=50)

lbl2=Label(window, text="Time Delay:", fg='black', font=("Helvetica", 10))
lbl2.place(x=400, y=50)

window.title('Acoustic Triangulation')
window.geometry("1000x400+10+20")
window.mainloop()

window.mainloop();

