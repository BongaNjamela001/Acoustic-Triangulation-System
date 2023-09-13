microphone = phased.OmnidirectionalMicrophoneElement('FrequencyRange',[20 20e3]);

Nele = 10;
ula = phased.ULA(Nele,0.05,'Element',microphone);
c = 343;%sound speed, in m/s

ang_dft = [-30; 0];
ang_cleanspeech = [-10; 10];
ang_laughter = [20; 0];

fs = 8000;
collector = phased.WidebandCollector('Sensor',ula,'PropagationSpeed',c,'SampleRate',fs,'NumSubbands',1000,'ModulatedInput', false);

t_duration = 3;  % 3 seconds
t = 0:1/fs:t_duration-1/fs;

prevS = rng(2008);
noisePwr = 1e-4; % noise power

% preallocate
NSampPerFrame = 1000;
NTSample = t_duration*fs;
sigArray = zeros(NTSample,Nele);
voice_dft = zeros(NTSample,1);
voice_cleanspeech = zeros(NTSample,1);
voice_laugh = zeros(NTSample,1);

% set up audio device writer
audioWriter = audioDeviceWriter('SampleRate',fs, 'SupportVariableSizeInput', true);
isAudioSupported = (length(getAudioDevices(audioWriter))>1);

dftFileReader = dsp.AudioFileReader('dft_voice_8kHz.wav','SamplesPerFrame',NSampPerFrame);
speechFileReader = dsp.AudioFileReader('cleanspeech_voice_8kHz.wav','SamplesPerFrame',NSampPerFrame);
laughterFileReader = dsp.AudioFileReader('laughter_8kHz.wav','SamplesPerFrame',NSampPerFrame);

% simulate
for m = 1:NSampPerFrame:NTSample
    sig_idx = m:m+NSampPerFrame-1;
    x1 = dftFileReader();
    x2 = speechFileReader();
    x3 = 2*laughterFileReader();
    temp = collector([x1 x2 x3],[ang_dft ang_cleanspeech ang_laughter]) + sqrt(noisePwr)*randn(NSampPerFrame,Nele);
    if isAudioSupported
        play(audioWriter,0.5*temp(:,3));
    end
    sigArray(sig_idx,:) = temp;
    voice_dft(sig_idx) = x1;
    voice_cleanspeech(sig_idx) = x2;
    voice_laugh(sig_idx) = x3;
end

plot(t,sigArray(:,3));
xlabel('Time (sec)'); ylabel ('Amplitude (V)');
title('Signal Received at Channel 3'); ylim([-3 3]);

angSteer = ang_dft;
beamformer = phased.TimeDelayBeamformer('SensorArray',ula,'SampleRate',fs,'Direction',angSteer,'PropagationSpeed',c)

signalsource = dsp.SignalSource('Signal',sigArray,'SamplesPerFrame',NSampPerFrame);

cbfOut = zeros(NTSample,1);

for m = 1:NSampPerFrame:NTSample
    temp = beamformer(signalsource());
    if isAudioSupported
        play(audioWriter,temp);
    end
    cbfOut(m:m+NSampPerFrame-1,:) = temp;
end

plot(t,cbfOut);
xlabel('Time (Sec)'); ylabel ('Amplitude (V)');
title('Time Delay Beamformer Output'); ylim([-3 3]);

agCbf = pow2db(mean((voice_cleanspeech+voice_laugh).^2+noisePwr)/mean((cbfOut - voice_dft).^2))