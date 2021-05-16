# python process_voice_1.py domisoldo.wav re_recorded.wav max_frq_spectrogram voice_2_index_offset
# python pr3_2voice.py domisoldo.wav 2nd_recording.wav 8000 6067
import numpy as np , scipy.io as sio , scipy.io.wavfile , matplotlib.pyplot as plt
import librosa.display , librosa , soundfile , signal, sys, sounddevice as sd
from scipy import signal
SAMPLE_RATE=48000; wav_file_1 = sys.argv[1]; wav_file_2 = sys.argv[2]
voice_1,sr=librosa.load(wav_file_1,sr=SAMPLE_RATE);sd.play(voice_1,sr);sd.wait()
voice_2,sr=librosa.load(wav_file_2,sr=SAMPLE_RATE);##sd.play(1.5*voice_2,sr);sd.wait()
# compute power of voice_1 & voice_2
power_1 = 0; power_2 = 0;
for i in range(len(voice_1)): power_1 += voice_1[i]**2;
for i in range(len(voice_2)): power_2 += voice_2[i]**2;
print('power of original voice and 2nd-recorded voice =',power_1,power_2);
ratio_amplitude = np.sqrt(power_1/power_2);sd.play(ratio_amplitude*voice_2,sr);sd.wait()
voice_2_index_offset = int(sys.argv[4]);
voice_2_= ratio_amplitude*voice_2[voice_2_index_offset:min(len(voice_1),len(voice_2))]
voice_1_=voice_1[:len(voice_2_)]; time_1 = np.linspace(0, len(voice_1_)/sr, len(voice_1_)); 
plt.figure(1); plt.plot(time_1,voice_1_,'b'); 
time_2 = np.linspace(0, len(voice_2_)/sr, len(voice_2_)); 
plt.plot(time_2,voice_2_,'r'); plt.xlabel('time (s)'); 
plt.title('Waveform: original(BLUE) 2nd recording(RED)'); plt.show()

hop_points = 512; n_fft = 8192  
stf_1 = librosa.stft(voice_1_, n_fft=n_fft, hop_length=hop_points)
stf_2 = librosa.stft(voice_2_, n_fft=n_fft, hop_length=hop_points)

mag_1 = np.abs(stf_1); log_spectrogram_1 = librosa.amplitude_to_db(mag_1)
mag_2 = np.abs(stf_2); log_spectrogram_2 = librosa.amplitude_to_db(mag_2)
hz_range = [0, float(sys.argv[3])]
bin_range = (np.array(hz_range)/(sr/2) * (n_fft/2+1)).astype(int)
log_spec_1 = log_spectrogram_1[bin_range[0]:bin_range[1],:]
x_axis_1 = np.arange(0, log_spec_1.shape[1]*hop_points/sr, hop_points/sr)
y_axis_1 = np.arange(hz_range[0], hz_range[1], (hz_range[1]-hz_range[0])/(bin_range[1]-bin_range[0]))
plt.figure(2);plt.subplot(2,1,1);
plt.pcolormesh(x_axis_1, y_axis_1, log_spec_1)
plt.xlabel("Time"); plt.ylabel("Frequency"); plt.title("Original Spectrogram (dB)"); #plt.show()
log_spec_2 = log_spectrogram_2[bin_range[0]:bin_range[1],:]; plt.subplot(2,1,2);
x_axis_2 = np.arange(0, log_spec_2.shape[1]*hop_points/sr, hop_points/sr)
plt.pcolormesh(x_axis_2, y_axis_1, log_spec_2)
plt.xlabel("Time"); plt.ylabel("Frequency"); plt.title("2nd Recording Spectrogram (dB)"); plt.show()
# average spectrum of dol mi sol doh
dol_start_time=0.95; dol_end_time=2.01; mi_start_time=2.02; mi_end_time=3.03; 
sol_start_time=3.18; sol_end_time=4.38;
dol_start_index=int(dol_start_time*sr/hop_points); dol_end_index=int(dol_end_time*sr/hop_points);
mi_start_index=int(mi_start_time*sr/hop_points); mi_end_index=int(mi_end_time*sr/hop_points);
sol_start_index=int(sol_start_time*sr/hop_points); sol_end_index=int(sol_end_time*sr/hop_points);

Nf = len(log_spectrogram_1); # n_fft/2
dol_1_spectrum=np.zeros(Nf,dtype='float'); mi_1_spectrum=np.zeros(Nf,dtype='float');
sol_1_spectrum=np.zeros(Nf,dtype='float');
dol_2_spectrum=np.zeros(Nf,dtype='float'); mi_2_spectrum=np.zeros(Nf,dtype='float');
sol_2_spectrum=np.zeros(Nf,dtype='float');
for i in range(dol_start_index,dol_end_index): 
    dol_1_spectrum += np.abs(mag_1[:,i])**2; dol_2_spectrum += np.abs(mag_2[:,i])**2;
dol_1_spectrum /= (dol_end_index-dol_start_index); dol_2_spectrum /= (dol_end_index-dol_start_index); 
dol_1_spectrum = np.sqrt(dol_1_spectrum ); dol_2_spectrum = np.sqrt(dol_2_spectrum ); 
for i in range(mi_start_index,mi_end_index): 
    mi_1_spectrum += np.abs(mag_1[:,i])**2; mi_2_spectrum += np.abs(mag_2[:,i])**2;
mi_1_spectrum /= (mi_end_index-mi_start_index); mi_2_spectrum /= (mi_end_index-mi_start_index); 
mi_1_spectrum = np.sqrt(mi_1_spectrum ); mi_2_spectrum = np.sqrt(mi_2_spectrum ); 
for i in range(sol_start_index,sol_end_index): 
    sol_1_spectrum += np.abs(mag_1[:,i])**2; sol_2_spectrum += np.abs(mag_2[:,i])**2;
sol_1_spectrum /= (sol_end_index-sol_start_index); sol_2_spectrum /= (sol_end_index-sol_start_index); 
sol_1_spectrum = np.sqrt(sol_1_spectrum ); sol_2_spectrum = np.sqrt(sol_2_spectrum ); 

#
fbin=np.linspace(0,sr/2,Nf);
plt.figure(4); plt.plot(fbin,dol_1_spectrum,'b'); plt.plot(fbin,dol_2_spectrum,'r'); 
plt.title('time average LINEAR spectrum of DO low: Original(BLUE) 2nd Recording(RED)'); plt.show()
plt.figure(5); plt.plot(fbin,mi_1_spectrum,'b'); plt.plot(fbin,mi_2_spectrum,'r'); 
plt.title('time average LINEAR spectrum of MI: Original(BLUE) 2nd Recording(RED)'); plt.show()
plt.figure(6); plt.plot(fbin,sol_1_spectrum,'b'); plt.plot(fbin,sol_2_spectrum,'r'); 
plt.title('time average LINEAR spectrum of SOL: Original(BLUE) 2nd Recording(RED)'); plt.show()

plt.figure(14); plt.semilogy(fbin,dol_1_spectrum,'b'); plt.semilogy(fbin,dol_2_spectrum,'r'); 
plt.title('time average LOG spectrum of DO low: Original(BLUE) 2nd Recording(RED)'); plt.show()
plt.figure(15); plt.semilogy(fbin,mi_1_spectrum,'b'); plt.semilogy(fbin,mi_2_spectrum,'r'); 
plt.title('time average LOG spectrum of MI: Original(BLUE) 2nd Recording(RED)'); plt.show()
plt.figure(16); plt.semilogy(fbin,sol_1_spectrum,'b'); plt.semilogy(fbin,sol_2_spectrum,'r'); 
plt.title('time average LOG spectrum of SOL: Original(BLUE) 2nd Recording(RED)'); plt.show()
plt.figure(17); plt.semilogy(fbin,doh_1_spectrum,'b'); plt.semilogy(fbin,doh_2_spectrum,'r');
plt.title('time average LOG spectrum of DO high: Original(BLUE) 2nd Recording(RED)'); plt.show()
