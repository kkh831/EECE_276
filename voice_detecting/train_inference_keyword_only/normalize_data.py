# python normalize_data.py ..\data\small_dataset ..\data\small_norm_melsp
import os, sys, tqdm, pickle, shutil
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import soundfile as sf
from scipy.io import wavfile
import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.compat.v1 as tf1
from tensorflow.python.ops import io_ops, gen_audio_ops
from model import prepare_model_settings, prepare_words_list, normalize_in_time_and_amplitude, which_set, BACKGROUND_NOISE_DIR_NAME
#=========================================================================#
def graph_audio2logmel(setting):
    mel_engine = dict()
    tf_audio = tf1.placeholder(tf.float32, [setting['desired_samples'], 1])#
    tf_stft = gen_audio_ops.audio_spectrogram(tf_audio, window_size=setting['window_size_samples'], stride=setting['window_stride_samples'], magnitude_squared=True)
    tf_L2M_mtx = tf.Variable(initial_value = setting['L2M'], dtype=tf.float32, name='L2M_mtx')
    tf_logmel = tf.math.log(tf.tensordot(tf_stft, tf_L2M_mtx, 1)+1e-6)
    mel_engine['sess'] = tf1.InteractiveSession(); mel_engine['audio'] = tf_audio; mel_engine['logmel'] = tf_logmel
    mel_engine['sess'].run(tf1.global_variables_initializer())
    if False: print(' [mel_engine]\n - AUDIO :{}\n - STFT  :{}\n - LogMEL:{}\n'.format(tf_audio, tf_stft, tf_logmel))
    return mel_engine
def calc_audio2logmel(mel_engine, audio):
    logmel = mel_engine['sess'].run(mel_engine['logmel'], feed_dict={mel_engine['audio']:audio})
    return logmel
#=========================================================================#
#=========================================================================#
## MAIN ##
print('#-------------------------------------------------------------------------#')
keywords = ['ALEXA','BIXBY','GOOGLE','JINIYA','KLOVA']
parent_dir_src = sys.argv[1]
#parent_dir_dst = parent_dir_src+'_preprocessed'
parent_dir_dst = sys.argv[2]
enable_noise = True
validation_percentage = 10
testing_percentage = 10
print('keywords              :',keywords)
print('parent_dir_src        :',parent_dir_src)
print('parent_dir_dst        :',parent_dir_dst)
print('enable_noise          :',enable_noise)
print('validation_percentage :',validation_percentage)
print('testing_percentage    :',testing_percentage)
print('#-------------------------------------------------------------------------#')
#-------------------------------------------------------------------------#
# Initial setting
try: 
    if os.path.exists(parent_dir_dst):
        input = input('Do you want to erase "{}"? (y/n) '.format(parent_dir_dst))
        if input.lower() == 'y' or input.lower() == 'yes': shutil.rmtree(parent_dir_dst)
        else: print('Remove {}. Stop program.'.format(parent_dir_dst)); sys.exit()
    os.makedirs(parent_dir_dst, exist_ok=False)
    os.makedirs(parent_dir_dst+'/training', exist_ok=False)
    os.makedirs(parent_dir_dst+'/validation', exist_ok=False)
    os.makedirs(parent_dir_dst+'/testing', exist_ok=False)
except Exception as ex:
    print('EX:', ex)
    print('Erase the {} folder before run this code.'.format(parent_dir_dst))
setting = prepare_model_settings(prepare_words_list(keywords), 48000, 1000, 20, 10, 80, "mel")
with open(os.path.join(parent_dir_dst,'data_setting.bin'),'wb') as f:
    pickle.dump(setting, f, protocol=pickle.HIGHEST_PROTOCOL)
mel_engine = graph_audio2logmel(setting)  
#-------------------------------------------------------------------------#
# Load noise
noise_audio = None
if enable_noise == True:
    noise_dir = os.path.join(parent_dir_src, BACKGROUND_NOISE_DIR_NAME)
    noise_audio = list()
    for filename in os.listdir(noise_dir):
        filepath = os.path.join(noise_dir, filename)
        [fs, audio] = wavfile.read(filepath)
        assert fs == 48000, '{}, sample_rate {} != 48000'.format(filepath, fs)
        audio = audio / max(abs(audio)) # norm in -1 ~ 1
        noise_audio.append(audio)
    noise_audio = np.hstack(noise_audio)
    print('noise_audio:',noise_audio.shape)

#=========================================================================#
for stu_dir in tqdm.tqdm(os.listdir(parent_dir_src)):
    stu_path = os.path.join(parent_dir_src, stu_dir)
    if stu_dir == BACKGROUND_NOISE_DIR_NAME: continue
    for key_dir in os.listdir(stu_path):
        if key_dir == "UNKNOWN": continue
        key_path = os.path.join(stu_path, key_dir)
        for filename in os.listdir(key_path):
            # 1. Load audio
            filepath = os.path.join(key_path, filename)
            [fs, audio]= wavfile.read(filepath)
            assert fs == 48000, '{}, sample_rate {} != 48000'.format(filepath, fs)
            if len(audio) < setting['desired_samples']:
                audio = np.hstack([audio, np.zeros(setting['desired_samples'] - len(audio))])
            elif len(audio) > setting['desired_samples']: 
                print('wavfile :{} = {} > 48000'.format(filepath, audio.shape))
                audio = audio[:48000] # cut leftover
            norm_audio = normalize_in_time_and_amplitude(audio)
            set_index = which_set(validation_percentage, testing_percentage)
            
            # 2. Mix noise
            if (enable_noise == True) and (set_index=="training"):
                bg_ratio = [1 if np.random.uniform(0, 0.3) < 0.3 else 0]
                bg_volume = np.random.uniform(0, 0.15)
                offset = np.random.randint(0, len(noise_audio)-setting['desired_samples'])
                noise_chunk = noise_audio[offset : offset+setting['desired_samples']]
                mix = norm_audio + noise_chunk * bg_volume * bg_ratio
            else:
                mix = norm_audio
            
            # 3. calc logMel
            mix = np.expand_dims(mix, axis=-1)
            logmel = calc_audio2logmel(mel_engine, mix) # logmel : [1,99,80]
            
            # 4. Show info; raw audio, norm audio, pcolormesh
            if False:
                #print('audio name:', filename)
                #print('audio:', audio.shape)
                #print('norm_audio:', norm_audio.shape)
                #print('noise_chunk:', noise_chunk.shape)
                sd.play(audio/max(abs(audio)), 48000); sd.wait()
                sd.play(norm_audio, 48000); sd.wait()
                plt.figure(1);plt.plot(audio/max(abs(audio)));plt.plot(norm_audio);
                plt.figure(2);plt.pcolormesh(np.squeeze(logmel).T); plt.show()
            
            # 5. save data
            savepath = os.path.join(parent_dir_dst, set_index, filename[:-4]+'.npy')
            np.save(savepath, logmel)
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
