import math, os, random, sys, seaborn, tqdm
import matplotlib.pyplot as plt
from scipy.io import wavfile
import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tf1
from tensorflow.compat.v1.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.compat.v1 import placeholder
from tensorflow.python.ops import io_ops, gen_audio_ops
from tensorflow.python.platform import gfile
from tensorflow.python.util import compat
#======================================================================================#
tf1.disable_eager_execution()
BACKGROUND_NOISE_DIR_NAME = '_background_noise_'
#======================================================================================#
def normalize_in_time_and_amplitude(audio_): # audio_ : should be (48000,)
    max_ = max(abs(audio_)); threshold = 0.01 * max_; length = audio_.shape[0]; start_ = 0; end_ = length; 
    audio_out = np.zeros(length, dtype='float')
    for i in range(length): 
        if(abs(audio_[i]) > threshold): start_ = i; break;
    for i in range(length): 
        if(abs(audio_[length-i-1]) > threshold): end_ = length-i-1; break;
    if( (start_ > 0) or (end_ < length) ): #print('start end =',start_,end_);
        for i in range(length):
            jx = start_ + i / length * (end_ - start_); j = int(np.floor(jx));
            audio_out[i] = ( audio_[j] * (j+1-jx) + audio_[j+1] * (jx-j) ) / max_;
    else:   audio_out = audio_/max_;
    return audio_out
    
def create_model(fingerprint_input, model_settings, model_architecture, is_training):
    if model_architecture == 'fc':
        return create_fc_model(fingerprint_input, model_settings, is_training)
    elif model_architecture == 'cnn2':
        return create_cnn2_model(fingerprint_input, model_settings, is_training)
    elif model_architecture == 'cnn3':
        return create_cnn3_model(fingerprint_input, model_settings, is_training)
    elif model_architecture == 'res':
        return create_res_model(fingerprint_input, model_settings, is_training)
    elif model_architecture == 'oracle':
        return create_oracle_model(fingerprint_input, model_settings, is_training)
    else:
        print('Model_type {} is not defined. Check model type'.format(model_architecture))
        assert(0)

def create_fc_model(tf_feature_in, model_settings, is_training):
    if is_training: dropout_rate = tf.compat.v1.placeholder(tf.float32, name='dropout_rate')
    
    network_output = Dense(model_settings['label_count'])(tf_feature_in)
    
    if is_training: return network_output, dropout_rate
    else: return network_output, network_output

def create_cnn2_model(tf_feature_in, model_settings, is_training):
    if is_training: dropout_rate = placeholder(tf.float32, name='dropout_rate')
    input_frequency_size = model_settings['fingerprint_width']
    input_time_size = model_settings['spectrogram_length']
    network_input = tf.reshape(tf_feature_in, [-1, input_time_size, input_frequency_size, 1])
    
    #conv1 = Conv2D(64, [20,8], strides=[1,1], activation = 'relu', padding = 'same')(network_input)
    conv1 = Conv2D(3, [20,8], strides=[1,1], activation = 'relu', padding = 'same')(network_input)
    if is_training: conv1 = Dropout(rate=dropout_rate)(conv1)
    pool1 = MaxPooling2D([2,2], strides=[2,2], padding='same')(conv1)   
    
    #conv2 = Conv2D(64, [10,4], strides=[1,1], activation = 'relu', padding = 'same')(network_input)
    conv2 = Conv2D(2, [10,4], strides=[1,1], activation = 'relu', padding = 'same')(pool1)
    if is_training: conv2 = Dropout(rate=dropout_rate)(conv2)
    pool2 = MaxPooling2D([2,2], strides=[2,2], padding='same')(conv2)

    flat = Flatten()(pool2)
    network_output = Dense(model_settings['label_count'])(flat)
    if is_training: return network_output, dropout_rate
    else: return network_output, [conv1, pool1, conv2, pool2]

def create_cnn3_model(tf_feature_in, model_settings, is_training):
    if is_training: dropout_rate = placeholder(tf.float32, name='dropout_rate')
    input_frequency_size = model_settings['fingerprint_width']
    input_time_size = model_settings['spectrogram_length']
    network_input = tf.reshape(tf_feature_in, [-1, input_time_size, input_frequency_size, 1])
    
    conv1 = Conv2D(3, [20,8], strides=[1,1], activation = 'relu', padding = 'same')(network_input)
    if is_training: conv1 = Dropout(rate=dropout_rate)(conv1)
    pool1 = MaxPooling2D([2,2], strides=[2,2], padding='same')(conv1)   
    
    conv2 = Conv2D(2, [10,4], strides=[1,1], activation = 'relu', padding = 'same')(pool1)
    if is_training: conv2 = Dropout(rate=dropout_rate)(conv2)
    pool2 = MaxPooling2D([2,2], strides=[2,2], padding='same')(conv2)
    
    conv3 = Conv2D(2, [10,4], strides=[1,1], activation = 'relu', padding = 'same')(pool2)
    if is_training: conv3 = Dropout(rate=dropout_rate)(conv3)
    
    flat = Flatten()(conv3)
    network_output = Dense(model_settings['label_count'])(flat)
    
    if is_training: return network_output, dropout_rate
    else: return network_output, [conv1, pool1, conv2, pool2, conv3]

def create_res_model(tf_feature_in, model_settings, is_training):
    if is_training: dropout_rate = placeholder(tf.float32, name='dropout_rate')
    input_frequency_size = model_settings['fingerprint_width']
    input_time_size = model_settings['spectrogram_length']
    network_input = tf.reshape(tf_feature_in, [-1, input_time_size, input_frequency_size, 1])

    conv1 = Conv2D(8, [20,8], strides=[1,1], activation = None, padding = 'same')(network_input)
    res1 = tf.nn.relu(network_input + conv1)
    if is_training: res1 = Dropout(rate=dropout_rate)(res1)
    pool1 = MaxPooling2D([2,2], strides=[2,2], padding='same')(res1)
    
    conv2 = Conv2D(6, [1,1], strides=[1,1], activation = 'relu', padding = 'same')(pool1)
    
    conv3 = Conv2D(6, [10,4], strides=[1,1], activation = None, padding = 'same')(conv2)
    res2 = tf.nn.relu(conv2 + conv3)
    if is_training: res2 = Dropout(rate=dropout_rate)(res2)
    pool2 = MaxPooling2D([2,2], strides=[2,2], padding='same')(res2)
    
    conv4 = Conv2D(6, [10,4], strides=[1,1], activation = None, padding = 'same')(pool2)
    res3 = tf.nn.relu(pool2 + conv4)
    if is_training: res3 = Dropout(rate=dropout_rate)(res3) 
    
    flat = Flatten()(res3)
    network_output = Dense(model_settings['label_count'])(flat)

    if is_training: return network_output, dropout_rate
    else: return network_output, None

#======================================================================================#
def split_wav(signal, fs):
    min_length_sec = 0.1
    max_length_sec = 1
    width=int(0.2*fs);  # 0.5
    shift=int(0.1*fs); # 0.1
    THR=0.4 # 0.15, 0.4

    out_length_min = int(min_length_sec*fs)
    out_length_max = int(max_length_sec*fs)
    utter_avg = np.mean(np.abs(signal)); 
    speech_or_silence = np.zeros(len(signal)); 
    num_steps = int((len(signal)-width)/shift)

    for fr in range(num_steps):
        begin = fr*shift
        frame=signal[begin:begin+width]
        if np.mean(np.abs(frame))>utter_avg*THR: speech_or_silence[begin:begin+width]=1

    speech_indexes = [np.where(speech_or_silence==0)[0], np.where(speech_or_silence==1)[0]]
    utterance_begin=speech_indexes[True][0]
    utterance_end=speech_indexes[False][np.where(speech_indexes[False]>utterance_begin)[0][0]]
    
    out_sample_count=0
    start_sample=utterance_begin
    end_sample=utterance_end
    out_buffer_list = list()
    wav_start_list = list()
    wav_end_list = list()
    while 1:
        assert(utterance_begin < utterance_end), 'begin(%d),end(%d)'%(utterance_begin,utterance_end)
        wav_start = utterance_begin
        out_buffer = signal[utterance_begin:utterance_end]
        out_sample_count = out_buffer.shape[0]

        if out_sample_count < out_length_min:
            while 1:
                ## Find next begin & end
                try: # Find next begin
                    utterance_begin = speech_indexes[   True][np.where(speech_indexes[   True] > utterance_end)[0][0]]
                except: # No more begin
                    break
                try: # Find next end
                    utterance_end = speech_indexes[False][np.where(speech_indexes[False]>utterance_begin)[0][0]]
                except: # No more end
                    break
            assert(utterance_begin < utterance_end), 'begin(%d),end(%d)'%(utterance_begin,utterance_end)
            out_buffer = np.concatenate([out_buffer, signal[utterance_begin:utterance_end]])
            out_sample_count = out_buffer.shape[0]
            if out_sample_count >= out_length_min: break
        #wav_end = utterance_end
        
        wav_end =  wav_start + int(max_length_sec*fs)
        if len(out_buffer) < out_length_max:
            lack_len = out_length_max - len(out_buffer); append_buffer =signal[utterance_end:utterance_end+lack_len]
            lack_len2 = lack_len - len(append_buffer)
            if lack_len2 == 0:
                out_buffer = np.concatenate([out_buffer, append_buffer])
            elif lack_len2 > 0:
                zero_buffer= np.zeros([lack_len2],np.int16)
                out_buffer = np.concatenate([out_buffer, append_buffer, zero_buffer])
            else: 
                assert(0)
        else: 
            out_buffer = out_buffer[:int(max_length_sec*fs)]

        out_buffer_list.append(out_buffer)
        wav_start_list.append(wav_start)
        wav_end_list.append(wav_end)
        
        ## Find next begin & end
        try: # Find next begin
            utterance_begin = speech_indexes[   True][np.where(speech_indexes[   True] > utterance_end)[0][0]]
        except: # No more begin
            break
        try: # Find next end
            utterance_end = speech_indexes[False][np.where(speech_indexes[False]>utterance_begin)[0][0]]
        except: # No more end
            break
    return out_buffer_list, [speech_or_silence, wav_start_list, wav_end_list]

def load_variables_from_checkpoint(sess, start_checkpoint):
    A =tf1.global_variables()
    B = list()
    for i, layer in enumerate(A):
        if 'L2M_mtx' in str(layer): continue
        B.append(layer)
    saver = tf1.train.Saver(B)
    saver.restore(sess, start_checkpoint)
def which_set(val_percentage, test_percentage):
    rand_val = random.random() *100 # 0~1 --> 0~100
    if rand_val < val_percentage: result = 'validation'
    elif rand_val < (test_percentage + val_percentage): result = 'testing'
    else: result = 'training'
    return result

def prepare_model_settings(allwords, sample_rate, clip_duration_ms, window_size_ms, window_stride_ms, feature_bin_count, preprocess):
    desired_samples = int(sample_rate * clip_duration_ms / 1000)
    window_size_samples = int(sample_rate * window_size_ms / 1000)
    window_stride_samples = int(sample_rate * window_stride_ms / 1000)
    assert(desired_samples > window_size_samples)
    spectrogram_length = 1 + int((desired_samples - window_size_samples) / window_stride_samples)
    average_window_width = -1
    fingerprint_width = feature_bin_count
    fingerprint_size = fingerprint_width * spectrogram_length
    label_count = len(allwords)

    sess = tf1.InteractiveSession()
    tf_L2M_mtx = tf.signal.linear_to_mel_weight_matrix(fingerprint_width, 
                                nextpow2(window_size_samples)//2+1, sample_rate, 0, 12000)
    L2M_mtx=sess.run(tf_L2M_mtx)
    tf1.reset_default_graph()
    sess.close()
    return {
      'desired_samples': desired_samples,
      'window_size_samples': window_size_samples,
      'window_stride_samples': window_stride_samples,
      'spectrogram_length': spectrogram_length,
      'fingerprint_width': fingerprint_width,
      'fingerprint_size': fingerprint_size,
      'label_count': label_count,
      'sample_rate': sample_rate,
      'preprocess': preprocess,
      'average_window_width': average_window_width,
      'clip_duration_ms': clip_duration_ms,
      'feature_bin_count': feature_bin_count,
      'window_size_ms': window_size_ms,
      'window_stride_ms': window_stride_ms,
      'allwords': allwords,
      'L2M':L2M_mtx}
      
def nextpow2(p):     
    n = 2
    while p > n:  
        n *= 2 
    return n
def print_table(list_):
    print("|"+" | ".join("%7s"%(str(x)) for x in list_)+" |")
def prepare_words_list(wanted_words):
    return wanted_words

def create_oracle_model(tf_feature_in, model_settings, is_training):
    if is_training: dropout_rate = placeholder(tf.float32, name='dropout_rate')
    input_frequency_size = model_settings['fingerprint_width']
    input_time_size = model_settings['spectrogram_length']
    network_input = tf.reshape(tf_feature_in, [-1, input_time_size, input_frequency_size, 1])
    #--------
    res1 = Conv2D(16, [20,8], strides=[1,1], activation = None, padding = 'same')(network_input)
    res1 = tf.nn.relu(network_input + res1)
    if is_training: res1 = Dropout(rate=dropout_rate)(res1)
    res1 = Conv2D(32, [2,2], strides=[2,2], activation = 'relu', padding = 'same')(res1)
    if is_training: res1 = Dropout(rate=dropout_rate)(res1) 
    #--------
    res2 = Conv2D(32, [10,4], strides=[1,1], activation = None, padding = 'same')(res1)
    res2 = tf.nn.relu(res1 + res2)
    if is_training: res2 = Dropout(rate=dropout_rate)(res2)
    res2 = Conv2D(64, [2,2], strides=[2,2], activation = 'relu', padding = 'same')(res2)
    if is_training: res2 = Dropout(rate=dropout_rate)(res2) 
    #--------
    res3 = Conv2D(64, [10,4], strides=[1,1], activation = None, padding = 'same')(res2)
    res3 = tf.nn.relu(res2 + res3)
    if is_training: res3 = Dropout(rate=dropout_rate)(res3) 
    res4 = Conv2D(64, [10,4], strides=[1,1], activation = None, padding = 'same')(res3)
    res4 = tf.nn.relu(res3 + res4)
    if is_training: res4 = Dropout(rate=dropout_rate)(res4) 
    res4 = Conv2D(128, [1,1], strides=[1,1], activation = 'relu', padding = 'same')(res4)
    if is_training: res4 = Dropout(rate=dropout_rate)(res4) 
    #--------
    res5 = Conv2D(128, [10,4], strides=[1,1], activation = None, padding = 'same')(res4)
    res5 = tf.nn.relu(res4 + res5)
    if is_training: res5 = Dropout(rate=dropout_rate)(res5) 
    
    res6 = Conv2D(128, [10,4], strides=[1,1], activation = None, padding = 'same')(res5)
    res6 = tf.nn.relu(res5 + res6)
    if is_training: res6 = Dropout(rate=dropout_rate)(res6) 
    res6 = Conv2D(64, [1,1], strides=[1,1], activation = 'relu', padding = 'same')(res6)
    if is_training: res6 = Dropout(rate=dropout_rate)(res6) 
    #--------
    res7 = Conv2D(64, [10,4], strides=[1,1], activation = None, padding = 'same')(res6)
    res7 = tf.nn.relu(res6 + res7)
    if is_training: res7 = Dropout(rate=dropout_rate)(res7) 
    
    res8 = Conv2D(64, [10,4], strides=[1,1], activation = None, padding = 'same')(res7)
    res8 = tf.nn.relu(res7 + res8)
    if is_training: res8 = Dropout(rate=dropout_rate)(res8) 
    res8 = Conv2D(16, [1,1], strides=[1,1], activation = 'relu', padding = 'same')(res8)
    if is_training: res8 = Dropout(rate=dropout_rate)(res8) 
    #--------
    flat = Flatten()(res8)
    network_output = Dense(model_settings['label_count'])(flat)

    if is_training: return network_output, dropout_rate
    else: return network_output, None    
#======================================================================================#
class AudioProcessor(object):
    def __init__(self, data_dir, sil_per, unk_per, wanted_words, load_all_mode):
        self.data_dir = data_dir
        self.prepare_data_index(sil_per, unk_per, wanted_words)
        self.wanted_words = wanted_words
        if load_all_mode == True: self.load_all_set();
        self.report_dataset()

    def prepare_data_index(self, sil_per, unk_per, wanted_words):
        # Make sure the shuffling and picking of unknowns is deterministic.
        random.seed(59185)
        wanted_words_index = {}
        for index, wanted_word in enumerate(wanted_words):
            wanted_words_index[wanted_word] = index
        self.data_index = {'validation': [], 'testing': [], 'training': []}
        all_words = {}
        
        search_path = os.path.join(self.data_dir, '*', '*.npy') # assume: data_dir/NHJ20192743/ALEXA/ALEXA~.wav
        for wav_path in gfile.Glob(search_path):
            set_index = wav_path.split(os.sep)[-2]
            word = wav_path.split(os.sep)[-1].split('_')[0]
            all_words[word] = True
            if word in wanted_words_index: self.data_index[set_index].append({'label': word, 'file': wav_path})
            else: pass
        if not all_words:
            raise Exception('No .wavs found at ' + search_path)
        for index, wanted_word in enumerate(wanted_words):
            if wanted_word not in all_words:
                raise Exception('X: '+wanted_word+', O: '+', '.join(all_words.keys()))

        # Make sure the ordering is random.
        for set_index in ['validation', 'testing', 'training']:
            random.shuffle(self.data_index[set_index])
        # Prepare the rest of the result data structure.
        self.words_list = prepare_words_list(wanted_words)
        self.word_to_index = {}
        for word in all_words:
            if word in wanted_words_index: self.word_to_index[word] = wanted_words_index[word]

    def set_size(self, mode):
        return len(self.data_index[mode])

    def load_all_set(self):
        for set_index in ['validation','testing','training']:
            for ii in range(len(self.data_index[set_index])):
                path = self.data_index[set_index][ii]['file']
                self.data_index[set_index][ii]['data'] =  np.load(path).flatten()
        return 
            

    def get_data_npy(self, how_many, offset, setting, mode, sess, load_all_mode=False):
        # Pick one of the partitions to choose samples from.
        candidates = self.data_index[mode]
        if how_many == -1: sample_count = len(candidates) # get_all
        else: sample_count = max(0, min(how_many, len(candidates) - offset))
        # Data and labels will be populated and returned.
        #data = np.zeros((sample_count, setting['fingerprint_size']))
        labels = np.zeros(sample_count)
        desired_samples = setting['desired_samples']
        pick_deterministically = (mode != 'training')
        # Use the preprocessor created earlier to repeatedly to generate the final output data
        data = list()
        for i in range(offset, offset + sample_count):
            # Pick which audio sample to use.
            sample = candidates[i]
            if load_all_mode==False: data_tensor = np.load(sample['file']).flatten()
            else: data_tensor = sample['data']
            data.append(data_tensor)
            labels[i - offset] = self.word_to_index[sample['label']]
        data = np.vstack(data)
        return data, labels

    def report_dataset(self):
        word_cnt = dict()
        for set_index in ['training','validation','testing']:
            word_cnt[set_index] = [0] * len(self.word_to_index)
            for data in self.data_index[set_index]:
                word_cnt[set_index][self.word_to_index[data['label']]] += 1
        #print('--------------------------------------------------------------------------------')
        #print('Setting: ',self.setting)
        print('------------------------------------------------------------')
        print_table(["     "]+ self.wanted_words)
        print_table(["Train"]+ word_cnt['training'])
        print_table(["Valid"]+ word_cnt['validation'])
        print_table(["Test "]+ word_cnt['testing'])
        print('------------------------------------------------------------')
        return 
