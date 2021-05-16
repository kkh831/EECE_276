# Usage: python weight_info.py (ckpt_path)
# Example: python weight_info.py result_dir/ckpt/ep~.ckpt
import os, sys, glob, pickle
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
import tensorflow.compat.v1 as tf1
import matplotlib.pyplot as plt
from scipy.io import wavfile
from matplotlib import gridspec
from tensorflow.python.ops import gen_audio_ops
from model import load_variables_from_checkpoint, create_model
#===================================================================================#
def create_inference_graph(setting):
  wav_data_placeholder = tf.compat.v1.placeholder(tf.string, [], name='wav_data')
  decoded_sample_data = tf.audio.decode_wav(wav_data_placeholder, desired_channels=1, desired_samples=setting['desired_samples'], name='decoded_sample_data')
  input_ = tf.reshape(decoded_sample_data.audio,[setting['desired_samples'],1])
  spectrogram = gen_audio_ops.audio_spectrogram(input_, window_size=setting['window_size_samples'], stride=setting['window_stride_samples'], magnitude_squared=True)

  L2M_mtx = tf.constant(setting['L2M'], shape = setting['L2M'].shape, dtype=tf.float32, name="L2M_mtx")
  tf_out = tf.math.log(tf.tensordot(spectrogram, L2M_mtx, 1)+1e-6)
  
  #print('tf_out:', setting['preprocess'],':',tf_out)
  fingerprint_size = setting['fingerprint_size']
  reshaped_input = tf.reshape(tf_out, [-1, fingerprint_size])
  logits, _ = create_model(reshaped_input, setting, setting['model_architecture'], is_training=False)
  softmax = tf.nn.softmax(logits, name='labels_softmax')
  return input_, softmax, tf_out
  
def off_axis(axes, num_plotted):
    num_plotted += 1
    if num_plotted == 16: return;
    [yy, xx] = axes.shape
    for y in range(num_plotted//4, yy):
        for x in range(num_plotted%4, xx):
            axes[y,x].axis("off")
    return

def plot_4d_weight(weight, bias):
    [width, height, ch_in, ch_out] = weight.shape
    if width == 1 and height == 1:
        print('Weight shape is ',weight.shape,'. So just print it:', weight)
    else:
        idx2 = 0; fig_cnt = 0; fig, axes = plt.subplots(4,4); fig_name='{}, (ch_in,ch_out)=({}, {}): '.format(layer_name, ch_in, ch_out)
        for ii in range(ch_in):
            for oo in range(ch_out):
                idx = ii * ch_out + oo
                idx2 = idx%16
                weight_plot = np.squeeze(weight[:,:, ii, oo].T)
                if len(weight_plot.shape) == 2:
                    axes[idx2//4, idx2%4].contourf(weight_plot)
                else: # It must be 1
                    axes[idx2//4, idx2%4].plot(weight_plot,'b*--')
                    
                if idx2 + 1 == 16: 
                    fig.suptitle(fig_name+str(fig_cnt)); off_axis(axes, idx2); plt.show(); plt.close(fig); 
                    fig_cnt += 1; fig, axes = plt.subplots(4,4);
        if idx2 + 1 != 16: fig.suptitle(fig_name+str(fig_cnt)); off_axis(axes, idx2); plt.show(); 
        plt.close(fig);
    plt.plot(bias, 'b.--'); plt.title(layer_name+': bias'); plt.show()
    return; 

#===================================================================================#
ckpt_dir =  '/'.join(os.path.split(sys.argv[1])[:-1])
setting_path = os.path.abspath(glob.glob(os.path.join(ckpt_dir,'..','setting.bin'))[0]); #print(setting_path)
with open(setting_path,'rb') as f:
    setting = pickle.load(f)
sess = tf.compat.v1.InteractiveSession()
tf_in, tf_out, tf_pp = create_inference_graph(setting)
load_variables_from_checkpoint(sess, sys.argv[1])
all_variables = tf1.global_variables()
var_without_L2M = list()
for i, variable in enumerate(all_variables):
    if 'L2M_mtx' in str(variable): continue
    var_without_L2M.append(variable)
model_type = setting['model_architecture']

#===================================================================================#
if 'fc' in model_type:
    assert(len(var_without_L2M) == 2), "len(var_without_L2M) == {}, should be 2".format(len(var_without_L2M))
    weight_flat = None; bias = None
    for i, tf_variable in enumerate(var_without_L2M):
        var_name = tf_variable.name
        variable = sess.run([tf_variable])[0]
        print('{}:{}'.format(var_name, variable.shape))
        if 'bias' in var_name: bias = variable
        elif 'kernel' in var_name: weight_flat = variable
    weight = weight_flat.reshape([setting['spectrogram_length'], setting['fingerprint_width'], setting['label_count']])
    print('weight:', weight.shape)
    print('bias:', bias.shape, bias)
    fig, axes = plt.subplots(2,4);
    for idx in range(setting['label_count']):
        axes[idx//4,idx%4].contourf(weight[:,:,idx].T)
    axes[1,3].axis("off"); plt.title('weight'); plt.show()
    plt.plot(bias, 'b.--'); plt.title('bias'); plt.show()
        
elif 'cnn' in model_type:
    weight = None; bias = None
    for i, tf_variable in enumerate(var_without_L2M):
        var_name = tf_variable.name
        layer_name = var_name.split('/')[0]
        variable = sess.run([tf_variable])[0]
        print('{}:{}'.format(var_name, variable.shape))
        
        if (type(weight) == type(None)) or (type(bias) == type(None)):
            if 'bias' in var_name : bias = variable
            elif 'kernel' in var_name: weight = variable
            else: print("Unexpectable", variable); assert(0)
            
        if (type(weight) != type(None)) and (type(bias) != type(None)):
            if len(weight.shape) == 2: pass
            else: 
                plot_4d_weight(weight, bias)
                last_bias = bias; weight = None; bias = None
    try: 
        try :
            weight = weight.reshape([25, 20, last_bias.shape[0], setting['label_count']]) # pool 2 time
        except:
            try:
                weight = weight.reshape([13, 10, last_bias.shape[0], setting['label_count']]) # pool 3 time
            except:
                weight = weight.reshape([7, 5, last_bias.shape[0], setting['label_count']]) # pool 4 time
        plot_4d_weight(weight, bias)
    except: 
        print('Failed to reshape for last dense weight')
        plt.contourf(np.squeeze(weight)); plt.title('dense weight without reshape'); plt.show()
        plt.plot(bias,'b.--'); plt.title('dense bias');  plt.show()

    
elif 'res' in model_type or 'oracle' in model_type:
    weight = None
    bias = None
    for i, tf_variable in enumerate(var_without_L2M):
        var_name = tf_variable.name
        layer_name = var_name.split('/')[0]
        variable = sess.run([tf_variable])[0]
        print('{}:{}'.format(var_name, variable.shape))
        
        if (type(weight) == type(None)) or (type(bias) == type(None)):
            if 'bias' in var_name : bias = variable
            elif 'kernel' in var_name: weight = variable
            else: print("Unexpectable", variable); assert(0)
        if (type(weight) != type(None)) and (type(bias) != type(None)):
            if len(weight.shape) == 2: pass
            else: 
                plot_4d_weight(weight, bias)
                last_bias = bias; weight = None; bias = None
    try: 
        try :
            weight = weight.reshape([25, 20, last_bias.shape[0], setting['label_count']]) # pool 2 time
        except:
            try:
                weight = weight.reshape([13, 10, last_bias.shape[0], setting['label_count']])  # pool 3 time
            except:
                weight = weight.reshape([7, 5, last_bias.shape[0], setting['label_count']])  # pool 4 time
        plot_4d_weight(weight, bias)
    except:
        print('Failed to reshape for last dense weight')
        plt.contourf(np.squeeze(weight)); plt.title('dense weight without reshape'); plt.show()
        plt.plot(bias,'b.--'); plt.title('dense bias');  plt.show()

else: print('No such model_architecture: {}'.format(model_type))
#===================================================================================#




















