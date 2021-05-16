# Usage: python freeze.py (checkpoint_path)
# Example: python freeze.py result_dir/ckpt/ep48_vl~~~.ckpt
import os, sys, pickle, glob, shutil
import tensorflow as tf
from model import create_model, load_variables_from_checkpoint
from tensorflow.python.framework import graph_util
from tensorflow.python.ops import gen_audio_ops as audio_ops
#===============================================================================================#

def create_inference_graph(setting):
  wav_data_placeholder = tf.compat.v1.placeholder(tf.string, [], name='wav_data')
  decoded_sample_data = tf.audio.decode_wav(wav_data_placeholder, desired_channels=1, desired_samples=setting['desired_samples'], name='decoded_sample_data')
  input_ = tf.reshape(decoded_sample_data.audio,[setting['desired_samples'],1])
  spectrogram = audio_ops.audio_spectrogram(input_, window_size=setting['window_size_samples'], stride=setting['window_stride_samples'], magnitude_squared=True)

  if setting['preprocess'] == "mfcc":
    tf_out = audio_ops.mfcc(spectrogram, setting['sample_rate'], dct_coefficient_count=setting['fingerprint_width'])
  elif setting['preprocess'] == "mel":
    L2M_mtx = tf.constant(setting['L2M'], shape = setting['L2M'].shape, dtype=tf.float32, name="L2M_mtx")
    tf_out = tf.math.log(tf.tensordot(spectrogram, L2M_mtx, 1)+1e-6)
  
  print('tf_out:', setting['preprocess'],':',tf_out)
  fingerprint_size = setting['fingerprint_size']
  reshaped_input = tf.reshape(tf_out, [-1, fingerprint_size])
  logits, _ = create_model(reshaped_input, setting, setting['model_architecture'], is_training=False)
  softmax = tf.nn.softmax(logits, name='labels_softmax')
  return input_, softmax

def save_graph_def(file_name, frozen_graph_def):
  tf.io.write_graph(frozen_graph_def, os.path.dirname(file_name), os.path.basename(file_name), as_text=False)
  tf.compat.v1.logging.info('Saved frozen graph to %s', file_name)

def save_saved_model(file_name, sess, input_tensor, output_tensor):
  # Store the frozen graph as a SavedModel for v2 compatibility.
  builder = tf.compat.v1.saved_model.builder.SavedModelBuilder(file_name)
  tensor_info_inputs = {'input': tf.compat.v1.saved_model.utils.build_tensor_info(input_tensor)}
  tensor_info_outputs = {'output': tf.compat.v1.saved_model.utils.build_tensor_info(output_tensor)}
  signature = tf.compat.v1.saved_model.signature_def_utils.build_signature_def(
          inputs=tensor_info_inputs, outputs=tensor_info_outputs, method_name=tf.compat.v1.saved_model.signature_constants.PREDICT_METHOD_NAME)
  builder.add_meta_graph_and_variables(sess, [tf.compat.v1.saved_model.tag_constants.SERVING],
      signature_def_map ={tf.compat.v1.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:signature})
  builder.save()

#===============================================================================================#
try :
    # [Freeze model]
    # Create the model and load its weights.
    ckpt_dir =  '/'.join(os.path.split(sys.argv[1])[:-1])
    setting_path = os.path.abspath(glob.glob(os.path.join(ckpt_dir,'..','setting.bin'))[0])
    print('setting_path:',setting_path)
    with open(setting_path,'rb') as f:
        setting = pickle.load(f)

    sess = tf.compat.v1.InteractiveSession()
    input_tensor, output_tensor = create_inference_graph(setting)
    load_variables_from_checkpoint(sess, sys.argv[1])
    print('--> input_tensor:',input_tensor)
    print('--> output_tensor:',output_tensor)

    # Turn all the variables into inline constants inside the graph and save it.
    tflite_dir = os.path.join(os.path.dirname(setting_path),'tflite')
    os.makedirs(tflite_dir, exist_ok=True)
    frozen_graph_def = graph_util.convert_variables_to_constants(sess, sess.graph_def, ['labels_softmax'])
    save_graph_def(os.path.join(tflite_dir,'graph'), frozen_graph_def)
    frozen_dir = os.path.join(tflite_dir,'frozen_model')
    if os.path.isdir(frozen_dir): shutil.rmtree(frozen_dir)
    save_saved_model(frozen_dir, sess, input_tensor, output_tensor)

    #----------------------------------------------------------------------------------#

    converter = tf.lite.TFLiteConverter.from_saved_model(frozen_dir)
    converter.allow_custom_ops = True
    tflite_model = converter.convert()
    tflite_save_path = os.path.join(tflite_dir,"model.tflite")
    open(tflite_save_path, "wb").write(tflite_model)
    print('Done. {} -> {}'.format(sys.argv[1], tflite_save_path))
except Exception as ex:
    print('Error: ', ex)
    print('Conversion failed. Read error message. Check your file & model')
    












