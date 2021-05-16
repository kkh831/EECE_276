# Usage: python train.py dataset_dir model_type result_dir max_epoch save_epoch_interval
# example: python train.py residual_conv ../dataset result3_residual_conv_1008 50 10
import os, sys, tqdm, pickle, random, glob
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tf1
from model import AudioProcessor, prepare_model_settings, create_model, prepare_words_list
from tensorflow.python.platform import gfile
#======================================================================================#
# 1. Variables
keywords = ['ALEXA','BIXBY','GOOGLE','JINIYA','KLOVA']
data_dir = sys.argv[2]; ckpt_dir = sys.argv[3]; model_type = sys.argv[1]# conv res my_own
max_epoch=int(sys.argv[4]);save_epoch_interval=int(sys.argv[5]);batch_size=8;init_lr=3e-4
#how_many = float(sys.argv[6])
#======================================================================================#
# 2. Build Preprocessor and Model
load_all_data_mode = True
allword = prepare_words_list(keywords)
setting_path = os.path.abspath(glob.glob(os.path.join(data_dir,'data_setting.bin'))[0])
with open(setting_path,'rb') as f:
    setting = pickle.load(f)
setting['model_architecture'] = model_type
setting['label_count'] = len(keywords)
setting['allwords'] = keywords
ap = AudioProcessor(data_dir, 14, 14, keywords, load_all_data_mode)
tf_in_size = setting['fingerprint_size']
tf_in = tf1.placeholder(tf.float32, [None, tf_in_size], name='fingerprint_input')
logits, dropout_rate = create_model(tf_in, setting, model_type, is_training=True)
tf_label = tf1.placeholder(tf.int64, [None], name='groundtruth_input')
tf_label_onehot = tf.one_hot(tf_label, depth=len(allword), dtype=tf.float32)
loss = tf1.losses.softmax_cross_entropy(onehot_labels=tf_label_onehot, logits=logits, label_smoothing=0.1)
tf_lr = tf1.placeholder(tf.float32, [], name='tf_lr')
optimizer = tf1.train.GradientDescentOptimizer(tf_lr).minimize(loss)
predicted_indices = tf.argmax(input=logits, axis=1)
correct_prediction = tf.equal(predicted_indices, tf_label)
confusion = tf.math.confusion_matrix(labels=tf_label, predictions=predicted_indices, num_classes=len(allword))
accuracy = tf.reduce_mean(input_tensor=tf.cast(correct_prediction, tf.float32))
#======================================================================================#
# 3. Train
sess = tf1.InteractiveSession()
sess.run(tf1.global_variables_initializer())
tf.io.write_graph(sess.graph_def, ckpt_dir, 'model.pbtxt')
tmp_words_list = list()
for word in ap.words_list:
    tmp_words_list.append(word)
        
with gfile.GFile(os.path.join(ckpt_dir, 'labels.txt'), 'w') as f:
    f.write('\n'.join(tmp_words_list))
with open(os.path.join(ckpt_dir,'setting.bin'),'wb') as f:
    pickle.dump(setting, f, protocol=pickle.HIGHEST_PROTOCOL)
all_variables = tf1.global_variables()
var_without_L2M = list()
for i, variable in enumerate(all_variables):
    if 'L2M_mtx' in str(variable): continue
    var_without_L2M.append(variable)
    print(variable)
saver = tf1.train.Saver(var_without_L2M, max_to_keep=100)

lr_val = init_lr
best_loss = float("inf")
patience = 0
print('\nTraining start')
for epoch in range(max_epoch):
    train_acc = 0; train_loss = 0
    if(epoch==0):
        set_size = ap.set_size('validation')
        val_acc = 0; val_loss = 0; val_confusion = np.zeros([len(allword),len(allword)])
        for i in tqdm.tqdm(range(0, set_size, batch_size)):
            validX, validY = ap.get_data_npy(batch_size, i, setting, 'validation', sess, load_all_data_mode)
            acc_, loss_, confusion_ = sess.run([accuracy, loss, confusion], 
                            feed_dict={tf_in: validX, tf_label: validY, dropout_rate: 0.0 })
            val_acc += (acc_ * min(batch_size, set_size - i)) / set_size
            val_loss += (loss_ * min(batch_size, set_size - i)) / set_size
            val_confusion += confusion_
    
        print('[Epoch Initial] lr(%.3e)| (T,V) | acc(%.1f%%, %.1f%%) | loss(%f, %f) |' %
                    (lr_val, train_acc * 100, val_acc * 100, train_loss, val_loss))
        print('Confusion Matrix for Validation: \n',val_confusion,'\n')
        checkpoint_path = os.path.join(ckpt_dir, 'ckpt','ep%d_vl%f.ckpt'%(epoch+1, val_loss))
        saver.save(sess, checkpoint_path)
        
    set_size = ap.set_size('training')
    #set_size = int(set_size * how_many / 100.0)
    for i in tqdm.tqdm(range(0, set_size, batch_size)):
        random.shuffle(ap.data_index['training'])
        trainX, trainY = ap.get_data_npy(batch_size, i, setting, 'training', sess, load_all_data_mode) 
        acc_, loss_, _ = sess.run([accuracy, loss, optimizer], 
                        feed_dict={tf_in: trainX, tf_label: trainY, tf_lr: lr_val, dropout_rate: 0.5})
        train_acc += (acc_ * min(batch_size, set_size - i)) / set_size
        train_loss += (loss_ * min(batch_size, set_size - i)) / set_size
    
    if epoch % save_epoch_interval == (save_epoch_interval-1): 
        set_size = ap.set_size('validation')
        val_acc = 0; val_loss = 0; val_confusion = np.zeros([len(allword),len(allword)])
        for i in tqdm.tqdm(range(0, set_size, batch_size)):
            validX, validY = ap.get_data_npy(batch_size, i, setting, 'validation', sess, load_all_data_mode)
            acc_, loss_, confusion_ = sess.run([accuracy, loss, confusion], 
                            feed_dict={tf_in: validX, tf_label: validY, dropout_rate: 0.0 })
            val_acc += (acc_ * min(batch_size, set_size - i)) / set_size
            val_loss += (loss_ * min(batch_size, set_size - i)) / set_size
            val_confusion += confusion_
    
        print('[Epoch #%d] lr(%.3e)| (T,V) | acc(%.1f%%, %.1f%%) | loss(%f, %f) |' %
                    (epoch+1, lr_val, train_acc * 100, val_acc * 100, train_loss, val_loss))
        print('Confusion Matrix for Validation: \n',val_confusion,'\n')

        checkpoint_path = os.path.join(ckpt_dir, 'ckpt','ep%d_vl%f.ckpt'%(epoch+1, val_loss))
        saver.save(sess, checkpoint_path)
        if val_loss >= best_loss : lr_val = max(1e-7, lr_val * 0.5);
        else: best_loss = val_loss

#======================================================================================#
print('============================================================================')
set_size = ap.set_size('testing')
tt_acc = 0; tt_loss = 0; tt_confusion = np.zeros([len(allword),len(allword)])
for i in tqdm.tqdm(range(0, set_size, batch_size)):
    testX, testY = ap.get_data_npy(batch_size, i, setting, 'testing', sess, load_all_data_mode)
    acc_, loss_, confusion_ = sess.run([accuracy, loss, confusion], 
                    feed_dict={tf_in: testX, tf_label: testY, dropout_rate: 0.0 })
    tt_acc += (acc_ * min(batch_size, set_size - i)) / set_size
    tt_loss += (loss_ * min(batch_size, set_size - i)) / set_size
    tt_confusion += confusion_

print('Test acc(%.1f%%) | loss(%f) |' % (tt_acc * 100, tt_loss))
print('Confusion Matrix for Testing: \n',tt_confusion,'\n')
print('============================================================================')


