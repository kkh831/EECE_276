import numpy as np, matplotlib.pyplot as plt, sys

f = open('average_cont.txt', 'r'); data_ecg_string = f.read(); f.close();
data_ecg_string_list=data_ecg_string.split("\n")[:-1]; 
data_ecg_int_list = [int(val) for val in data_ecg_string_list]

plt.figure(1);plt.plot(data_ecg_int_list,'b.');
plt.title('diff_cut');plt.show()
