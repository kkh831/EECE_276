import numpy as np, matplotlib.pyplot as plt, sys
f = open('inter_L.txt','r'); data_ecg_string = f.read();f.close();
data_ecg_string_list=data_ecg_string.split("\n")[:-1];
data_ecg_int_list = [int(float((val))) for val in data_ecg_string_list]
start_index = 105445;
end_index = 268605;
plt.figure(1);plt.plot(data_ecg_int_list,'b.');
plt.title('L');plt.show()
