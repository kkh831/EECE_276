# quality factor calculator
# python cal_qual_factor.py data_filename.txt
# final3_out.txt: Rs Ls Rp Lp Cp
import cmath as cm, numpy as np, matplotlib.pyplot as plt, sys

def read_text_file(filename):
    fp = open(filename,'r')
    a_list = list()
    while True:
        a = fp.readline().split()
        if a is None or len(a)==0: break
        else: a_list.append([float(b) for b in a])
    np_array = np.array(a_list)
    return np_array

param = read_text_file(sys.argv[1]);

series = pow(param[0, 1]/220e-6,0.5)/param[0, 0]
parallel = param[0, 2]*pow(param[0, 4]/param[0, 3],0.5)

print('quality factor of series part: ', series);
print('quality factor of parallel part: ', parallel);
