# python tf_exp_circuit.py
import cmath as cm, numpy as np, matplotlib.pyplot as plt, sys
f=np.logspace(1,5,1000);
R2=3300;
wH=2*np.pi*1/(1/1000000/100*(100+R2)+1/19894);
wL=2*np.pi*(159+3.94);
s=1j*2*np.pi*f; 
Numerator=(100+R2)/100*0.09*s
Denominator=(1+s/wH)*(s+wL)
H=Numerator/Denominator

plt.figure(1); plt.subplot(2,1,1); plt.semilogx(f,abs(H),'bo'); 
plt.title('magnitude of TF vs f');

ph_H=np.zeros(len(f), dtype='float'); 
for i in range(len(f)): ph_H[i]=cm.phase(H[i])*180/np.pi; 

plt.subplot(2,1,2); plt.semilogx(f,ph_H,'b'); 
plt.title('phase of TF vs f'); plt.show();
