# python tf_speaker.py
import cmath as cm, numpy as np, matplotlib.pyplot as plt, sys
f=np.logspace(1,5,1000);
Rs=8.579999999999922; Ls= 0.000299400000000235; Cs=220e-6
Rp= 25.700000000000607; Lp= 0.02009339999994006; Cp= 0.00018679999999999966
s=1j*2*np.pi*f; 
Numerator=s*s*s*Rp*Lp*Cs*Cp
Denominator=(s*s*Ls*Cs+s*Cs*Rs+1)*(s*s*Lp*Cp*Rp+s*Lp+Rp)+s*s*Lp*Cs*Rp
H=Numerator/Denominator

plt.figure(1); plt.subplot(2,1,1); plt.loglog(f,abs(H),'bo'); 
plt.title('magnitude of TF vs f');

ph_H=np.zeros(len(f), dtype='float'); 
for i in range(len(f)): ph_H[i]=cm.phase(H[i])*180/np.pi; 

plt.subplot(2,1,2); plt.semilogx(f,ph_H,'b'); 
plt.title('phase of TF vs f'); plt.show();
