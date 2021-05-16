import cmath as cm, numpy as np, matplotlib.pyplot as plt, sys
f=np.logspace(1,5,1000);
R2=3300;
wH=2*np.pi*1/(1/1000000/100*(100+R2)+1/19894);
wL=2*np.pi*(159+3.94);
s=1j*2*np.pi*f; 
Numerator=(100+R2)/100*0.09*s
Denominator=(1+s/wH)*(s+wL)
H=Numerator/Denominator

f2=np.array([200, 400, 1000, 4000])
h2=np.array([1606.9, 38431, 20664, 6910.7])
h2=h2/5000

plt.figure(1); plt.loglog(f,abs(H)); plt.loglog(f2, h2, 'ro'); 
plt.title('blue: calc of TF-f   red: meas Iout/Vin');
plt.xlim(100,20000);
plt.show();
