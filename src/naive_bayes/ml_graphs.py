import numpy as np
import reliability_curve_stuff as rcs
from sklearn import preprocessing

csv = np.genfromtxt('predictions.csv', delimiter=',')
true_lab = csv[:,0]
prob_lab = csv[:,1]
inv_prob_lab = csv[:,2]

a = 0
b = 0
d = 0
c = 0
for tru, prob in zip(true_lab, prob_lab):
    if tru == 1 and prob == 1:
        a += 1
    elif tru == 1 and prob == 0:
        c += 1
    elif tru == 0 and prob == 1:
        b += 1
    else:
        d += 1

pod = a/(a+c)
pofd = b/(b+d)
sr = a/(a+b)
acc = (a+d)/(a+b+c+d)
csi = a/(a+b+c)

print('POD: {0}%'.format(pod))
print('POFD: {0}%'.format(pofd))
print('SR: {0}%'.format(sr))
print('ACC: {0}%'.format(acc))
print('CSI: {0}%'.format(csi))

print(a)
print(b)
print(c)
print(d)