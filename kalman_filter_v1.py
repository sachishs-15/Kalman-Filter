import numpy as np
import matplotlib.pyplot as plt
import math

#function to convert a list of strings to a list of floats
def toDec(cd):
    ls = []
    for i in cd:
        ls.append(float(i))
    return ls

#function to multiply three matrices
def mult(a, b, c):
    return (a @ b) @ c

#opening the kalmann.txt file
file = open("Files/kalmann.txt", "r")

#storing the initial coordinates of the robot
initial = list(file.readline().strip().split(' , '))

initial = toDec(initial)
initial.append(0)
initial.append(0)

#storing the measured data input from the file
values = []
while True:
    line = file.readline()

    if not line:
        break

    else:
        values.append(list(line.strip().split(' , ')))

#variables to store the sum of all x and y coordinates
sumx = 0
sumy = 0
sumvelx = 0
sumvely = 0
n = len(values)

for line in values:
    coods = toDec(line)

    sumx = sumx + coods[0]
    sumy = sumy + coods[1]
    sumvelx += coods[2]
    sumvely += coods[3]

#variablels to store mean
meanx = sumx/n
meany = sumy/n
meanvelx = sumvelx/n
meanvely = sumvely/n

#variables to store variance
varx = 0
vary = 0
varvelx = 0
varvely = 0
for line in values:
    coods = toDec(line)

    varx += (coods[0] - meanx)**2
    vary += (coods[1] - meany)**2
    varvelx += (coods[2] - meanvelx)**2
    varvely += (coods[3] - meanvely)**2


varx = varx/n
vary = vary/n
varvelx = varvelx/n
varvely = varvely/n

#measurement noisecovariance matrix
R = [[varx, 0], [0, vary]] 

#process noise covariance matrix
Q = np.array([[varx, 0, math.sqrt(varx)*math.sqrt(varvelx),0],
               [0, vary, 0,  math.sqrt(vary)*math.sqrt(varvely)],
               [math.sqrt(varx)*math.sqrt(varvelx), 0, varvelx, 0],
               [0, math.sqrt(vary)*math.sqrt(varvely), 0, varvely]])


t = 1 #denoting the time difference between two states or measurements
A = [[1, 0, t, 0], [0, 1, 0, t], [0, 0, 1, 0], [0, 0, 0, 1]]
A = np.array(A)

H = [[1,0,0,0], [0,1,0,0]] #transformation matrix
H = np.array(H)

x = np.array([[initial[0]], [initial[1]], [initial[2]], [initial[3]]]) #initial state

P = Q

I = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]] #identity matrix

xcoods = []
xmeas = []
ymeas = []
ycoods = []

xcoods.append(initial[0])
ycoods.append(initial[1])
xmeas.append(initial[0])
ymeas.append(initial[1])

for i in range(len(values)):

    meas = toDec(values[i])
    z = [[meas[0]], [meas[1]]]
    z = np.array(z) #measured values

    xd = x   #x(k-1)
    
    Pd = P

    xmeas.append(meas[0])
    ymeas.append(meas[1])

    #PREDICT STEP

    xt = A @ xd  #x(k)-
    Pt = mult(A, Pd, A.transpose()) + Q #p(k) -

    #UPDATE STEP

    tt = np.linalg.inv(R + mult(H, Pd, H.transpose()))
    kgain = mult(Pd, H.transpose(), tt) #Kalman Gain

    x = [[xt[0][0]], [xt[1][0]], [xt[2][0]], [xt[3][0]]] + kgain @ (z - H @ xt)
    x[2][0] = meas[2]
    x[3][0] = meas[3]
    
    P = (I - kgain@H) @ Pt

    xcoods.append(x[0][0])
    ycoods.append(x[1][0])

plt.plot(xmeas, ymeas, color = 'b', label = 'Measured Trajectory')
plt.xlabel('X')
plt.ylabel('Y')
plt.title("Path before Kalman Filtering")
plt.show()


plt.plot(xcoods, ycoods, color = 'r', label = 'Most Probable Trajectory')
plt.title("Path after Kalman Filtering")
plt.xlabel('X')
plt.ylabel('Y')
plt.show()


plt.plot(xcoods, ycoods, color = 'r', label = 'Most Probable Trajectory')
plt.plot(xmeas, ymeas, color = 'b', label = 'Measured Trajectory')
plt.xlabel('X')
plt.title("Comparison between the two paths")
plt.ylabel('Y')
plt.legend()
plt.show()