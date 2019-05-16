import numpy as np
from random import random, randint, gauss
import imageio
import matplotlib.pyplot as plt
import Tools as tp
from scipy.optimize import curve_fit

# Simulation Parameters
save = False

# Lattice Parameters
M = 64 # Dimension of sides
b1 = 0.16 # Inverse temperature of top heat bath
b2 = 0.15 # inverse temperature of bottom heat bath
hw = 1 # value of h-bar * omega in each oscillator

# Initialize system
n = np.full((M,M),int(0.5*0.5*(1/np.tanh(0.5*(b1+b2)))))
s = np.zeros((M,M))
b = np.full((M,M),0.5*(b1+b2))

# Set boundaries for heat baths temperatures
for i in range(M):
    b[0][i] = b1
    b[M-1][i] = b2

prior_n = n.copy()
prior_s = s.copy()
prior_b = b.copy()

def getP(x,y): # reutrns the probability factor for the chosen oscillator
    return [np.exp(-b[x][y]*hw*(n[x][y]+0.5+i)) for i in [-1,0,1]]

def getCoord(): # Uses Fouriers Law of heat conduction to find most probable coord
    T = 1/b.copy()
    dT = np.gradient(T)
    dT = dT-np.amin(dT)
    dT = dT/np.amax(dT)
    pBool = True
    randCoord = (0,0)
    while pBool:
        randCoord = (randint(0,M-1),randint(0,M-1))
        randNum1 = random()
        randNum2 = random()
        pBool = randNum1 > dT[0][randCoord] and randNum2 > dT[1][randCoord]
    return randCoord

def tick(time,gradT = False): # Runs the system one step in the future
    global prior_n,prior_s,prior_b
    a_list = []
    e_list = []
    for d in range(int(M**0.5)):
        if gradT:
            a = getCoord()
        else:
            a = (randint(0,M-1),randint(0,M-1)) # choose random oscillator
        a_list.append(a)
        p = getP(a[0],a[1])
        z = np.sum(p) # partition function for chosen oscillator
        if a[0] == 0 or a[0] == M-1: # Adjacent to heat bath chosen
            rand_var = random()
            if rand_var < p[0]/z and n[a] > 0: # The heat bath takes an energy level
                n[a]-=1
            elif rand_var > (p[0]/z)+(p[1]/z): # The heat bath gives an energy level
                n[a]+=1
        else:
            adjacent = tp.getAdjacent(a[0],a[1],periodicY=True,bounds=(0,M-1))
            rand_var1 = random()
            if rand_var1 > p[1]/z: # True if it goes up or down
                rand_var2 = random()
                adj_p = [getP(m[0],m[1]) for m in adjacent]
                adj_z = [np.sum(m) for m in adj_p]
                if rand_var1 > (p[0]/z)+(p[1]/z) and n[a]>0: # True if it goes down
                    lcl_norm = np.sum([adj_p[m][0]/adj_z[m] for m in range(len(adjacent))])
                    prob = 0
                    i = 0
                    for m in adj_p: # Choose which adjacent oscillator to give it to
                        prob+=((m[0]/adj_z[i])/lcl_norm)
                        if rand_var2 < prob:
                            e_list.append(adjacent[i])
                            n[adjacent[i]] +=1
                            n[a] -=1
                            break
                        else:
                            i+=1
                else: # It goes up an energy level
                    lcl_norm = np.sum([adj_p[m][2]/adj_z[m] for m in range(len(adjacent))])
                    prob = 0
                    i = 0
                    for m in adj_p: # Choose which adjacent oscillator to take from
                        prob+=((m[2]/adj_z[i])/lcl_norm)
                        if rand_var2 < prob and n[adjacent[i]] > 0:
                            e_list.append(adjacent[i])
                            n[adjacent[i]] -=1
                            n[a] +=1
                            break
                        else:
                            i+=1
    # Adjust Entropy for local area
    for a in a_list+e_list:
        t_area = [a]+tp.getAdjacent(a[0],a[1],periodicY=True,bounds=(0,M-1))
        q = np.sum([n[k] for k in t_area])
        N = len(t_area)
        omega = np.math.factorial(q+N-1)/(np.math.factorial(N-1)*np.math.factorial(q))
        s[a] = np.log(omega)
    # Adjust Beta for all affected
    ds = s-prior_s
    de = hw*(n-prior_n)
    for a in a_list+e_list:
        if a[0] != 0 and a[0] != M-1:
            if ds[a]!= 0 and de[a]!=0:
                b[a] = np.abs(ds[a]/de[a])
    # Overwrite prior values
    prior_s = s.copy()
    prior_n = n.copy()
    prior_b = b.copy()
        

save_n = [n.copy()]
save_s = [s.copy()]
save_b = [b.copy()]
def lin(x,A,B):
    return A*x+B

def steadyState():
    bMu = [np.average(sb) for sb in save_b[int(len(save_b)*0.75):]]
    t1,t2 = curve_fit(lin,range(int(len(save_b)*0.75),len(save_b)),bMu)
    return t1[0]<=0.01 and t1[0] >=-0.01

def run(nFrame, skip,gradT=False,ss=False):
    for i in range(nFrame):
        tick(i,gradT=gradT)
        if i % skip == 0:
            print(i/skip)
            save_n.append(n.copy())
            save_s.append(s.copy())
            save_b.append(b.copy())
            if ss:
                if steadyState():
                    break

resolution = (256,256)
    
def saveMov(): # saves to mp4
    global final_n, final_s, final_b
    final_n = []
    final_s = []
    final_b = []
    nMax = np.amax(save_n)
    nMin = np.amin(save_n)
    sMax = np.amax(save_s)
    sMin = np.amin(save_s)
    bMax = np.amax(save_b)
    bMin = np.amin(save_b)
    
    for i in range(len(save_n)):
        print(i+1)
        final_n.append(tp.saveFrame(save_n[i],nMax,nMin,resolution))
        final_s.append(tp.saveFrame(save_s[i],sMax,sMin,resolution))
        final_b.append(tp.saveFrame(save_b[i],bMax,bMin,resolution))
    # Write to mp4
    imageio.mimwrite('ESolid{}by{}numbers_{}b1_{}b2.mp4'.format(M,M,b1,b2),final_n,fps = 60)
    imageio.mimwrite('ESolid{}by{}entropy_{}b1_{}b2.mp4'.format(M,M,b1,b2),final_s,fps = 60)
    imageio.mimwrite('ESolid{}by{}beta_{}b1_{}b2.mp4'.format(M,M,b1,b2),final_b,fps = 60)

if save:
    saveMov()
if __name__ == '__main__':
    nFrame = 1000000 # How many steps it takes
    skip_frame =  1000 # How often to save data (in case memory overflow)
    run(nFrame,skip_frame)
    
