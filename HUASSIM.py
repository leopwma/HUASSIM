#********************************************************************************
#
#   Copyright (C) 2019 Culham Centre for Fusion Energy,
#   United Kingdom Atomic Energy Authority, Oxfordshire OX14 3DB, UK
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
#*******************************************************************************
#
#   Program: HUASSIM: Huang diffuse X-ray scattering simulation based on elastic dipole tensor
#   File: HUASSIM.py
#   Version: 1.0
#   Date:    May 2019
#   Author:  Pui-Wai (Leo) MA
#   Contact: Leo.Ma@ukaea.uk
#   Address: Culham Centre for Fusion Energy, OX14 3DB, United Kingdom
#
#*******************************************************************************/



from numba import jit
from atomic_form_factor import atomic_form_factor
import numpy as np
import numpy.linalg as LA
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors

######################################################################################################
#
# initializing variables
#
######################################################################################################

C11 = 0e0
C12 = 0e0
C44 = 0e0
q_range = 0.01 # default
grid = 100 # default
h = np.zeros(3)
p = np.zeros(3)
filename = "temp"
tag = ""
P11 = 0e0
P22 = 0e0
P33 = 0e0
P12 = 0e0
P23 = 0e0
P31 = 0e0
element = "None"
a_lattice = 0e0
cutoff = 1e8 # default

######################################################################################################
#
# read variables from "input_data"
#
######################################################################################################

f2 = open("input_data","r")
for line in f2:
    x = line.split()
    if line[0] == "#" or line.strip() == "":
       pass
    elif x[0] == "C11":
        C11 = float(x[1])
    elif x[0] == "C12":
        C12 = float(x[1])
    elif x[0] == "C44":
        C44 = float(x[1])
    elif x[0] == "q_range":
        q_range = float(x[1])
    elif x[0] == "grid":
        grid = int(float(x[1]) + 1e-5)
    elif x[0] == "p":
        p[0] = float(x[1])
        p[1] = float(x[2])
        p[2] = float(x[3])
    elif x[0] == "h":
        h[0] = float(x[1])
        h[1] = float(x[2])
        h[2] = float(x[3])
    elif x[0] == "filename":
        filename = x[1]
    elif x[0] == "tag":
        tag = x[1]
    elif x[0] == "P11":
        P11 = float(x[1])
    elif x[0] == "P22":
        P22 = float(x[1])
    elif x[0] == "P33":
        P33 = float(x[1])
    elif x[0] == "P12":
        P12 = float(x[1])
    elif x[0] == "P23":
        P23 = float(x[1])
    elif x[0] == "P31":
        P31 = float(x[1])
    elif x[0] == "element":
        element = x[1]
    elif x[0] == "a_lattice":
        a_lattice = float(x[1])
    elif x[0] == "cutoff":
        cutoff = float(x[1])
    #print (x)

f2.close()


######################################################################################################
#
# Calculation method according to:
#
# P. H. Dederichs, J. Phys. F: Metal Phys. Vol 3 Feb 1973, page 471
#
######################################################################################################


@jit(nopython=True)
def delta(i,j):
    if (i == j):
        return 1.0
    else:
        return 0.0

@jit(nopython=True)
def delta4(i,j,k,l):
    if i == j and i == k and  i == l:
        return 1.0
    else:
        return 0.0

@jit(nopython=True)
def Cijkl(C11, C12, C44):
    C = np.zeros((3,3,3,3))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    C[i,j,k,l] = C12*delta(i,j)*delta(k,l) \
                               + C44*(delta(i,k)*delta(j,l)+delta(i,l)*delta(j,k)) \
                               + (C11-C12-2e0*C44)*delta4(i,j,k,l)
    return C

@jit(nopython=True)
def g_tilda(kappa, C):
    gij = np.zeros((3,3))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    if C[i,k,j,l] > 1e-5:
                        gij[i,j] += C[i,k,j,l]*kappa[k]*kappa[l]

    gij = LA.inv(gij)
    return gij



@jit(nopython=True)
def S_Huang(q, h, C, Pi):
     
    q_unit = q/LA.norm(q)
    h_unit = h/LA.norm(h)

    gij = g_tilda(q_unit,C)

    T = np.zeros((3,3))
    for m in range(3): 
        for n in range(3):
            for l in range(3):
                T[m,n] += h_unit[l]*gij[l,m]*q_unit[n]

    gamma = np.zeros(3)
 
    gamma[0] = ((T[0,0] + T[1,1] + T[2,2])**2)/3e0
    gamma[1] = ((T[0,0] - T[1,1])**2 + (T[1,1] - T[2,2])**2 + (T[2,2] - T[0,0])**2)/3e0
    gamma[2] = ((T[0,1] + T[1,0])**2 + (T[1,2] + T[2,1])**2 + (T[0,2] + T[2,0])**2)/2e0

    return  ((LA.norm(h)/LA.norm(q))**2)*np.dot(gamma,Pi)


@jit(nopython=True)
def gcd(x):
    y = np.fabs(x)
    a = y[0]
    for b in y[1:]:
        while b > 0:
            a, b = b, a % b
    return a

######################################################################################################

x_vec = h
y_vec = np.cross(p,h)

gcd_y_vec = gcd(y_vec)
y_vec /= gcd_y_vec

x_proj = x_vec/LA.norm(x_vec)
y_proj = y_vec/LA.norm(y_vec)

Pi = np.ones(3)
Pi[0] = 1e0/3e0*((P11 + P22 + P33)**2)
Pi[1] = 1e0/6e0*((P11 - P22)**2 + (P22 - P33)**2 + (P33 - P11)**2)
Pi[2] = 2e0/3e0*(P12**2 + P23**2 + P31**2)

eVperAcubic_to_GPa = 160.21766208
C11 /= eVperAcubic_to_GPa
C12 /= eVperAcubic_to_GPa
C44 /= eVperAcubic_to_GPa
C = np.zeros((3,3,3,3))

aff = atomic_form_factor(LA.norm(h)/a_lattice*2e0*np.pi,element)

start = -int(grid/2)
end   =  int(grid/2)

grid_sq = grid*grid

X_hist = np.zeros(grid_sq)
Y_hist = np.zeros(grid_sq)
Z_hist = np.zeros(grid_sq)
N_count = 0

X_contour = np.zeros((grid,grid))
Y_contour = np.zeros((grid,grid))
Z_contour = np.zeros((grid,grid))

C = Cijkl(C11, C12, C44)

for i in range(start,end):
    dx = (float(i) + 0.5)/float(grid/2)*q_range
    m = i + int(grid/2)
    for j in range(start,end):
        dy = (float(j) + 0.5)/float(grid/2)*q_range
        q = dx*x_proj + dy*y_proj
            
        SS = S_Huang(q, h, C, Pi)
        cell_volume = a_lattice**3
        SS *= aff/(cell_volume**2)

        if SS > cutoff:
            SS = cutoff

        X_hist[N_count] = dx
        Y_hist[N_count] = dy
        Z_hist[N_count] = SS
        N_count += 1

        n = j + int(grid/2)
        X_contour[m,n] = dx
        Y_contour[m,n] = dy
        Z_contour[m,n] = SS



######################################################################################################
#
# Output figures: 2D histogram and 2D contourf plots
#
######################################################################################################

for i in range(3):
    if x_vec[i] > 0e0:
        x_vec[i] += 1e-5
    else:
        x_vec[i] -= 1e-5
    if y_vec[i] > 0e0:
        y_vec[i] += 1e-5
    else:
        y_vec[i] -= 1e-5
    if p[i] > 0e0:
        p[i] += 1e-5
    else:
        p[i] -= 1e-5


h_plane     = "(" + str(int(x_vec[0])) + " " + str(int(x_vec[1])) + " " + str(int(x_vec[2])) + ")"
p_plane   = "(" + str(int(p[0])) + " " + str(int(p[1])) + " " + str(int(p[2])) + ")"

x_vec = str(x_vec.astype(int))
y_vec = str(y_vec.astype(int))


title = tag + "   h=" + x_vec + "   p=" + p_plane

######################################################################################################

fig1, ax1 = plt.subplots()
#h_ax1, xedges_ax1, yedges_ax1, image_ax1 = ax1.hist2d(X_hist, Y_hist, bins=grid,  weights=Z_hist, cmap='gray', norm=colors.LogNorm())
h_ax1, xedges_ax1, yedges_ax1, image_ax1 = ax1.hist2d(X_hist, Y_hist, bins=grid,  weights=Z_hist, cmap='gray')

ax1.set_xlabel(str(round(q_range,5)) + " of the " + x_vec + " unit vector")
ax1.set_ylabel(str(round(q_range,5)) + " of the " + y_vec + " unit vector")

locs = ax1.get_xticks()
labels = ax1.get_xticklabels()
#print (locs, labels)
for i in range(len(locs)):
    labels[i] = str(round(locs[i]/q_range,2))
ax1.set_xticklabels(labels)

locs = ax1.get_yticks()
labels = ax1.get_yticklabels()
#print (locs, labels)
for i in range(len(locs)):
    labels[i] = str(round(locs[i]/q_range,2))
ax1.set_yticklabels(labels)

ax1.set_title(title)

fig1.colorbar(image_ax1)
fig1.savefig(filename + "_histogram.png", dpi=600)

######################################################################################################

fig2, ax2 = plt.subplots()
#image_ax2 = ax2.contourf(X_contour, Y_contour, Z_contour, cmap='gray', norm=colors.LogNorm())
image_ax2 = ax2.contourf(X_contour, Y_contour, Z_contour, cmap='gray')

ax2.set_xlabel(str(round(q_range,5)) + " of the " + x_vec + " unit vector")
ax2.set_ylabel(str(round(q_range,5)) + " of the " + y_vec + " unit vector")

locs = ax2.get_xticks()
labels = ax2.get_xticklabels()
#print (locs, labels)
for i in range(len(locs)):
    labels[i] = str(round(locs[i]/q_range,2))
ax2.set_xticklabels(labels)

locs = ax2.get_yticks()
labels = ax2.get_yticklabels()
#print (locs, labels)
for i in range(len(locs)):
    labels[i] = str(round(locs[i]/q_range,2))
ax2.set_yticklabels(labels)

ax2.set_title(title)

fig2.colorbar(image_ax2)
fig2.savefig(filename + "_contour.png", dpi=600)

######################################################################################################
