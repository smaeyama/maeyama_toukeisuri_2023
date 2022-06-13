#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import matplotlib.pyplot as plt

A1 = np.loadtxt("./data/A1.txt")
A2 = np.loadtxt("./data/A2.txt")
Exi = np.loadtxt("./data/Exi.txt")
omega = np.loadtxt("./data/omega.txt")
gamma1 = np.loadtxt("./data/gamma1.txt")
rr0 = np.loadtxt("./data/rr0.txt")

p=2
m=5

plt.rcParams["font.size"]=7
plt.rcParams["axes.titlesize"]=7

from matplotlib import colors
cmap = plt.cm.Greys
bounds = np.linspace(-1.05, 1.05, 22)
norm = colors.BoundaryNorm(bounds, cmap.N)

fig=plt.figure(figsize=(5.3,3.5),dpi=150)
ax=fig.add_subplot(231)
quad = ax.pcolormesh(np.arange(m),np.arange(m),A1,
                     cmap=cmap,norm=norm,shading="auto")
ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
ax.set_title(r"(a) SVAR $A_{ij}(1)$")
# ax.set_title(r"$A_1[i,j]$")
secax = ax.secondary_xaxis('top')
secax.set_xlabel(r'Index $j$')
secax.set_xlim(0-0.5,m-0.5)
secax.set_xticks(np.arange(m))
ax.set_xticks([])
ax.set_ylim(m-0.5,0-0.5)
ax.set_yticks(np.arange(m-1,-1,-1))
ax.set_ylabel(r"Index $i$")
fig.colorbar(quad,ticks=np.linspace(-1, 1, 11),shrink=0.68)
ax.set_aspect("equal")


ax=fig.add_subplot(232)
quad = ax.pcolormesh(np.arange(m),np.arange(m),A2,
                     cmap=cmap,norm=norm,shading="auto")
ax.set_ylim(m-0.5,0-0.5)
ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
ax.set_title(r"(b) SVAR $A_{ij}(2)$")
# ax.set_title(r"$A_2[i,j]$")
secax = ax.secondary_xaxis('top')
secax.set_xlabel(r'Index $j$')
secax.set_xlim(0-0.5,m-0.5)
secax.set_xticks(np.arange(m))
ax.set_xticks([])
ax.set_ylim(m-0.5,0-0.5)
ax.set_yticks(np.arange(m-1,-1,-1))
ax.set_ylabel(r"Index $i$")
fig.colorbar(quad,ticks=np.linspace(-1, 1, 11),shrink=0.68)
ax.set_aspect("equal")


ax=fig.add_subplot(234)
quad = ax.pcolormesh(np.arange(m),np.arange(m),omega,
                     cmap=cmap,norm=norm,shading="auto")
ax.set_ylim(m-0.5,0-0.5)
ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
ax.set_title(r"(d) Markov $\Omega_{ij}$")
secax = ax.secondary_xaxis('top')
secax.set_xlabel(r'Index $j$')
secax.set_xlim(0-0.5,m-0.5)
secax.set_xticks(np.arange(m))
ax.set_xticks([])
ax.set_ylim(m-0.5,0-0.5)
ax.set_yticks(np.arange(m-1,-1,-1))
ax.set_ylabel(r"Index $i$")
fig.colorbar(quad,ticks=np.linspace(-1, 1, 11),shrink=0.68)
ax.set_aspect("equal")

ax=fig.add_subplot(235)
quad = ax.pcolormesh(np.arange(m),np.arange(m),-gamma1,
                     cmap=cmap,norm=norm,shading="auto")
ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
ax.set_title(r"(e)Memory $-\Gamma_{ij}(1)$")
secax = ax.secondary_xaxis('top')
secax.set_xlabel(r'Index $j$')
secax.set_xlim(0-0.5,m-0.5)
secax.set_xticks(np.arange(m))
ax.set_xticks([])
ax.set_ylim(m-0.5,0-0.5)
ax.set_yticks(np.arange(m-1,-1,-1))
ax.set_ylabel(r"Index $i$")
fig.colorbar(quad,ticks=np.linspace(-1, 1, 11),shrink=0.68)
ax.set_aspect("equal")


bounds = np.linspace(-0.16, 0.16, 22)
norm = colors.BoundaryNorm(bounds, cmap.N)

ax=fig.add_subplot(233)
quad = ax.pcolormesh(np.arange(m),np.arange(m),Exi,
                     cmap=cmap,norm=norm,shading="auto")
ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
ax.set_title(r"(c) SVAR $\mathrm{Cov}=\langle \xi_i \xi_j \rangle$")
secax = ax.secondary_xaxis('top')
secax.set_xlabel(r'Index $j$')
secax.set_xlim(0-0.5,m-0.5)
secax.set_xticks(np.arange(m))
ax.set_xticks([])
ax.set_ylim(m-0.5,0-0.5)
ax.set_yticks(np.arange(m-1,-1,-1))
ax.set_ylabel(r"Index $i$")
fig.colorbar(quad,ticks=np.linspace(-0.15, 0.15, 7),shrink=0.68)
ax.set_aspect("equal")

ax=fig.add_subplot(236)
quad = ax.pcolormesh(np.arange(m),np.arange(m),rr0,
                     cmap=cmap,norm=norm,shading="auto")
ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
ax.set_title(r"(f) Uncorrelated $\langle r_i r_j \rangle$")
secax = ax.secondary_xaxis('top')
secax.set_xlabel(r'Index $j$')
secax.set_xlim(0-0.5,m-0.5)
secax.set_xticks(np.arange(m))
ax.set_xticks([])
ax.set_ylim(m-0.5,0-0.5)
ax.set_yticks(np.arange(m-1,-1,-1))
ax.set_ylabel(r"Index $i$")
fig.colorbar(quad,ticks=np.linspace(-0.15, 0.15, 7),shrink=0.68)
ax.set_aspect("equal")

fig.tight_layout()
plt.savefig("fig_svar_matrix.pdf")
plt.show()


# In[ ]:




