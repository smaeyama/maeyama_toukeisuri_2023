#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from mzprojection import mzprojection_multivariate_discrete_time, split_long_time_series, calc_correlation

data = np.loadtxt("data/model1.dat")
nrec=int(data.shape[0]/1)
m=data.shape[1]
data=data[:nrec,:m]
print(data.shape)

t_raw = np.arange(nrec)
u_raw = data[:-1,:]
f_raw = data[1:,:]


# In[2]:


#= Split a long time series data into samples of short-time data =
ista    = 10                # Start time step number for sampling   
nperiod = 10                # Time step length of a short-time sample                 
nshift  = 1                 # Length of time shift while sampling
t   =split_long_time_series(t_raw,ista=ista,nperiod=nperiod,nshift=nshift)
u   =split_long_time_series(u_raw,ista=ista,nperiod=nperiod,nshift=nshift)
f   =split_long_time_series(f_raw,ista=ista,nperiod=nperiod,nshift=nshift)


# In[3]:


#= Mori-Zwanzig projection =
omega, memoryf, s, r = mzprojection_multivariate_discrete_time(u, f, flag_terms=True, flag_debug=True)


# In[4]:


t_cor = np.arange(memoryf.shape[0])
nsample, nperiod, nu = u.shape
nf = f.shape[2]
u0 = u[:,0,:]
f0 = f[:,0,:]
r0 = r[:,0,:]
uu = calc_correlation(u,u0)
ff = calc_correlation(f,f0)
fu = calc_correlation(f,u0)
rr = calc_correlation(r,r0)
ru = calc_correlation(r,u0)
uu0_inv = np.linalg.inv(uu[0,:,:])
mr = r[:,:,:] - np.tensordot(u[:,0,:],memoryf[:,:,:],axes=((-1),(-1)))
mru = calc_correlation(mr,u0)
uu1_inv = np.linalg.inv(uu[1,:,:])

i,j=1,1
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlabel("Correlation time t")
ax.set_ylabel(r"$\Gamma(t_n) = - \langle M r(t_{n-1}) u \rangle \cdot \langle u u \rangle^{-1}$")
ax.plot(0,omega[i,j],"s",label="$\Omega_{ij}$")
ax.plot(t_cor,memoryf[:,i,j],label="$\Gamma_{ij}(t_n)$",c="C0")
ax.plot(t_cor,np.dot(-mru,uu0_inv)[:,i,j],"--",label=r"$-\langle Mr_i(t_{n-1}) u_k \rangle \langle u u \rangle^{-1}_{k,j}$",lw=3)
# ax.plot(t_cor,np.dot(rr,uu1_inv.T)[:,i,j],"--",label=r"$-\langle Mr_i(t_{n-1}) u_k \rangle \langle u u \rangle^{-1}_{k,j}$",lw=3) # Lin'21arXiv
plt.legend()
plt.show()
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlabel("Correlation time t")
ax.set_ylabel(r"Correlation $\langle f(t_n) u \rangle \gg \langle r(t_n) u \rangle$")
ax.plot(t_cor,ru[:,i,j],label=r"$\langle r_i(t_n) u_j \rangle$")
ax.plot(t_cor,fu[:,i,j],label=r"$\langle f_i(t_n) u_j \rangle$")
plt.legend()
plt.show()
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlabel("Time t")
ax.set_ylabel("$f(t_n)=\Omega \cdot u(t_n) + s(t_n) + r(t_n)$")
ax.plot(t_cor,f[10,:,i],label="$f_i(t_n)$")
ax.plot(t_cor,np.dot(u[10,:,:],omega.T)[:,i],label="$\Omega_{ij} u_j(t_n)$")
ax.plot(t_cor,s[10,:,i],label="$s_i(t_n)$")
ax.plot(t_cor,r[10,:,i],label="$r_i(t_n)$")
ax.scatter(t_cor,np.dot(u[10,:,:],omega.T)[:,i]+s[10,:,i]+r[10,:,i])
plt.legend()
plt.show()


# In[5]:


fig = plt.figure(figsize=(12,4))
ax=fig.add_subplot(111)
vmax=np.max(abs(omega))
quad = ax.pcolormesh(np.arange(nu),np.arange(nf),omega[:,:],
                     cmap='jet',shading="auto",vmin=-vmax,vmax=vmax)
ax.set_xlabel(r"Index $j$")
ax.set_ylabel(r"Index $i$")
ax.set_title(r"Markov coefficient matrix $\Omega_{ij}$ [$f_i(t)=\Omega_{ij}u_j(t)+s_i(t)+r_i(t)$]")
ax.set_aspect("equal")
fig.colorbar(quad)
fig.tight_layout()
plt.show()


# In[6]:


def plot_memoryf(t_cor,memoryf,i=0):
    fig = plt.figure(figsize=(10,4))
    ax=fig.add_subplot(121)
    ax.axhline(0,lw=0.5,c="k")
    vmax=np.max(abs(memoryf[:,i,:]))
    for j in range(nu):
        vmax_j=np.max(abs(memoryf[:,i,j]))
        if vmax_j > 0.3*vmax:
            ax.plot(t_cor,memoryf[:,i,j],"o-",label="$\Gamma_{"+"{:},{:}".format(i,j)+"}(t)$")
    ax.set_xlabel(r"Correlation time $t$")
    ax.set_ylabel(r"Memory function $\Gamma_{ij}(t)$")
    ax.legend(loc="center left",bbox_to_anchor=(1,0.5))

    ax=fig.add_subplot(122)
    ax.axhline(0,lw=0.5,c="k")
    vmax=np.max(abs(memoryf[:,i,:]))
    quad = ax.pcolormesh(t_cor,np.arange(nu),memoryf[:,i,:].T,
                         cmap='jet',shading="auto",vmin=-vmax,vmax=vmax)
    ax.set_xlabel(r"Correlation time $t$")
    ax.set_ylabel(r"Index $j$")
    fig.colorbar(quad)
    fig.suptitle(r"Memory function $\Gamma_{i,j}(t)$ for $i="+r"{:}$".format(i))
    fig.tight_layout()

    
for i in range(nf):
    plot_memoryf(t_cor,memoryf,i); plt.show()


# # Compare with SVAR
# ### (i) Markov coefficient and memory function

# In[7]:


A1 = omega-memoryf[0]
A2 = -memoryf[1]
A3 = -memoryf[2]

from matplotlib import colors
cmap = plt.cm.RdBu_r
bounds = np.linspace(-1.05, 1.05, 22)
norm = colors.BoundaryNorm(bounds, cmap.N)

np.set_printoptions(precision=4, floatmode='fixed',suppress=True)
print("A_1=\n",A1)
fig=plt.figure()
ax=fig.add_subplot(111)
quad = ax.pcolormesh(np.arange(m),np.arange(m),A1,
                     cmap=cmap,norm=norm,shading="auto")
ax.set_ylim(m-0.5,0-0.5)
ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
ax.set_title(r"$\Omega_{ij}$")
# ax.set_title(r"$A_1[i,j]$")
secax = ax.secondary_xaxis('top')
secax.set_xlabel(r'Index $j$')
ax.set_ylabel(r"Index $i$")
fig.colorbar(quad,ticks=np.linspace(-1, 1, 21))
fig.tight_layout()
plt.show()

print("A_2=\n",A2)
fig=plt.figure()
ax=fig.add_subplot(111)
quad = ax.pcolormesh(np.arange(m),np.arange(m),A2,
                     cmap=cmap,norm=norm,shading="auto")
ax.set_ylim(m-0.5,0-0.5)
ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
ax.set_title(r"$-\Gamma_{ij}(t_1)$")
# ax.set_title(r"$A_2[i,j]$")
secax = ax.secondary_xaxis('top')
secax.set_xlabel(r'Index $j$')
ax.set_ylabel(r"Index $i$")
fig.colorbar(quad,ticks=np.linspace(-1, 1, 21))
fig.tight_layout()
plt.show()

print("A_3=\n",A3)
fig=plt.figure()
ax=fig.add_subplot(111)
quad = ax.pcolormesh(np.arange(m),np.arange(m),A3,
                     cmap=cmap,norm=norm,shading="auto")
ax.set_ylim(m-0.5,0-0.5)
ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
ax.set_title(r"$A_3[i,j]$")
secax = ax.secondary_xaxis('top')
secax.set_xlabel(r'Index $j$')
ax.set_ylabel(r"Index $i$")
fig.colorbar(quad,ticks=np.linspace(-1, 1, 21))
fig.tight_layout()
plt.show()


# ### (ii) Statistics of the uncorrelated term

# In[8]:


for im in range(m):
    ave = np.average(r[:,:,im])
    var = np.var(r[:,:,im])
    print(r"r_{:}".format(im)+": average={:8.4f}".format(ave),", variance={:8.4f}".format(var))

def gaussian(x,mean=0,std=1):
    return np.exp(-(x-mean)**2/(2*std**2))/np.sqrt(2*np.pi*std**2)

fig = plt.figure(figsize=(14,4))
ncol=6
for im in range(m):
    ax = plt.subplot2grid((int(m/ncol)+1,ncol), (int(im/ncol),im%ncol))
    d = r[:,:,im]
    x_hist, bins = np.histogram(d,bins=100)
    x_range = (bins[:-1] + bins[1:])/2
    bin_width = bins[1]-bins[0]
    pdf = x_hist / d.size / bin_width
    #print("# Check normalization of PDF", np.sum(pdf)*bin_width)
    ax.plot(x_range,pdf)
    ax.plot(x_range,gaussian(x_range,mean=0,std=np.sqrt(0.1)))
    ax.set_xlabel("$r_{:}$".format(im))
    ax.set_ylim(0,1.4)
    ax.set_xlim(-2,2)
fig.suptitle("Probability distribution function (PDF)")
fig.tight_layout()
plt.show()

fig = plt.figure(figsize=(8,12))
ncol=1
for im1 in range(m):
    ax = plt.subplot2grid((int(m/ncol)+1,ncol), (int(im1/ncol),im1%ncol))
    ax.axhline(0,lw=0.5,c="k")
    for im2 in range(m):
        ax.plot(rr[:,im1,im2],label=r"$\langle y_{:}(t),y_{:} \rangle$".format(im1,im2))
        ax.set_xlabel(r"Time step")
        ax.set_ylabel(r"$\langle r_{:}(t),r_j \rangle$".format(im1))
        #ax.set_xlim(0,9)
        ax.legend(loc="center left", bbox_to_anchor=(1,0.5))
fig.suptitle(r"Cross-correlation function (CCF) $\langle r_i(t),r_j \rangle$")
fig.tight_layout()
plt.show()


# ### (iii) Reproduce simulation data based on MZ coefficients 

# In[9]:


p=nperiod

A=np.zeros([p+1,m,m])
A[1,:,:] = omega-memoryf[0,:,:]
A[2:,:,:] = -memoryf[1:,:,:]
A0inv = np.linalg.inv(np.identity(m)-A[0,:,:])
e_var=[np.var(r[:,:,im]) for im in range(m)]
r_mean=np.average(r[:,:,:],axis=(0,1))
r_cov=rr[0,:,:]
print(r_mean,r_cov)

nrec = 200000
yprev = np.random.normal(loc=0.0,scale=np.sqrt(0.1),size=(p,m))
y = [*yprev]
for t in range(nrec-p):
#     aype = np.random.normal(loc=0.0,scale=np.sqrt(e_var),size=(m))
    aype = np.random.multivariate_normal(mean=r_mean,cov=r_cov)
    for i in range(1,A.shape[0]):
        aype = aype + np.dot(A[i,:,:],y[-i])
    yt = np.dot(A0inv,aype)
    y.append(yt)
y=np.array(y)
print(y.shape)

fig=plt.figure(figsize=(18,2))
ax=fig.add_subplot(111)
for im in range(m):
    ax.plot(y[:,im],label="$y_{:}$".format(im))
ax.set_xlabel("Time step")
ax.set_xlim(0,None)
ax.legend()
plt.show()


# In[10]:


for im in range(m):
    ave = np.average(y[:,im])
    var = np.var(y[:,im])
    print(r"y_{:}".format(im)+": average={:8.4f}".format(ave),", variance={:8.4f}".format(var))

fig = plt.figure(figsize=(14,4))
ncol=6
for im in range(m):
    ax = plt.subplot2grid((int(m/ncol)+1,ncol), (int(im/ncol),im%ncol))
    d = y[:,im]
    x_hist, bins = np.histogram(d,bins=100)
    x_range = (bins[:-1] + bins[1:])/2
    bin_width = bins[1]-bins[0]
    pdf = x_hist / d.size / bin_width
    #print("# Check normalization of PDF", np.sum(pdf)*bin_width)
    ax.plot(x_range,pdf)
    ax.set_xlabel("$y_{:}$".format(im))
    ax.set_ylim(0,0.8)
    ax.set_xlim(-5,5)
fig.suptitle("Probability distribution function (PDF)")
fig.tight_layout()
plt.show()


# In[11]:


fig = plt.figure(figsize=(8,12))
ncol=1
for im1 in range(m):
    ax = plt.subplot2grid((int(m/ncol)+1,ncol), (int(im1/ncol),im1%ncol))
    d1 = y[:,im1]
    d1 = d1 - np.average(d1)
    ax.axhline(0,lw=0.5,c="k")
    for im2 in range(m):
        d2 = y[:,im2]
        d2 = d2 - np.average(d2)
        ccf = np.correlate(d1,d2,mode="full")[len(d1)-1:]
        ccf = ccf / np.sqrt(np.sum(d1*d1)*np.sum(d2*d2))
        ax.plot(ccf[:],label=r"$\langle y_{:}(t),y_{:} \rangle$".format(im1,im2))
        ax.set_xlabel(r"Time step")
        ax.set_ylabel(r"$\langle y_{:}(t),y_j \rangle$".format(im1))
        ax.set_xlim(0,100)
        ax.legend(loc="center left", bbox_to_anchor=(1,0.5))
fig.suptitle(r"Cross-correlation function (CCF) $\langle y_i(t),y_j \rangle$")
fig.tight_layout()
plt.show()


# In[12]:


np.set_printoptions(precision=4, floatmode='maxprec',suppress=True)
print("rr0=\n",np.round(rr[0,:,:],3),rr[0].max(),rr[0].min())


# # Fig. 4

# In[13]:


plt.rcParams["font.size"]=7
plt.rcParams["axes.titlesize"]=7

model="model1"
p_svar=2
m_svar=5
coeff_svar = np.loadtxt("./coefficients_"+str(model)+".txt")
B=coeff_svar.reshape(p_svar+1,m_svar,m_svar)
Vinv = np.linalg.inv(np.identity(m_svar)-B[0,:,:])

A1 = np.dot(Vinv,B[1,:,:])
A2 = np.dot(Vinv,B[2,:,:])
Exi = 0.1*np.dot(Vinv,Vinv.T)

# A1 = omega-memoryf[0]
# A2 = -memoryf[1]
# A3 = -memoryf[2]

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
secax.set_xticks(np.arange(m_svar))
ax.set_xticks([])
ax.set_ylim(m-0.5,0-0.5)
ax.set_yticks(np.arange(m_svar-1,-1,-1))
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
secax.set_xticks(np.arange(m_svar))
ax.set_xticks([])
ax.set_ylim(m-0.5,0-0.5)
ax.set_yticks(np.arange(m_svar-1,-1,-1))
ax.set_ylabel(r"Index $i$")
fig.colorbar(quad,ticks=np.linspace(-1, 1, 11),shrink=0.68)
ax.set_aspect("equal")


ax=fig.add_subplot(234)
quad = ax.pcolormesh(np.arange(m),np.arange(m),omega-memoryf[0],
                     cmap=cmap,norm=norm,shading="auto")
ax.set_ylim(m-0.5,0-0.5)
ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
ax.set_title(r"(d) Markov $\Omega_{ij}$")
secax = ax.secondary_xaxis('top')
secax.set_xlabel(r'Index $j$')
secax.set_xlim(0-0.5,m-0.5)
secax.set_xticks(np.arange(m_svar))
ax.set_xticks([])
ax.set_ylim(m-0.5,0-0.5)
ax.set_yticks(np.arange(m_svar-1,-1,-1))
ax.set_ylabel(r"Index $i$")
fig.colorbar(quad,ticks=np.linspace(-1, 1, 11),shrink=0.68)
ax.set_aspect("equal")

ax=fig.add_subplot(235)
quad = ax.pcolormesh(np.arange(m),np.arange(m),-memoryf[1],
                     cmap=cmap,norm=norm,shading="auto")
ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
ax.set_title(r"(e)Memory $-\Gamma_{ij}(1)$")
secax = ax.secondary_xaxis('top')
secax.set_xlabel(r'Index $j$')
secax.set_xlim(0-0.5,m-0.5)
secax.set_xticks(np.arange(m_svar))
ax.set_xticks([])
ax.set_ylim(m-0.5,0-0.5)
ax.set_yticks(np.arange(m_svar-1,-1,-1))
ax.set_ylabel(r"Index $i$")
fig.colorbar(quad,ticks=np.linspace(-1, 1, 11),shrink=0.68)
ax.set_aspect("equal")


bounds = np.linspace(-0.16, 0.16, 22)
norm = colors.BoundaryNorm(bounds, cmap.N)

ax=fig.add_subplot(233)
quad = ax.pcolormesh(np.arange(m),np.arange(m),Exi,
                     cmap=cmap,norm=norm,shading="auto")
ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
ax.set_title(r"(c) SVAR $E=\langle \xi_i \xi_j \rangle$")
secax = ax.secondary_xaxis('top')
secax.set_xlabel(r'Index $j$')
secax.set_xlim(0-0.5,m-0.5)
secax.set_xticks(np.arange(m_svar))
ax.set_xticks([])
ax.set_ylim(m-0.5,0-0.5)
ax.set_yticks(np.arange(m_svar-1,-1,-1))
ax.set_ylabel(r"Index $i$")
fig.colorbar(quad,ticks=np.linspace(-0.15, 0.15, 7),shrink=0.68)
ax.set_aspect("equal")

ax=fig.add_subplot(236)
quad = ax.pcolormesh(np.arange(m),np.arange(m),rr[0,:,:],
                     cmap=cmap,norm=norm,shading="auto")
ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
ax.set_title(r"(f) Uncorrelated $\langle r_i r_j \rangle$")
secax = ax.secondary_xaxis('top')
secax.set_xlabel(r'Index $j$')
secax.set_xlim(0-0.5,m-0.5)
secax.set_xticks(np.arange(m_svar))
ax.set_xticks([])
ax.set_ylim(m-0.5,0-0.5)
ax.set_yticks(np.arange(m_svar-1,-1,-1))
ax.set_ylabel(r"Index $i$")
fig.colorbar(quad,ticks=np.linspace(-0.15, 0.15, 7),shrink=0.68)
ax.set_aspect("equal")

fig.tight_layout()
plt.savefig("fig_svar_matrix.pdf")
plt.show()


# In[14]:


np.savetxt("./data/A1.txt",A1)
np.savetxt("./data/A2.txt",A2)
np.savetxt("./data/Exi.txt",Exi)
np.savetxt("./data/omega.txt",omega)
np.savetxt("./data/gamma1.txt",memoryf[1,:,:])
np.savetxt("./data/rr0.txt",rr[0,:,:])


# In[15]:


im=0
d = y[:,im]
x_hist, bins = np.histogram(d,bins=100)
x_range = (bins[:-1] + bins[1:])/2
bin_width = bins[1]-bins[0]
pdf = x_hist / d.size / bin_width
x_range = (bins[:-1] + bins[1:])/2

im1=0
d1 = y[:,im1]
d1 = d1 - np.average(d1)
ccf = []
for im2 in range(m):
    d2 = y[:,im2]
    d2 = d2 - np.average(d2)
    wccf = np.correlate(d1,d2,mode="full")[len(d1)-1:]
    wccf = wccf / np.sqrt(np.sum(d1*d1)*np.sum(d2*d2))
    ccf.append(wccf)
ccf=np.array(ccf)[:,0:100]
np.savetxt("./data/range_y0_reproduction.txt",x_range)
np.savetxt("./data/pdf_y0_reproduction.txt",pdf)
np.savetxt("./data/ccf_y0_reproduction.txt",ccf)


# In[16]:


print(pdf.shape,ccf.shape)
plt.plot(x_range,pdf)
plt.show()
plt.plot(ccf.T)
plt.show()


# In[ ]:





# In[ ]:




