#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt

ranges = np.loadtxt("./data/range_y0_simulation.txt")
pdfs = np.loadtxt("./data/pdf_y0_simulation.txt")
ccfs= np.loadtxt("./data/ccf_y0_simulation.txt")
ranger = np.loadtxt("./data/range_y0_reproduction.txt")
pdfr = np.loadtxt("./data/pdf_y0_reproduction.txt")
ccfr= np.loadtxt("./data/ccf_y0_reproduction.txt")


fig=plt.figure(figsize=(5,2.7),dpi=150)
plt.rcParams["font.size"]=8
plt.rcParams["axes.titlesize"]=8
# plt.rcParams["legend.fontsize"]=7
ax=fig.subplots(2,2,gridspec_kw={'width_ratios': [1, 4],'height_ratios': [1, 1]})
ax[0][0].set_title("(a) PDF in simulation")
ax[0][0].plot(ranges,pdfs,c="k")
ax[0][0].set_xlabel(r"$Y_0(n)$")
ax[0][0].set_xlim(-2,2)
ax[0][0].set_ylim(0,0.8)

ax[0][1].set_title("(b) Cross-correlation in simulation")
linestyles=["solid","dashed","dotted","dashdot",(0, (3, 1, 1, 1, 1, 1))]
im1=0
for im2 in range(ccfs.shape[0]-1):
    ax[0][1].plot(ccfs[im2,:],linestyle=linestyles[im2],c="k",label=r"$\langle Y_{:}(n)Y_{:} \rangle$".format(im1,im2))
ax[0][1].set_xlabel("Time step")
ax[0][1].set_xlim(0,40)
ax[0][1].set_ylim(-0.3,1.05)
ax[0][1].axhline(0,lw=0.5,c="k")
ax[0][1].legend(loc="center left", bbox_to_anchor=(1,0.5))

ax[1][0].set_title("(c) PDF in reproduction")
ax[1][0].plot(ranger,pdfr,c="k")
ax[1][0].set_xlabel(r"$y_0(n)$")
ax[1][0].set_xlim(-2,2)
ax[1][0].set_ylim(0,0.8)

ax[1][1].set_title("(d) Cross-correlation in reproduction")
im1=0
for im2 in range(ccfr.shape[0]-1):
    ax[1][1].plot(ccfr[im2,:],linestyle=linestyles[im2],c="k",label=r"$\langle y_{:}(n)y_{:} \rangle$".format(im1,im2))
ax[1][1].set_xlabel("Time step")
ax[1][1].set_xlim(0,40)
ax[1][1].set_ylim(-0.3,1.05)
ax[1][1].axhline(0,lw=0.5,c="k")
ax[1][1].legend(loc="center left", bbox_to_anchor=(1,0.5))
fig.tight_layout()
plt.savefig("fig_svar_ccf.pdf")
plt.show()


# In[ ]:




