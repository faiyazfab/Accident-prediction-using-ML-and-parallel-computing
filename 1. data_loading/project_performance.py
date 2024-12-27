#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


configurations = ["2 CPUs", "4 CPUs", "6 CPUs", "8 CPUs"]
speedups = [1.0, 1.8, 2.5, 3.0]
efficiencies = [100, 90, 83, 75]


plt.figure(figsize=(10, 6))
plt.plot(configurations, speedups, marker='o', linestyle='-', label='Speedup')
plt.title('Speedup Analysis')
plt.xlabel('Configuration')
plt.ylabel('Speedup')
plt.grid(True)
plt.legend()
plt.savefig('/home/wei.shao/Project/Speedup_Analysis.png')
plt.close()


plt.figure(figsize=(10, 6))
plt.plot(configurations, efficiencies, marker='o', linestyle='-', color='orange', label='Efficiency')
plt.title('Efficiency Analysis')
plt.xlabel('Configuration')
plt.ylabel('Efficiency (%)')
plt.grid(True)
plt.legend()
plt.savefig('/home/wei.shao/Project/Efficiency_Analysis.png')
plt.close()

print("Performance analysis plots saved in /home/wei.shao/Project/")


