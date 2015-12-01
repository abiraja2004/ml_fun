# -*- coding: utf-8 -*-
#
# import csv
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import pandas as pd


df = pd.DataFrame.from_csv('BGvsKHvsLuolvsKobevsRWvsRose.csv', index_col=None)
table = df.pivot_table('VORP', 'Season', 'Player')
#VORP are the values, Season the index, Player the columns

#print table

#Player  PlayerA  PlayerB
#Season
#'0405’     0.70     0.23
#'0506’     0.14    -0.30

ax = table.plot(marker='o')
ax.set_title('VORP vs Season')
ax.set_ylabel('VORP')
ax.legend(loc=3,prop={'size':10})
plt.show()