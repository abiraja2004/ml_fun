# -*- coding: utf-8 -*-
# import csv
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pylab as plt
import statsmodels.api as sm
#import ggplot as gg
from ggplot import *

df = pd.DataFrame.from_csv('leagues_NBA_2013_14_total_edited.csv', index_col=None)
df = df[df['WSp48'] > -0.05]
#table = df.pivot_table('WinPct', 'WSp48','Player')
x=df['WSp48']
y=df['WinPct']
t=df['Team']
#VORP are the values, Season the index, Player the columns

#print table

#Player  PlayerA  PlayerB
#Season
#'0405’     0.70     0.23
#'0506’     0.14    -0.30
lowess = sm.nonparametric.lowess(y, x, frac=0.1)
#df = gg.df
#qplot(df.x, df.y) + geom_smooth(color="blue")
#ax = table.plot(marker='o')
#ax.set_title('WinPct vs WSp48')
#ax.set_ylabel('WinPct')
#ax.legend(loc=3,prop={'size':10})
#
g=ggplot(aes(x='WSp48', y='WinPct',group='Team'), data=df) +\
    geom_point() + ggtitle("WinPct vs WSp48") +\
    stat_smooth(colour='blue', span=0.2)
# plot <- ggplot(df, aes(parent, child, size=count)) + geom_point() + ggtitle("Bubble Plot of Height")
#plt.plot(x, y, '+')
#plt.plot(lowess[:, 0], lowess[:, 1],color='red')
g.draw()
plt.show()