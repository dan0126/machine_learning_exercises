import os.path
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg
import sys
import random
import math
import pandas as pd
from sklearn import linear_model
from mpl_toolkits.mplot3d import Axes3D


##########################__REGRESSÃO LINEAR EM UMA DIMENSÃO__##################

def valores(b): #retorna a lista dos valores

    basepath = '/Users/dannilin/Desktop/IC/dados/'

    l = []

    dat = open(basepath + b).readlines()
    
    for i in range(len(dat)):

         l.append(float((dat[i].rstrip('\n')).replace(',' , '.')))

    return l
        
def A():

    x = valores('mol.txt')

    l = []

    m = []

    m1 = []

    a = 0

    b = 0

    c = 0

    d = 0

    for i in range(len(x)):

        l.append(1)
    
    for j in range(len(x)):

        #a += len(x)+1

        b += x[j]
    
        d += x[j]*x[j]
        
    a = len(x)+1

    m.append(a)
    m.append(b)
    m1.append(b)
    m1.append(d)

    return m,m1

def B():

    y = valores('boiling.txt')

    x = valores('mol.txt')

    m = []

    a = 0

    b = 0

    for j in range(len(x)):

        a += y[j]

        b += y[j]*x[j]

    m.append(a)
    m.append(b)

    return m

def solve():

    l = [A()[0],A()[1]]

    b = B()

    return np.linalg.solve(l,b)
    #onde o primeiro valor é o coeficiente linear e o segundo é o coeficiente angular

def ord_lin():

    x = valores('mol.txt')

    l = []

    a = solve()[0]

    b = solve()[1]

    for i in range(len(x)):

        l.append(b*x[i]+a)

    return l

def aad():

    o = valores('boiling.txt')

    p = ord_lin()

    a = 0

    for i in range(len(o)):

        a += (np.abs(np.round(p[i],decimals = 2) - o[i])/o[i])

    return a/len(o)*100

def r2():

    o = valores('boiling.txt')

    p = ord_lin()

    a = 0

    b = 0

    c = 0

    d = 0

    e = 0

    for i in range(len(o)):

        a += o[i]*p[i]

        b += p[i] #x

        c += o[i] #y

        d += p[i]*p[i]

        e += o[i]*o[i]

    return (len(o)*a-(b*c))/np.sqrt(((len(o)*(d))-(b*b))*((len(o)*e)-(c*c)))

def grafico2():

    y = valores('boiling.txt')

    y_lin = ord_lin()

    x = valores('mol.txt')

    y_lin = ord_lin()

    plt.style.use('default')
    plt.style.use('ggplot')

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(x,y_lin , color='k', label = ('Regression model com AAD = ' + str(np.round(aad(),decimals = 2))+ '% '))
    ax.scatter(x, y, edgecolor='k', facecolor='grey', alpha=0.7, label='Sample data')
    ax.set_ylabel('Boiling point, T [K]', fontsize=14)
    ax.set_xlabel('Molecular weight, Mw [g/mol]', fontsize=14)
    ax.text(0.8, 0.1, 'aegis4048.github.io', fontsize=13, ha='center', va='center',
         transform=ax.transAxes, color='grey', alpha=0.5)
    ax.legend(facecolor='white', fontsize=11)
    ax.set_title('$R^2= %.2f$' % r2(), fontsize=18)

    fig.tight_layout()

    plt.show()
        

################################################################################


