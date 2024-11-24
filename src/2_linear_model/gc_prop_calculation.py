#!/usr/bin/env python
# coding: utf-8
import autograd.numpy as np

def lhs_cal(tag, Y):
    if tag == 'FLVL':
        a = 4.53
        lhs = np.log(Y/a)
    elif tag == 'Tm':
        a = 198
        lhs = np.exp(Y/a)
    elif tag == 'Tc':
        a = 250
        lhs = np.exp(Y/a)
    elif tag == 'Tb':
        a = 240
        lhs = np.exp(Y/a)
    elif tag == 'Vc':
        a = 20
        lhs = Y - a
    elif tag == 'Vm':
        a = 0.01
        lhs = Y - a
    elif tag == 'Pc':
        a = 0.1347
        lhs = np.power((Y-0.0519),-0.5) - a
    elif tag == 'Svb':
        a = 80
        lhs = Y - parameter
    elif tag == 'Omega':
        a = [0.9080, 0.1055, 1.0012]
        lhs = np.power(np.exp(Y/a[0]), a[1]) - a[2]
    elif tag == 'LogKow':
        a = 0.5
        lhs = Y - a
    elif tag == 'Hv':
        a = 10
        lhs = Y - a
    elif tag == 'Hvb':
        a = 15
        lhs = Y - a
    elif tag == 'Hf':
        a = 80
        lhs = Y - a
    elif tag == 'Hfus':
        a = -2
        lhs = Y - a
    elif tag == 'Hild_Solub':
        a = 21
        lhs = Y - a
    elif tag == 'Gf':
        a = 8
        lhs = Y - a
    elif tag in {'Hans_Solub_D', 'Hans_Solub_H', 'Hans_Solub_P'}:
        a = null
        lhs = Y
    return a, lhs
    
def y_pred_cal(x, a, theta, const_fg, const_sg, tag):
    if tag == 'FLVL':
        y_pred = a * np.exp(const_fg + const_sg + np.matmul(x, theta))
    elif tag in {'Tc', 'Tm', 'Tb'}:
        y_pred = a * np.log(const_fg + const_sg + np.matmul(x, theta))
    elif tag in {'Vc', 'Gf', 'Hf', 'Hfus', 'HV', 'Hvb', 'Svb', 'Hild_Solub', 'Vm', 'LogKow'}:
        y_pred = (const_fg + const_sg + np.matmul(x, theta)) + a
    elif tag == 'Pc':
        y_pred = 1/((const_fg + const_sg + np.matmul(x, theta) + a)**2) + 0.0519
    elif tag == {'Hans_Solub_D', 'Hans_Solub_H', 'Hans_Solub_P'}:
        y_pred = const_fg + const_sg + np.matmul(x, theta)
    elif tag == 'Omega':
        y_pred = a[0] * np.log(np.sign(const_fg + const_sg + np.matmul(x, theta) + a[2]) * np.abs(const_fg + const_sg + np.matmul(x, theta) + a[2])**(1/a[1]))
    return y_pred

