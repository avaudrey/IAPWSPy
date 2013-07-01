#!/usr/bin/python
# -*- coding: utf-8 -*-

#----------------------------------------------------------------------------
#   Copyright (C) 2013 <Alexandre Vaudrey>                                  |
#                                                                           |
#   This program is free software: you can redistribute it and/or modify    |
#   it under the terms of the GNU General Public License as published by    |
#   the Free Software Foundation, either version 3 of the License, or       |
#   (at your option) any later version.                                     |
#                                                                           |
#   This program is distributed in the hope that it will be useful,         |
#   but WITHOUT ANY WARRANTY; without even the implied warranty of          |
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the           |
#   GNU General Public License for more details.                            |
#                                                                           |
#   You should have received a copy of the GNU General Public License       |
#   along with this program.  If not, see <http://www.gnu.org/licenses/>.   |
#---------------------------------------------------------------------------|

import numpy as np
import pylab as pl
import scipy.optimize as op
from functools import wraps

__docformat__ = "restructuredtext en"
__author__ = "Alexandre Vaudrey <alexandre.vaudrey@gmail.com>"
__date__ = "01/02/2013"

# Fundamentals physical datas : Temperature [K], pressure [bar] and density
# [kg/m3] at the critical point : 
Tc , pc , rhoc = 647.096 , 220.064 , 322.0
# and at the triple point :
Tt , pt = 273.16 , 611.657e-5

# ---- Miscellaneous functions used in different steps of the calculation -----
# If the temperature used as argument is greater than the critical one or lower
# than the triple one, we "cut and replace"
def _cut(T):
    # We cut any value of the temperature that is not the available range of
    # values
    if (T > Tc):
        return Tc
    elif (T < Tt):
        return Tt
    else:
        return T
# Function used to calculate the different experimental formula, both based on
# a sum of power laws :
def powerweight(x,expo,weight):
    """
    Sum of power laws used to calculate the physical properties of water.
    """
    return np.dot(weight,np.power(x,expo))
# Decorator which make the difference between a 'float' or a 'list'/'array' as
# an argument and just deal with it.
def vectorize(func):
    """Check whether the argument of function 'func' is a float or a list/array."""
    # Decorator used to not loose the informations (as e.g. the docstring) of
    # the decorated function, see : 
    # http://stackoverflow.com/questions/308999/what-does-functools-wraps-do
    @wraps(func)
    # Actual decorator
    def wrapper(*args):
        if np.isscalar(args[0]):
            # If argument of function 'func' is a float, just go on
            return func(*args)
        elif (type(args[0]) == list) or (type(args[0]) == tuple) or \
                (type(args[0]) == np.ndarray) :
            # If it's a list, a tuple or an array, we just 'map' it and convert
            # the result in a array
            return np.array(map(func,list(args[0])))
        else:
            # In any other case, we just try and hope
            try:
                return func(*args)
            except TypeError:
                return "'Wrong type of argument'"
    return wrapper
# ---- Actual calculation of the water physical properties --------------------
# Saturated vapor pressure of water
@vectorize
def psat(T):
    """
    Calculation of the saturation pressure, in bar, for a given temperature, in
    K. Value of the temperature used as argument must be between the triple
    point temperature and the critical one. If not, its value is corrected.
    """
    # Ensure that the temperature value is within the available range of values,
    # i.e. between the triple point temperature and the critical one
    T = _cut(T)
    # Sort of "Carnot" temperature regarding to the critical one, parameter used
    # as temperature on the correlations
    theta = 1-T/Tc
    # Experimental parameters given by the IAPWS data, exponents
    expo = np.array([1.0,1.5,3.0,3.5,4.0,7.5])
    # and relating weights
    weight = np.array([-7.85951783,1.84408259,-11.7866497,22.6807411,-15.9618719,\
                       1.80122502])
    # And calculation of the saturation pressure
    return pc*np.exp(powerweight(theta,expo,weight)/(1-theta))
@vectorize
def psat_derivative(T):
    """
    First derivative of the saturated pressure 'psat' regarding to the
    temperature 'T', used in different calculations.
    """
    # Ensure that the temperature value is within the available range of values,
    # i.e. between the triple point temperature and the critical one
    T = _cut(T)
    # Sort of "Carnot" temperature regarding to the critical one, parameter used
    # as temperature on the correlations
    theta = 1-T/Tc
    # Experimental parameters given by the IAPWS data, exponents
    expo = np.array([0.0,0.5,2.0,2.5,3.0,6.5])
    # and relating weights
    weight = np.array([-7.85951783,1.5*1.84408259,-3*11.7866497,3.5*22.6807411,\
                       -4*15.9618719,7.5*1.80122502])
    # Value of the saturated pressure
    ps = psat(T)
    # And calculation of the saturation pressure
    return -ps/T*(powerweight(theta,expo,weight)+np.log(ps/pc))
# Reciprocal function of the previous one, which calculate the saturated vapor
# temperature corresponding to a given pressure.
@vectorize
def Tsat(p):
    """
    Calculation of the saturation temperature, in K, for a given pressure, in
    bar. Tini is the (optional) initial guess of temperature used to start the
    fixed point algorithm. "acc" is the accuracy needed for the result.
    """
    def _cutp(p):
        # Correction of the pressure value if it is out of the available range.
        if (p > pc):
            return pc
        elif (p < pt):
            return pt
        else:
            return p
    p = _cutp(p)
    # If argument of function 'func' is a float, just go on
    # Shortcut is the pressure p is the critical one
    if (p == pc):
        return Tc
    else:
        # If not, we just use the newton method from scipy.optimize to solve the
        # problem. Function f to find the root
        def f(T):
            return psat(T)-p
        # and resolution
        return op.newton(f,Tc,fprime=psat_derivative)
# Density of the saturated liquid water 
@vectorize
def liquid_density(T):
    """
    Calculation of the liquid water density (in [kg/m3]) at the saturated state
    for a given value of the temperature in K. Value of the temperature used as
    argument must be between the triple point temperature and the critical one.
    If not, its value is corrected.
    """
    # Ensure that the temperature value is within the available range of values,
    # i.e. between the triple point temperature and the critical one
    T = _cut(T)
    # Sort of "Carnot" temperature regarding to the critical one, parameter used
    # as temperature on the correlations
    theta = 1-T/Tc
    # Experimental parameters given by the IAPWS data, exponents
    expo = np.array([1,2,5,16,43,110])/3.0
    # and relating weights
    weight = np.array([1.99274064,1.09965342,-0.510839303,-1.75493479,\
                       -45.5170352,-6.7469945e+5])
    # And calculation of the saturation pressure
    return rhoc*(1+powerweight(theta,expo,weight))
# Specific volume of liquid water
def liquid_specific_volume(T):
    """
    Specific volume of liquid water (in [m3/kg]) at the saturated state for a given
    value of temperature in K. Value of the temperature used as argument must be
    between the triple point temperature and the critical one. If not, its value
    is corrected.
    """
    return 1/liquid_density(T)
# Density of the saturated water vapor
@vectorize
def vapor_density(T):
    """
    Calculation of the water vapor density (in [kg/m3]) at the saturated state
    for a given value of the temperature in K. Value of the temperature used as
    argument must be between the triple point temperature and the critical one.
    If not, its value is corrected.
    """
    # Ensure that the temperature value is within the available range of values,
    # i.e. between the triple point temperature and the critical one
    T = _cut(T)
    # Sort of "Carnot" temperature regarding to the critical one, parameter used
    # as temperature on the correlations
    theta = 1-T/Tc
    # Experimental parameters given by the IAPWS data, exponents
    expo = np.array([2,4,8,18,37,71])/6.0
    # and relating weights
    weight = np.array([-2.0315024,-2.6830294,-5.38626492,-17.2991605,\
                       -44.7586581,-63.9201063])
    # And calculation of the saturation pressure
    return rhoc*np.exp(powerweight(theta,expo,weight))
# Specific volume of water vapor
def vapor_specific_volume(T):
    """
    Specific volume of water vapor (in [m3/kg]) at the saturated state for a given
    value of temperature in K. Value of the temperature used as argument must be
    between the triple point temperature and the critical one. If not, its value
    is corrected.
    """
    return 1/vapor_density(T)
# Specific enthalpy of the liquid phase
@vectorize
def liquid_enthalpy(T):
    """
    Calculation of the liquid water specific enthalpy (in [kJ/kg]) at the
    saturated state for a given value of the temperature in K. Value of the
    temperature used as argument must be between the triple point temperature
    and the critical one. If not, its value is corrected.
    """
    # Ensure that the temperature value is within the available range of values,
    # i.e. between the triple point temperature and the critical one
    T = _cut(T)
    # Dimensionless parameter used to represent temperature (Warning, this one
    # is different from the others used in previous correlations !).
    theta = T/Tc
    # Experimental parameters given by the IAPWS data, exponents
    expo = np.array([0.0,-19.0,1.0,4.5,5.0,54.5])
    # and relating weights
    weight = np.array([-1135.905627715,-5.65134998e-8,2690.66631,127.287297,\
                       -135.003439,0.981825814])
    # Density
    rho = liquid_density(T)
    # And calculation of the saturation pressure
    return powerweight(theta,expo,weight)+1e+2*T/rho*psat_derivative(T)
# Specific enthalpy of the vapor phase
@vectorize
def vapor_enthalpy(T):
    """
    Calculation of the water vapor specific enthalpy (in [kJ/kg]) at the
    saturated state for a given value of the temperature in K. Value of the
    temperature used as argument must be between the triple point temperature
    and the critical one. If not, its value is corrected.
    """
    # Ensure that the temperature value is within the available range of values,
    # i.e. between the triple point temperature and the critical one
    T = _cut(T)
    # Dimensionless parameter used to represent temperature (Warning, this one
    # is different from the others used in previous correlations !).
    theta = T/Tc
    # Experimental parameters given by the IAPWS data, exponents
    expo = np.array([0.0,-19.0,1.0,4.5,5.0,54.5])
    # and relating weights
    weight = np.array([-1135.905627715,-5.65134998e-8,2690.66631,127.287297,\
                       -135.003439,0.981825814])
    # Density
    rho = vapor_density(T)
    # And calculation of the saturation pressure
    return powerweight(theta,expo,weight)+1e2*T/rho*psat_derivative(T)
# Specific internal energy of the liquid phase
@vectorize
def liquid_internal_energy(T):
    """
    Calculation of the liquid water specific internal energy (in [kJ/kg]) at the
    saturated state for a given value of the temperature in K. Value of the
    temperature used as argument must be between the triple point temperature
    and the critical one. If not, its value is corrected.
    """
    # Ensure that the temperature value is within the available range of values,
    # i.e. between the triple point temperature and the critical one
    T = _cut(T)
    # Dimensionless parameter used to represent temperature (Warning, this one
    # is different from the others used in previous correlations !).
    theta = T/Tc
    # Experimental parameters given by the IAPWS data, exponents
    expo = np.array([0.0,-19.0,1.0,4.5,5.0,54.5])
    # and relating weights
    weight = np.array([-1135.905627715,-5.65134998e-8,2690.66631,127.287297,\
                       -135.003439,0.981825814])
    # Density and pressure
    rho , ps = liquid_density(T) , psat(T)
    # And calculation of the saturation pressure
    return powerweight(theta,expo,weight)+1e+2*(T*psat_derivative(T)-ps)/rho
# Specific internal energy of the vapor phase
@vectorize
def vapor_internal_energy(T):
    """
    Calculation of the water vapor specific internal energy (in [kJ/kg]) at the
    saturated state for a given value of the temperature in K. Value of the
    temperature used as argument must be between the triple point temperature
    and the critical one. If not, its value is corrected.
    """
    # Ensure that the temperature value is within the available range of values,
    # i.e. between the triple point temperature and the critical one
    T = _cut(T)
    # Dimensionless parameter used to represent temperature (Warning, this one
    # is different from the others used in previous correlations !).
    theta = T/Tc
    # Experimental parameters given by the IAPWS data, exponents
    expo = np.array([0.0,-19.0,1.0,4.5,5.0,54.5])
    # and relating weights
    weight = np.array([-1135.905627715,-5.65134998e-8,2690.66631,127.287297,\
                       -135.003439,0.981825814])
    # Density and pressure
    rho , ps = vapor_density(T) , psat(T)
    # And calculation of the saturation pressure
    return powerweight(theta,expo,weight)+1e+2*(T*psat_derivative(T)-ps)/rho
# Drawing of some interesting graphs
def draw_mollier_diagram(N=200):
    """
    Draw of the water Mollier (p,h) diagram in a semi log y coordinates graph.
    """
    T = np.linspace(Tt,Tc,N)
    pl.semilogy(liquid_enthalpy(T),psat(T),'b-',lw=1.5)
    pl.semilogy(vapor_enthalpy(T),psat(T),'b-',lw=1.5)
    pl.xlabel(r'Specific enthalpy in ${\rm kJ/kg}$',size=18)
    pl.ylabel(r'Pressure in ${\rm bar}$',size=18)
    pl.ylim(ymin=pt)
    pl.show()
def draw_mollier_point(T,lines=False):
    """
    Draw the point corresponding to temperature "T" on the Mollier diagram.
    """
    if lines:
        pl.semilogy([liquid_enthalpy(Tt),vapor_enthalpy(T),\
                     vapor_enthalpy(T)],[psat(T),psat(T),pt],'k--')
        pl.semilogy([liquid_enthalpy(T),liquid_enthalpy(T)],[psat(T),pt],'k--')
    pl.scatter([liquid_enthalpy(T),vapor_enthalpy(T)],[psat(T)]*2,s=50,\
            marker='o')
    pl.xlim(xmin=0)
    pl.ylim(ymin=pt)
def draw_clapeyron_diagram(N=500):
    """
    Draw of the water Clapeyron (p,v) diagram in a semi log x coordinates graph.
    """
    T = np.logspace(np.log10(Tt),np.log10(Tc),N)
    pl.semilogx(liquid_specific_volume(T),psat(T),'b-',lw=1.5)
    pl.semilogx(vapor_specific_volume(T),psat(T),'b-',lw=1.5)
    pl.xlabel(r'Specific volume in ${\rm m}^3/{\rm kg}$',size=18)
    pl.ylabel(r'Pressure in ${\rm bar}$',size=18)
    pl.show()
def draw_clapeyron_point(T,lines=False):
    """
    Draw the point corresponding to temperature "T" on the Clapeyron diagram.
    """
    if lines:
        pl.semilogx([liquid_specific_volume(Tt),vapor_specific_volume(T),\
                     vapor_specific_volume(T)],[psat(T),psat(T),pt],'k--')
        pl.semilogx([liquid_specific_volume(T),liquid_specific_volume(T)],\
                    [psat(T),pt],'k--')
    pl.scatter([liquid_specific_volume(T),vapor_specific_volume(T)],[psat(T)]*2,s=50,\
            marker='o')
    pl.ylim(xmin=liquid_specific_volume(Tt))
    pl.ylim(ymin=pt)
