#!/usr/bin/env python
# coding: utf-8

# # XMCD Beamline V 0.1 - sketch of a first test
# 
# The following is a first test for the XMCD Beamline
# 
# For the moment it hopefully works for 3d transition metals with $O_h$ symmetry



import numpy as np
import subprocess
from math import *
import matplotlib.pyplot as plt
import json
from IPython.display import display, Math
from tabulate import tabulate
from scipy.integrate import romb

# ###  dictionary with initial parameters as taken from M.W.Haverkort's PhD Thesis (p.156)

parameters = {
    "Cu": {
        "Nelec": 9,
        "zeta_3d": 0.102,
        "F2dd": 12.854,
        "F4dd": 7.980,
        "zeta_2p": 13.498,
        "F2pd": 8.177,
        "G1pd": 6.169,
        "G3pd": 3.510,
        "Xzeta_3d": 0.124,
        "XF2dd": 13.611,
        "XF4dd": 8.457
    },
    "Ni": {
        "Nelec": 8,
        "zeta_3d": 0.083,
        "F2dd": 12.233,
        "F4dd": 7.597,
        "zeta_2p": 11.507,
        "F2pd": 7.720,
        "G1pd": 5.783,
        "G3pd": 3.290,
        "Xzeta_3d": 0.102,
        "XF2dd": 13.005,
        "XF4dd": 8.084
    },
    "Co": {
        "Nelec": 7,
        "zeta_3d": 0.066,
        "F2dd": 11.604,
        "F4dd": 7.209,
        "zeta_2p": 9.748,
        "F2pd": 7.259,
        "G1pd": 5.394,
        "G3pd": 3.068,
        "Xzeta_3d": 0.083,
        "XF2dd": 12.395,
        "XF4dd": 7.707
    },
    "Fe": {
        "Nelec": 6,
        "zeta_3d": 0.052,
        "F2dd": 10.965,
        "F4dd": 6.815,
        "zeta_2p": 8.200,
        "F2pd": 6.792,
        "G1pd": 5.000,
        "G3pd": 2.843,
        "Xzeta_3d": 0.067,
        "XF2dd": 11.778,
        "XF4dd": 7.327
    },
    "Mn": {
        "Nelec": 5,
        "zeta_3d": 0.040,
        "F2dd": 10.315,
        "F4dd": 6.413,
        "zeta_2p": 6.846,
        "F2pd": 6.320,
        "G1pd": 4.603,
        "G3pd": 2.617,
        "Xzeta_3d": 0.053,
        "XF2dd": 11.154,
        "XF4dd": 6.942
    },
     "Cr": {
        "Nelec": 4,
        "zeta_3d": 0.030,
        "F2dd": 9.648,
        "F4dd": 6.001,
        "zeta_2p": 5.668,
        "F2pd": 5.840,
        "G1pd": 4.201,
        "G3pd": 2.387,
        "Xzeta_3d": 0.041,
        "XF2dd": 10.521,
        "XF4dd": 6.551
    },
    "V": {
        "Nelec": 3,
        "zeta_3d": 0.022,
        "F2dd": 8.961,
        "F4dd": 5.576,
        "zeta_2p": 4.650,
        "F2pd": 5.351,
        "G1pd": 3.792,
        "G3pd": 2.154,
        "Xzeta_3d": 0.031,
        "XF2dd": 9.875,
        "XF4dd": 6.152
    },
    "Ti": {
        "Nelec": 2,
        "zeta_3d": 0.016,
        "F2dd": 8.243,
        "F4dd": 5.132,
        "zeta_2p": 3.776,
        "F2pd": 4.849,
        "G1pd": 3.376,
        "G3pd": 1.917,
        "Xzeta_3d": 0.023,
        "XF2dd": 9.21,
        "XF4dd": 5.744
    },
    "Sc": {
        "Nelec": 1,
        "zeta_3d": 0.010,
        "F2dd": 0,
        "F4dd": 0,
        "zeta_2p": 3.032,
        "F2pd": 4.332,
        "G1pd": 2.950,
        "G3pd": 1.674,
        "Xzeta_3d": 0.017,
        "XF2dd": 8.530,
        "XF4dd": 5.321
    }
}

def get_atomic_params(ion, charge):
    
    f = open('parameters.json','r',encoding='utf-8')
    xdat = json.load(f, )
    if str(int(charge[0]) - 2) != 0 :
        conf = '3d'+ str(parameters[ion]['Nelec']-int(charge[0]) + 2)
        conf_xas = '2p5,3d'+ str(parameters[ion]['Nelec']-int(charge[0]) + 2 + 1)
    else: 
        conf = '3d'+ str(parameters[ion]['Nelec'])
        conf_xas = '2p5,3d'+ str(parameters[ion]['Nelec']+1)
    
    print(conf, conf_xas)
    indict = xdat['elements'][ion]['charges'][charge]['configurations'][conf]['terms']['Atomic']['parameters']['variable']
    findict = xdat['elements'][ion]['charges'][charge]['configurations'][conf_xas]['terms']['Atomic']['parameters']['variable']
    return [indict,findict]

# ### Given the details of the system, create a Quanty input file

def XAS_Qty(Label, ion, beta, tenDq, B, X, T ,params, charge = '2+', lmct=False, lmct_params=None):# 
    """
    Creates a Quanty input based on the template XAS_Template.lua 
    with the name Label.lua
    
    Arguments:
        Label  : Name of the system (string)
        ion    : The 3d metal (string)
        beta   : Scaling factor (float)
        tenDq     : 10Dq parameter (float) ! for Oh, for the moment
        B      : Magnetic field vector [Tesla] ([3x1] array or list)
        X      : Exchange field vector [Tesla] ([3x1] array or list)
        T.     : Temperature [Kelvin] (float)
        params : Dictionary of parameters
    """
    param_in, param_out = get_atomic_params(ion, charge)

    # Get info from parameters:
    Nelec   = params[ion]['Nelec']-int(charge[0]) + 2
    zeta_3d = param_in['ζ(3d)']
    F2dd    = param_in['F2(3d,3d)']
    F4dd    = param_in['F4(3d,3d)']
    zeta_2p = param_out['ζ(2p)']
    F2pd    = param_out['F2(2p,3d)']
    G1pd    = param_out['G1(2p,3d)']
    G3pd    = param_out['G3(2p,3d)']
    Xzeta_3d= param_out['ζ(3d)']
    XF2dd   = param_out['F2(3d,3d)']
    XF4dd   = param_out['F4(3d,3d)']
    
    #scaling:
    F2dd=beta*F2dd
    F4dd=beta*F4dd 
    F2pd=beta*F2pd
    G1pd=beta*G1pd
    G3pd=beta*G3pd 
    XF2dd=beta*XF2dd
    XF4dd=beta*XF4dd
    
    
    Npsi = factorial(10) / (factorial(Nelec) * factorial(10-Nelec))
    
    Verb = 0
    H_atomic = 0
    H_crystal_field = 1

    deltaL1_i = '3' #eV
    deltaL1_f = '2' #eV
    VegL1_i = '2' #eV
    VegL1_f =  VegL1_i
    Vt2gL1_i = '1' 
    Vt2gL1_f = Vt2gL1_i
    tenDqL1_i = '0.7'
    tenDqL1_f = tenDqL1_i

    deltaL2_i = '3' #eV
    deltaL2_f = '2' #eV
    VegL2_i = '2' #eV
    VegL2_f =  VegL2_i
    Vt2gL2_i = '1' 
    Vt2gL2_f = Vt2gL2_i
    tenDqL2_i = '0.7'
    tenDqL2_f = tenDqL2_i

    if lmct:
        H_3d_lig_hyb_lmct = '1'
        if lmct_params==None:
            print('Using default lmct parameters')

        else:
            deltaL1_i = str(lmct_params['deltaL1_i'])
            deltaL1_f = str(lmct_params['deltaL1_f'])
            VegL1_i   = str(lmct_params['VegL1_i'])
            VegL1_f   = str(lmct_params['VegL1_f'])
            Vt2gL1_i  = str(lmct_params['Vt2gL1_i'])
            Vt2gL1_f  = str(lmct_params['Vt2gL1_f'])
            tenDqL1_i = str(lmct_params['tenDqL1_i'])
            tenDqL1_f = str(lmct_params['tenDqL1_f'])

            deltaL2_i = str(lmct_params['deltaL2_i'])
            deltaL2_f = str(lmct_params['deltaL2_f'])
            VegL2_i   = str(lmct_params['VegL2_i'])
            VegL2_f   = str(lmct_params['VegL2_f'])
            Vt2gL2_i  = str(lmct_params['Vt2gL2_i'])
            Vt2gL2_f  = str(lmct_params['Vt2gL2_f'])
            tenDqL2_i = str(lmct_params['tenDqL2_i'])
            tenDqL2_f = str(lmct_params['tenDqL2_f'])
    else:
        H_3d_lig_hyb_lmct = '0'
    H_3d_lig_hyb_mlct = 0
    H_Bfield = 1
    H_Xfield = 1
    
    # Read in the file
    with open('XAS_Template.lua', 'r') as file :
        filedata = file.read()

    # Replace the target string
    # Hamiltonian properties:
    filedata = filedata.replace('$Verbosity', '0')
    filedata = filedata.replace('$H_atomic', '1')
    if tenDq == 0:
        print('10Dq is zero: disable that part of the Hamiltonian')
        filedata = filedata.replace('$H_crystal_field', '0')
    else:
        filedata = filedata.replace('$H_crystal_field', '1')
    filedata = filedata.replace('$H_3d_ligands_hybridization_lmct', H_3d_lig_hyb_lmct)
    filedata = filedata.replace('$H_3d_ligands_hybridization_mlct', '0')
    filedata = filedata.replace('$H_magnetic_field', '1')
    filedata = filedata.replace('$H_exchange_field', '1')
    
    # System
    filedata = filedata.replace('$NElectrons_3d', str(Nelec))
    filedata = filedata.replace('$U(3d,3d)_i_value', '0')
    filedata = filedata.replace('$F2(3d,3d)_i_value', str(F2dd))
    filedata = filedata.replace('$F2(3d,3d)_i_scale', '0.8')
    filedata = filedata.replace('$F4(3d,3d)_i_value', str(F4dd))
    filedata = filedata.replace('$F4(3d,3d)_i_scale', '0.8')
    
    filedata = filedata.replace('$U(3d,3d)_f_value', '0')
    filedata = filedata.replace('$F2(3d,3d)_f_value', str(XF2dd))
    filedata = filedata.replace('$F2(3d,3d)_f_scale', '0.8')
    filedata = filedata.replace('$F4(3d,3d)_f_value', str(XF4dd))
    filedata = filedata.replace('$F4(3d,3d)_f_scale', '0.8')
    
    filedata = filedata.replace('$U(2p,3d)_f_value', '0')
    filedata = filedata.replace('$F2(2p,3d)_f_value', str(F2pd))
    filedata = filedata.replace('$F2(2p,3d)_f_scale', '0.8')
    filedata = filedata.replace('$G1(2p,3d)_f_value', str(G1pd))
    filedata = filedata.replace('$G1(2p,3d)_f_scale', '0.8')
    filedata = filedata.replace('$G3(2p,3d)_f_value', str(G3pd))
    filedata = filedata.replace('$G3(2p,3d)_f_scale', '0.8')
    
    filedata = filedata.replace('$zeta(3d)_i_value', str(zeta_3d))
    filedata = filedata.replace('$zeta(3d)_i_scale', '1')
    
    filedata = filedata.replace('$zeta(3d)_f_value', str(Xzeta_3d))
    filedata = filedata.replace('$zeta(3d)_f_scale', '1')
    
    filedata = filedata.replace('$zeta(2p)_f_value', str(zeta_2p))
    filedata = filedata.replace('$zeta(2p)_f_scale', '1')
    
    
    filedata = filedata.replace('$10Dq(3d)_i_value', str(tenDq))
    filedata = filedata.replace('$10Dq(3d)_f_value', str(tenDq))
    
    filedata = filedata.replace('$Delta(3d,L1)_i_value',deltaL1_i)
    filedata = filedata.replace('$Delta(3d,L1)_f_value',deltaL1_f)
    filedata = filedata.replace('$10Dq(L1)_i_value',tenDqL1_i)
    filedata = filedata.replace('$Veg(3d,L1)_i_value',VegL1_i)
    filedata = filedata.replace('$Vt2g(3d,L1)_i_value',Vt2gL1_i)
    filedata = filedata.replace('$10Dq(L1)_f_value',tenDqL1_f)
    filedata = filedata.replace('$Veg(3d,L1)_f_value',VegL1_f)
    filedata = filedata.replace('$Vt2g(3d,L1)_f_value',Vt2gL1_f)
    filedata = filedata.replace('$Delta(3d,L2)_i_value',deltaL2_i)
    filedata = filedata.replace('$Delta(3d,L2)_f_value',deltaL2_f)
    filedata = filedata.replace('$10Dq(L2)_i_value',tenDqL2_i)
    filedata = filedata.replace('$Veg(3d,L2)_i_value',VegL2_i)
    filedata = filedata.replace('$Vt2g(3d,L2)_i_value',Vt2gL2_i)
    filedata = filedata.replace('$10Dq(L2)_f_value',tenDqL2_f)
    filedata = filedata.replace('$Veg(3d,L2)_f_value',VegL2_f)
    filedata = filedata.replace('$Vt2g(3d,L2)_f_value',Vt2gL2_f)
        
    filedata = filedata.replace('$Bx_i_value',str(B[0]))
    filedata = filedata.replace('$By_i_value',str(B[1]))
    filedata = filedata.replace('$Bz_i_value',str(B[2]))
    filedata = filedata.replace('$Bx_f_value',str(B[0]))
    filedata = filedata.replace('$By_f_value',str(B[1]))
    filedata = filedata.replace('$Bz_f_value',str(B[2]))
    
    filedata = filedata.replace('$Hx_i_value',str(X[0]))
    filedata = filedata.replace('$Hy_i_value',str(X[1]))
    filedata = filedata.replace('$Hz_i_value',str(X[2]))
    filedata = filedata.replace('$Hx_f_value',str(X[0]))                                                                
    filedata = filedata.replace('$Hy_f_value',str(X[1]))
    filedata = filedata.replace('$Hz_f_value',str(X[2]))
    
    
    filedata = filedata.replace('$NConfigurations', str(2))
    filedata = filedata.replace('$NPsisAuto', str(1))
    filedata = filedata.replace('$NPsis', str(Npsi))
    filedata = filedata.replace('$T', str(T))

    # Get energy data:
    f = open('parameters.json','r',encoding='utf-8')
    xdat = json.load(f, )
    iondata = xdat['elements'][ion]['charges']['2+']['symmetries']['Oh']['experiments']['XAS']['edges']
    Edge = iondata['L2,3 (2p)']['axes'][0][4]
    Gmin = iondata['L2,3 (2p)']['axes'][0][5][0]
    Gmax = iondata['L2,3 (2p)']['axes'][0][5][1]
    Gamma = iondata['L2,3 (2p)']['axes'][0][6]
    
    filedata = filedata.replace('$Gmin1', str(Gmin))
    filedata = filedata.replace('$Gmax1', str(Gmax))
    filedata = filedata.replace('$Egamma1', str(Edge + 10))
    filedata = filedata.replace('$BaseName', Label+'_')
    filedata = filedata.replace('$k1', '{0,0,1}')
    filedata = filedata.replace('$eps11', '{0,1,0}')
    filedata = filedata.replace('$eps12', '{1,0,0}')
    filedata = filedata.replace('$spectra', "'Isotropic','Circular Dichroism'")
    filedata = filedata.replace('$Eedge1', str(Edge))
    filedata = filedata.replace('$Emin1', str(Edge-10))
    filedata = filedata.replace('$Emax1', str(Edge+30))
    filedata = filedata.replace('$NE1', str(2048))
    filedata = filedata.replace('$Gamma1', str(Gamma))
    filedata = filedata.replace('$DenseBorder', str(2048))
    
    
    # Write the file out again
    with open(Label+'.lua', 'w') as file:
        file.write(filedata)
    
    
    
    


# ### Interface to Quanty

def run_Qty(Label, Qty_path='/Users/Botel001/Programs/Quanty'):
    """
    Runs Quanty with the input file specified by Label.lua, and
    returns the standard output and error (if any)
    
    Arguments:
      Label    : Name of the system (string)
      Qty_path : Path to the Quanty executable (string)
      
    Returns:
      result   : a subprocess CompltedProcess object 
                 result.stdout  is the standard output
                 result.stderr  is the standard error 
    """
    command = [Qty_path, Label+'.lua']
    result = subprocess.run(command, capture_output=True, text=True)
    return result


# ### Read the output of a Quanty calculation and extract expectation values

def treat_output(Rawout):
    """
    From the standar output of a Quanty calculation with the XAS_Template, 
    it extracts the relevant expctation value
    
    Arguments:
      Rawout   : a subprocess CompltedProcess object
    
    Returns:
      A dictionary with the relevant expectation values
      
    """
    out = Rawout.stdout.split('\n')
    rline = 0
    for iline in range(len(out)):
        if '!*!' in out[iline]:
            #print(out[iline-2])
            #print(out[iline])
            rline=iline
    
    Odata = out[rline].split()
    E = Odata[2]
    S2 = Odata[3]
    L2 = Odata[4]
    J2 = Odata[5]
    S_k = Odata[6]
    L_k = Odata[7]
    J_k = Odata[8]
    T_k = Odata[9]
    LdotS = Odata[10]
    return {'E':E,'S2':S2,'L2':L2,'J2':J2,'S_k':S_k,'L_k':L_k,'J_k':J_k,'T_k':T_k,'LdotS':LdotS}



def post_proc(ion, label,outdic):
    xz = np.loadtxt(label+'__iso.spec', skiprows=5)
    mcd =  np.loadtxt(label+'__cd.spec', skiprows=5)
    xl =  np.loadtxt(label+'__l.spec', skiprows=5)
    xr =  np.loadtxt(label+'__r.spec', skiprows=5)
    mcd2 = xr.copy()
    mcd2[:,2]=xl[:,2]-xr[:,2]
    npts = np.shape(xz)[0]
    mcd2 = mcd
    # TOTAL spectra
    xas = xz.copy()
    xas[:,2] = xz[:,2]+xl[:,2]+xr[:,2]

    xas0 = xz.copy()
    xas0[:,2] =  (xl[:,2]+xr[:,2])/2+xl[:,2]+xr[:,2]

    dx = xz.copy()
    dx[:,2] = xl[:,2] + xr[:,2] - 2*xz[:,2]
    # ### Integration using Trapezoidal rule 

    # In[13]:


    nh = 10-parameters[ion]['Nelec']
    
    trapz=False
    if trapz :
        tot = np.trapz(xas[:,2],xas[:,0])
        tot0 = np.trapz(xas0[:,2],xas0[:,0])
        dx0 = np.trapz(dx[:,2], dx[:,0])
    else:
        tot = romb(xas[:,2],dx =(xas[1,0]-xas[0,0]))
        tot0 = romb(xas0[:,2],dx=(xas0[1,0]-xas0[0,0]))
        dx0 = romb(dx[:,2], dx=(dx[1,0]-dx[0,0]))

    
    deltaXas = dx0 /tot
    if trapz:
        
        lz = -2*nh*np.trapz(mcd2[:,2], mcd2[:,0])/tot
        szef = 3/2*nh*(np.trapz(mcd2[0:npts//2,2], mcd2[0:npts//2,0])-2*np.trapz(mcd2[npts//2:,2], mcd2[npts//2:,0]))/tot
        lz0 = -2*nh*np.trapz(mcd2[:,2], mcd2[:,0])/tot0
        szef0 = 3/2*nh*(np.trapz(mcd2[0:npts//2,2], mcd2[0:npts//2,0])-2*np.trapz(mcd2[npts//2:,2], mcd2[npts//2:,0]))/tot0

    else:
        print(len(mcd2[npts//2:,2]), len(mcd2[0:npts//2+1]))
        mydelta=mcd2[1,0]-mcd2[0,0]
        lz = -2*nh*romb(mcd2[:,2],mydelta)/tot
        szef = 3/2*nh*(romb(mcd2[0:npts//2+1,2],mydelta)-2*romb(mcd2[npts//2:,2], mydelta))/tot
        lz0 = -2*nh*romb(mcd2[:,2],mydelta)/tot0
        szef0 = 3/2*nh*(romb(mcd2[0:npts//2+1,2],mydelta)-2*romb(mcd2[npts//2:,2],mydelta))/tot0


    # ### Display

    # In[14]:

    ## Instead of equations, print only the values:
    
    #eq1 = r'\langle L_z \rangle = - 2 \langle N_h \rangle \frac{\int_{L_3 + L_2} (\alpha^{+}-\alpha^{-}) }{\int \alpha} = %6.4f'%(lz) 
    #eq2 = r'\langle S^{eff}_z \rangle = 3/2 \langle N_h \rangle \frac{\int_{L_3} (\alpha^{+}-\alpha^{-}) - 2 \int_{L_2} (\alpha^{+}-\alpha^{-})  }{\int \alpha} = %6.4f'%(szef) 
    #eq3 = r'\langle L_z^{(\mathrm{theo})} \rangle =  %s'%(outdic['L_k']) 
    #eq4 = r'\langle S^{eff(\mathrm{theo})}_z \rangle  = %6.4f'%(float(outdic['S_k'])+float(outdic['T_k'])) 
    #eq5 = r'\langle S^{(\mathrm{theo})}_z \rangle  = %6.4f'%(float(outdic['S_k']))#+float(outdic['T_k'])) 
    #eq6 = r'\langle T^{(\mathrm{theo})}_z \rangle  = %6.4f'%(float(outdic['T_k'])) 

# In[15]:


    #display(Math(eq1))
    #display(Math(eq2))
    #display(Math(eq3))
    #display(Math(eq4))
    #display(Math(eq5))
    #display(Math(eq6))
    tfmt = 'html'

    Lz_t = float(outdic['L_k'])
    Sz_t =  float(outdic['S_k'])
    Tz_t =  float(outdic['T_k']) 
    Seff_t =  float(outdic['S_k'])+float(outdic['T_k']) 
    
    table1 = [[r'L$_z$', r'S$_{eff}$', r'S$_{z}$', r'T$_{z}$'],
        [Lz_t,Seff_t, Sz_t, Tz_t]]

    print("Theoretical values (Quanty):")
    display(tabulate(table1,headers='firstrow', tablefmt=tfmt))

    print("Sum rules :")
    table2 = [[r'sL$_z$', 'sS$_{eff}$'],[lz, szef]]
    display(tabulate(table2,headers='firstrow', tablefmt=tfmt))

    print("Sum rules 0:")
    table3 = [[r's$_0$L$_z$', 's$_0$S$_{eff}$'],[lz0, szef0]]
    display(tabulate(table3,headers='firstrow', tablefmt=tfmt))

    print("Deviations:")
    table4 =[[r'$\Delta$XAS (%)', r'$\Delta$L$_{z}$ (%)', r'$\Delta$S$_{eff}$ (%)',
              r'$\Delta_0$L$_{z}$ (%)', r'$\Delta_0$S$_{eff}$ (%)'],
             [deltaXas*100,100*(abs(Lz_t) -abs(lz))/Lz_t,
              100*(abs(Seff_t)-abs(szef))/Seff_t,
              100*(abs(Lz_t) -abs(lz0))/Lz_t,
              100*(abs(Seff_t)-abs(szef0))/Seff_t  ]]
    display(tabulate(table4,headers='firstrow', tablefmt=tfmt))
    
    figs, ax = plt.subplots(2, sharex=True, figsize=(8,8))
    figs.suptitle(label, fontsize=16)
    ax[0].plot(xz[:,0], xz[:,2], 'r--',label='z-pol')
    ax[0].plot(xl[:,0], xl[:,2], 'b', label='left')
    ax[0].plot(xr[:,0], xr[:,2], 'g',label='right')
    ax[0].set_xlim(-10,20)
    ax[0].legend(fontsize=14)
    ax[0].set_ylabel('Intensity (a.u.)', fontsize=16)

    ax[1].plot(xas[:,0], xas[:,2]/3, 'k', label='average')
    ax[1].plot(mcd[0:npts//2,0], mcd[0:npts//2,2],'r', label= r'L$_3$')
    ax[1].plot(mcd[npts//2:,0], mcd[npts//2:,2],'b', label= r'L$_2$')
    ax[1].set_xlim(-10,20)
    ax[1].legend(fontsize=14)
    ax[1].set_ylabel('Intensity (a.u.)', fontsize=16)
    ax[1].set_xlabel('Energy (eV)', fontsize=16)

    plt.show()






