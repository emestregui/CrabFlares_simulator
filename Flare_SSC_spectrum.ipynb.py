#!/usr/bin/env python
# coding: utf-8

# In[1]:

#get_ipython().magic(u'matplotlib inline')
import naima 
from astropy import table
import numpy as np
import math
from astropy.io import ascii
from astropy.constants import c,m_e
import astropy.units as u
from naima.models import (ExponentialCutoffPowerLaw, Synchrotron,
                          InverseCompton)
#import matplotlib
#matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from ipywidgets.widgets.interaction import interact, fixed
from matplotlib.backends.backend_pdf import PdfPages
from scipy.optimize import minimize
from astropy.table import Table, Column
import csv

#filename = '/Users/guest1/Desktop/gettingstarted/flare/'
#C:\Users\enriq\OneDrive\Escritorio\Tesis\final_flare_simuls
flarename = '2011'
mcc = (m_e * c**2).cgs
crab_distance=  2.2 * u.kpc
amp0=2.e42
make_plot = True
make_tables = False
syn_max_0 = 200 * u.MeV
tvar = 6 * u.h

nloops = 4
sizeloops = 20

comovf = False
##
def lorentzfactor(B=140*u.mG):
    return (10*(syn_max_0.to('MeV')/(600*u.MeV))**-1 * (tvar.to('s')/(4e4*u.s))**-2 * B.to('mG').value**-3)

def eprimpev(syn_max_0,tvar,delta=1.0):
    return ((syn_max_0.to('MeV')/(600*u.MeV))**(2./3.) * (tvar.to('s')/(4e4*u.s))**(1./3.) * (delta/10.0)**(-1./3.))*u.PeV

#analytic estimate for the cutoff energy
def ecutoff(index,B,syn_max):
    Bint=((B).to(u.Gauss)).value
    #print('Esto es Ep:', (((syn_max/mcc)/((-index+7./2.)**3*4./27.*3.4e-14*Bint*4./3.))**0.5*mcc).cgs)
    return (((syn_max/mcc)/((-index+7./2.)**3*4./27.*3.4e-14*Bint*4./3.))**0.5*mcc).cgs

def cutoffinitval(index,B,init_ecut = 1 * u.PeV,index_ref = 1.5,b_ref = 1.99 * u.mG):
    correcut_factor = (init_ecut.to('PeV').value/((3.5-index_ref)**(-3./2.)*(b_ref.to('mG').value)**(-0.5)))
    ecut_scl = correcut_factor * (3.5-index)**(-3./2.)*(B.to('mG').value)**(-0.5)
    return ecut_scl * u.PeV

############################# Definitions for E_cutoff, amplitude fittings

def log_likelihood(e_cutoff, data, amplitude, alpha, B, Eemin, Eemax):
    crab_distance=  2.2 * u.kpc
    ECPL = ExponentialCutoffPowerLaw(amplitude=amplitude / u.eV,
                                     e_0=1 * u.TeV,
                                     alpha=alpha,
                                     e_cutoff=abs(e_cutoff) * u.erg)
    SYN = Synchrotron(ECPL, B=B, Eemin=Eemin, Eemax=Eemax)
    model=SYN.sed(data['energy'].quantity,distance=crab_distance)
    sigma=(data['flux_error_lo'].quantity+data['flux_error_hi'].quantity)/2
    sigma2=np.where(sigma != 0, sigma, np.ones_like(sigma))    
    loglik = np.sum(np.log((model.value - data['flux'].data)**2))
    return loglik - 2*np.sum(np.log(sigma2))


def trylik(e_cutoff, data, amplitude, alpha, B, Eemin, Eemax):
    nll = lambda *args: -log_likelihood(*args)
    initial = e_cutoff 
    soln = minimize(nll, initial.value, args=(data, amplitude, alpha, B, Eemin, Eemax))
    m_ecut = soln.x
    llkh = log_likelihood(m_ecut, data, amplitude, alpha, B, Eemin, Eemax)
    return llkh

def tryecut(ecut_arr, data, amplitude, alpha, B, Eemin, Eemax):
    liktrial = np.zeros(len(ecut_arr))
    for i in range(len(ecut_arr)):
        ECPL = ExponentialCutoffPowerLaw(amplitude=amplitude/ u.eV,
                                    e_0=1 * u.TeV,
                                    alpha=alpha,
                                    e_cutoff=ecut_arr[i])
        SYN = Synchrotron(ECPL, B=B, Eemin=Eemin, Eemax=Eemax)
        amp_cor=fitfactor(data,SYN)
        liktrial[i] = trylik(ecut_arr[i], data, amplitude*amp_cor, alpha, B, Eemin, Eemax)
      
    result = ecut_arr[np.where(liktrial == np.min(liktrial))]
    #print(liktrial)
    dof = len(np.asarray(data['flux'].data))
    if np.min(liktrial) > dof:
        print('Warning: Minimum Log-likelihood > degrees of freedom')
    return result 

###########################################################################
    
# analytic fitting for the spectrum normalization
def fitfactor(data,spectrum):
    #one should be careful here: in data_flare.flux corresponds to nuFnu
    model=spectrum.sed(data['energy'].quantity,distance=crab_distance)
    flux_error=(data['flux_error_lo'].quantity+data['flux_error_hi'].quantity)/2
    flux_error2=np.where(flux_error != 0, flux_error, np.ones_like(flux_error))
    s1=np.where(flux_error != 0, model**2/flux_error2**2, np.zeros_like(flux_error))
    sum1=sum(s1)
    s2=np.where(flux_error != 0, model*data['flux']/flux_error2**2, np.zeros_like(flux_error))
    sum2=sum(s2)
    return (sum2/sum1)

############### Final fitting for the Cutoff energy

def fitcutoff(e_cutoff, data, amplitude, alpha, B, Eemin, Eemax):
    nll = lambda *args: -log_likelihood(*args)
    initial = e_cutoff 
    soln = minimize(nll, initial.value, args=(data, amplitude, alpha, B, Eemin, Eemax))
    m_ecut = soln.x
    print("Initial value: ", initial.to(u.PeV))
    llkh = log_likelihood(m_ecut, data, amplitude, alpha, B, Eemin, Eemax)
    dof = len(np.asarray(data['flux'].data))
    print("Maximum likelihood estimates: ", soln.nfev, " iterations", llkh, "Log-likelihood", dof, 'dof')
    if llkh > dof:
        print('Warning: Final Log-likelihood > degrees of freedom')
    #print("m = ", m_ecut * u.erg)
    return(m_ecut * u.erg)

def flare_rad(index,LE_cutoff,Ee_syn_max,B_flare,Ecut_0):
    ECPL = ExponentialCutoffPowerLaw(amplitude=amp0/ u.eV,
                                        e_0=1 * u.TeV,
                                        alpha=index,
                                        e_cutoff=Ecut_0)
    SYN = Synchrotron(ECPL, B=B_flare, Eemin=LE_cutoff, Eemax=Ee_syn_max)
    amp_cor=fitfactor(data_flare,SYN) # Fit particle distribution prefactor
    # Fit particle cutoff, with analytic initial value
    Ecut_flare = fitcutoff(Ecut_0, data_flare, amp0*amp_cor, index, B_flare, LE_cutoff, Ee_syn_max)
    print('Correct ecut: ', Ecut_flare.to(u.PeV))
    # Final particle spectrum and synchrotron
    ECPL = ExponentialCutoffPowerLaw(amplitude=amp0*amp_cor/ u.eV,
                                        e_0=1 * u.TeV,
                                        alpha=index,
                                        e_cutoff=Ecut_flare)
    SYN = Synchrotron(ECPL, B=B_flare, Eemin=LE_cutoff, Eemax = Ee_syn_max)
    if flarename == '2011':
        Rflare = 2.8e-4 * u.pc   ## 3.2 * u.pc, 2.8e-4, 1.7e-4
    elif flarename == '2013':
        Rflare = 1.7e-4 * u.pc
    else:
        Rflare == 3.2 * u.pc
    Esy = np.logspace(0.0, 12, 100) * u.eV #np.exp(14)
    Lsy = SYN.flux(Esy, distance=0 * u.cm)  # use distance 0 to get luminosity
    phn_sy = Lsy / (4 * np.pi * Rflare**2 * c) * 2.24
    fields =['CMB', ['FIR', 70 * u.K, 0.5 * u.eV / u.cm**3], ['NIR', 5000 * u.K, 1 * u.eV / u.cm**3],['SSC', Esy, phn_sy]]
    IC=InverseCompton(ECPL,seed_photon_fields=fields, Eemin=LE_cutoff,Eemax=Ee_syn_max)
    We = IC.compute_We(Eemin=1 * u.TeV)
    print('Estoy aqui: ', We)
    return SYN,IC,amp_cor,We,Ecut_flare


################################################ Start
### Paper models
#index_models = np.linspace(1.0, 3.0, 21)
index_models = np.asarray([1.5])

B_flare_models = np.concatenate((10**(np.linspace(math.log10(50.0), math.log10(5000.0),11)),np.asarray([1000,100,10,3])))
#B_flare_models = 10**(np.linspace(math.log10(50.0), math.log10(5000.0),11)) * u.uG

B_flare_models = np.sort(B_flare_models) * u.uG

########

#index_models = np.concatenate((np.asarray([1.0]),np.linspace(1.6,1.9,4),np.asarray([2.5]),),axis=0)
#index_models = np.asarray([2.5])
#B_flare_models = 10**(np.linspace(math.log10(50.0), math.log10(5000.0),11)) * u.uG
#B_flare_models = np.asarray(np.concatenate((B_flare_models[[0]],B_flare_models.value[7:11]),axis=0)) * u.uG
B_flare_models = np.asarray([100.0]) * u.uG

with open('total_energy_' + flarename + '_' + str(len(index_models)*len(B_flare_models)) + 'models.csv', 'w') as totenergtab:
    writer_totenerg = csv.writer(totenergtab)
    if comovf == False:
        writer_totenerg.writerows([str('B_microG'),str('Index'),str('We_ergs'),str('Ep.erg.'),str('Amplitude.1.eV')])
    else:
        writer_totenerg.writerows([str('B_microG'),str('Index'),str('We_ergs'),str('Ep.erg.'),str('sigma10'),str('Rpc')])
    for m in range(len(np.asarray(B_flare_models.value))):
        for n in range(len(index_models)):
            if comovf == False:
                data_flare = ascii.read('CrabNebula_spectrum_flare' + flarename + '_fermi.ecsv.txt')
            else:
                data_flare = ascii.read('CrabNebula_spectrum_flare' + flarename + '_fermi.ecsv.txt')
                delta = lorentzfactor(B=B_flare_models[m])
                print('Lorentz factor required: ', delta)
                data_flare['energy'] = data_flare['energy'] / delta
                
            #A nalytic estimate: cutoffinitval(index_models[n],B_flare_models[m])#ecutoff(index_models[n],B_flare_models[m],syn_max_0)
            minindex = index_models[n] #np.min(index_models)
            maxindex = index_models[n] #np.max(index_models)
            minbfield = np.min(B_flare_models)
            maxbfield = np.max(B_flare_models)
            minfact = 0.2
            maxfact = 1.2
            ecut_0_arr =  np.logspace(np.log10(minfact * cutoffinitval(index = minindex, B = maxbfield).value), maxfact * np.log10(cutoffinitval(index = maxindex, B = minbfield).value),sizeloops) * u.PeV

            if comovf == False:
                ecut_0_arr = ecut_0_arr.to('erg')
            else:
                ecut_0_arr = ecut_0_arr.to('erg')  * 0.1 *(B_flare_models[m].to('mG').value *((syn_max_0.to('MeV').value/600) * (tvar.to('s').value/4e4)))
            for k in range(nloops):
                if k == 0:
                    ecut_i_arr = ecut_0_arr
                    res = ecut_0_arr
                    nbin = sizeloops
                    print('Executing realization: ', k+1, 'searching from', min(ecut_i_arr.to('PeV')), 'to ',  max(ecut_i_arr.to('PeV')))
                if k > 0:
                    if int(np.where(res == initial_ecut)[0]) == 0:
                        lolim = res[0] - abs(res[1]-res[0])*2.0
                        hilim = res[int(np.where(res == initial_ecut)[0])+1]
                    elif int(np.where(res == initial_ecut)[0]) == int(len(res)-1):
                        lolim = res[int(np.where(res == initial_ecut)[0])-1]
                        hilim = res[int(len(res)-1)] + abs(res[int(len(res)-1)] - res[int(len(res)-2)])*2.0
                    else:
                        lolim = res[int(np.where(res == initial_ecut)[0])-1]
                        hilim = res[int(np.where(res == initial_ecut)[0])+1]
                    nbin = int(1 + (sizeloops//(k+1)))
                    ecut_i_arr =  np.linspace(lolim.value,hilim.value,nbin) * u.erg
                    res = ecut_i_arr
                    print('Executing realization: ', k+1, 'searching from', min(ecut_i_arr.to('PeV')), 'to ',  max(ecut_i_arr.to('PeV')))
    
                                  
                initial_ecut = tryecut(ecut_i_arr,data_flare, amp0, index_models[n], B_flare_models[m], Eemin = 50 * u.GeV, Eemax = 15000 * np.log10(cutoffinitval(index = maxindex, B = minbfield).value) * u.PeV) 
                
            print([B_flare_models[m],index_models[n],initial_ecut])

            SYN_f,IC_f,amp_cor,We,Ecut_flare=flare_rad(index=index_models[n],LE_cutoff=50.0*u.GeV,Ee_syn_max=15000*initial_ecut.to('PeV'),B_flare=B_flare_models[m],Ecut_0=initial_ecut) #50 * u.PeV   20*ecut_0
            if comovf == True:
                eprimpev_ =  eprimpev(syn_max_0,tvar,delta)
                print('Predicted E^prime_PeV',eprimpev_)
            print('Amplitude correction: ', amp_cor, amp0*amp_cor/ u.eV)
            try:
                err_ep = res[int(np.where(abs(Ecut_flare-res) == min(abs(Ecut_flare-res)))[0]) + 1] - res[int(np.where(abs(Ecut_flare-res) == min(abs(Ecut_flare-res)))[0])]
            except:
                err_ep = abs(res[int(np.where(abs(Ecut_flare-res) == min(abs(Ecut_flare-res)))[0]) - 1] - res[int(np.where(abs(Ecut_flare-res) == min(abs(Ecut_flare-res)))[0])])

            print('Energy cutoff: ',Ecut_flare, '+-', str(err_ep/2.))
            print('Total energy in electrons above 1 TeV: We = ',We.value, ' erg')

            energy = np.logspace(-7, 15, 100) * u.eV

            #synp = SYN_f.sed(energy, crab_distance).value
            #print('Synchrotron maximum energy', energy[int(np.where(synp == np.max(synp))[0])].to('MeV'))
            #print(energy[energy.value > 1e6].to('MeV'))
            
            # In[7]:
            if make_plot == True:
                figure, ax = plt.subplots(1, 1, figsize=(10,5))
                data_steady = ascii.read('CrabNebula_spectrum.ecsv.txt')
                naima.plot_data(data_steady, e_unit=u.eV,figure=figure)
                naima.plot_data(data_flare, e_unit=u.eV,figure=figure)

                ax.loglog(energy,SYN_f.sed(energy, crab_distance),lw=3, c='r', label='SYN')

                ax.loglog(energy, IC_f.sed(energy, crab_distance, seed='CMB') + IC_f.sed(energy, crab_distance, seed='FIR') +
                IC_f.sed(energy, crab_distance, seed='NIR') + IC_f.sed(energy, crab_distance, seed='SSC'),
                lw=3, c='r', ls='--',label='IC')
                                
                for i, seed, ls in zip(range(4), ['CMB', 'FIR', 'NIR','SSC'], [':', '-.', ':','--']):
                    ax.loglog(energy, IC_f.sed(energy, crab_distance, seed=seed),
                              lw=3, c=naima.plot.color_cycle[i + 1], label=seed, ls=ls)
                
                #Sensitivity curves
                #hess_sens = ascii.read('HESS_sens_10h.ecsv.txt')
                #ax.loglog(hess_sens['energy'],hess_sens['flux'],lw=3, c='darkmagenta', label='H.E.S.S. 10h', linestyle='dashed')
                CTAsN = ascii.read('CTA_prod3b_v2_North_20deg_05h_DiffSens.txt') ##CTA_N_sens_5h.ecsv.txt for requierements
                ax.loglog(1e12*CTAsN['energy'],CTAsN['flux'],lw=3, c='darkmagenta', label='CTA-N 5h')
                
                ax.set_ylim(1e-24, 9e-8) ## 1e-24, 3e-8
                ax.set_xlim(1e0, 1e15) ##1e3 1e15
                ax.legend(loc='lower left', frameon=False, fontsize=20)
                ax.tick_params(axis='x', labelsize=22)
                ax.tick_params(axis='y', labelsize=22)
                ax.xaxis.label.set_size(22)
                ax.yaxis.label.set_size(22)
                figure.tight_layout()
                
                pdf_name = flarename + '_flare_' + str(B_flare_models[m]) + '_' + str(index_models[n]) + '.pdf'
                with PdfPages(pdf_name) as pdf:
                    pdf.savefig(figure)
                plt.show()


            if make_tables == True:
                table = Table()

                table['Energy'] = Column(energy.to('TeV').value, unit = 'TeV')
                table['IC_flux_CMB'] = Column(IC_f.sed(energy, crab_distance, seed='CMB').to('cm-2 s-1 TeV').value, unit = 'cm-2 s-1 TeV')
                table['IC_flux_FIR'] = Column(IC_f.sed(energy, crab_distance, seed='FIR').to('cm-2 s-1 TeV').value, unit = 'cm-2 s-1 TeV')
                table['IC_flux_NIR'] = Column(IC_f.sed(energy, crab_distance, seed='NIR').to('cm-2 s-1 TeV').value, unit = 'cm-2 s-1 TeV')
                table['IC_flux_SSC'] = Column(IC_f.sed(energy, crab_distance, seed='SSC').to('cm-2 s-1 TeV').value, unit = 'cm-2 s-1 TeV')
            
            
                # write it
                with open('./table_models/' + flarename + '/IC_sedTable'+ '_' + str(round(B_flare_models.value[m],1)) + 'uG_' + str(round(index_models[n],1)) + '.csv', 'w') as csvfile:
                    writer = csv.writer(csvfile)
                    [writer.writerow(r) for r in table]

                # In[ ]:

                table2 = Table()

                table2['Energy'] = Column(energy.to('TeV').value, unit = 'TeV')
                table2['SYN_flux'] = Column(SYN_f.sed(energy, crab_distance).to('cm-2 s-1 TeV').value, unit = 'cm-2 s-1 TeV')

                with open('./table_models/' + flarename + '/SYN_sedTable'+ '_' + str(round(B_flare_models.value[m],1)) + 'uG_' + str(round(index_models[n],1)) + '.csv', 'w') as csvfile:
                    writer = csv.writer(csvfile)
                    [writer.writerow(r) for r in table2]
            if comovf == True:
                rcm = 3e15 * (eprimpev_.value * B_flare_models[m].to('mG').value**-1) * u.cm
                rpc = float(rcm.to('pc').value)
                print('Gyro radius [pc]: ', rpc)
            if comovf == True:
                writer_totenerg.writerow([B_flare_models.value[m],index_models[n],float(We.value),float(Ecut_flare.value),delta/10.0,rpc]) #,sigma10,rpc
            else:
                writer_totenerg.writerow([B_flare_models.value[m],index_models[n],float(We.value),float(Ecut_flare.value),amp0*amp_cor])
            
totenergtab.close()
