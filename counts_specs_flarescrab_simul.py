import matplotlib.pyplot as plt
import astropy.units as u
energy = 1 * u.TeV
import numpy as np
import time
from astropy.io import fits
from gammapy.scripts.cta_utils import CTAObservationSimulation
from gammapy.scripts import CTAPerf
from gammapy.spectrum.models import LogParabola
from gammapy.spectrum.models import PowerLaw
from gammapy.spectrum.models import ExponentialCutoffPowerLaw
from gammapy.scripts.cta_utils import Target
from gammapy.scripts.cta_utils import ObservationParameters
from gammapy.spectrum import SpectrumFit
from gammapy.spectrum.results import SpectrumFitResult
from gammapy.background import FOVCube
from scipy.stats import norm
from gammapy.spectrum.models import TableModel

N_simul = 10000


# Observation parameters
alpha = 0.2 * u.Unit('')
livetime = 10.0 * u.h
emin = 0.02 * u.TeV
emax = 300.0 * u.TeV
obs_param = ObservationParameters(
                                  alpha=alpha, livetime=livetime,
                                  emin=emin, emax=emax
                                  )
filename = 'North_5h.fits'
cta_perf = CTAPerf.read(filename)
#cta_perf.peek()


# Define Spectral Model and Parameters    ################# WARNING, parameters must be in Tev #########################

spectype = 'LogParabola'    ### 'LogParabola', 'PowerLaw', 'ExponentialCutoffPowerLaw'

index= 2.39 * u.Unit('') ### HEGRA: 2.62  HESS: 2.39      ### Needed with:  'PowerLaw', 'ExponentialCutoffPowerLaw'
lambda_= 0.07 * u.Unit('TeV-1')            ### HESS: 0.07 ## MAGIC: 0.05263 (1/Ecutoff)      ### Needed with: 'ExponentialCutoffPowerLaw'
amplitude= 3.23e-11  * u.Unit('cm-2 s-1 TeV-1') ### HEGRA: 2.83e-11 HESS: 3.76e-11 MAGIC: 3.23e-11  HESSII: 1.79e-10 ### Needed for: All models
reference= 1.0 * u.TeV     ### HEGRA, HESS, MAGIC: 1        HESSII: 0.521    ### Needed for: All models
alphapar= 2.47 * u.Unit('')  ### MAGIC: 2.47 HESSII: 2.1      ### Needed for: 'LogParabola'
beta= 0.104 * u.Unit('')   ### MAGIC: 0.24/np.log(10) HESSII: 0.24          ### Needed for: 'LogParabola'

#######################################################################

######## Define Flare model

flare_index = 3.161 * u.Unit('')
flare_amplitude = 3.320e-11 * u.Unit('cm-2 s-1 TeV-1')
flare_reference = 1.25 * u.TeV
lambda_flare = 0.0 * u.Unit('TeV-1')

flare_model = ExponentialCutoffPowerLaw(index=flare_index, amplitude=flare_amplitude, reference=flare_reference,lambda_ = lambda_flare)


if spectype == 'LogParabola':
    neb_model = LogParabola(amplitude=amplitude, reference=reference, alpha=alphapar, beta=beta)    ###MAGIC   HESSII

elif spectype == 'ExponentialCutoffPowerLaw':
    neb_model = ExponentialCutoffPowerLaw(index=index, amplitude=amplitude, reference=reference, lambda_=lambda_) ## HESS

elif spectype == 'PowerLaw':
    neb_model = PowerLaw(index=index, amplitude=amplitude, reference=reference)    ### HEGRA

else:
    raise ValueError('Spectra Model must be either "LogParabola", "PowerLaw", or "ExponentialCutoffPowerLaw"')


energy_array = np.linspace(emin.value, emax.value, 300) * u.TeV
flare_plus_neb_model = np.zeros(len(energy_array)) * u.Unit('cm-2 s-1 TeV-1')
for i in range(len(energy_array)):
    flare_plus_neb_model.value[i] = (neb_model.evaluate(energy_array[i], amplitude=amplitude, reference=reference, alpha=alphapar, beta=beta).value + flare_model.evaluate(energy_array[i], index = flare_index, amplitude = flare_amplitude, reference = flare_reference,lambda_ = lambda_flare).value) * 1e11


#print flare_plus_neb_model
sum_model = TableModel(energy_array,flare_plus_neb_model,scale=1.0e-11)


# No EBL model needed
target = Target(name='Crab', model=sum_model)

#events = np.zeros(N_simul) * u.ct
cnts_specs = []

# Simulation
t_start = time.clock()

for i in range(N_simul):
    simu = CTAObservationSimulation.simulate_obs(perf=cta_perf,
                                                 target=target,
                                                 obs_param=obs_param)

    #print(simu)
    #flux_points = CTAObservationSimulation.plot_simu(simu, target, filename, livetime) * u.Unit('cm-2 s-1 TeV-1')
    #print(flux_points)
    #plt.show()
    #print(simu.on_vector.data.data.value)

    #fit_crab = SpectrumFit(obs_list = simu, model=models2fit[0],fit_range=(emin, emax), stat = "cash")
    #fit = fit_crab.fit()
    #fit_err = fit_crab.est_errors()
    #result = fit_crab.result
    #ax0, ax1 = results[0].plot(figsize=(8,8))
    #plt.show()
    
    #expc_cnts = result[0].expected_source_counts
    #cnts_spec_res = result[0].expected_source_counts.data.data.value
    
    cnts_spec_res = simu.on_vector.data.data.value-(simu.off_vector.data.data.value*alpha)
    cnts_specs.append(cnts_spec_res)

#print(cnts_specs)
t_end = time.clock()
print('\nsimulation done in {} s'.format(t_end-t_start))

#print(result[0].expected_source_counts.energy)

sum_cnts_specs = np.zeros(len(cnts_specs[0]))
for l in range(N_simul):
    sum_cnts_specs += np.asarray(cnts_specs[l])
mean_cnts_specs = sum_cnts_specs/N_simul * u.ct

print('Mean counts spectrum: ', mean_cnts_specs)

sd_cnts_specs = np.zeros(len(cnts_specs[0]))
for z in range(len(cnts_specs[0])):
    sd_arr = np.zeros(N_simul)
    for m in range(N_simul):
        sd_arr[m] = cnts_specs[m][z]
    sd_cnts_specs[z] = np.std(sd_arr)

print('Sd counts spectrum: ', sd_cnts_specs)

tot_events = np.zeros(N_simul)
for v in range(N_simul):
    tot_events[v] = cnts_specs[v].sum()


print('Number of simulations: ', N_simul,  ' Mean number of total events: ', np.mean(tot_events), ' Sd: ', np.std(tot_events))



