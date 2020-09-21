import matplotlib.pyplot as plt
import astropy.units as u
energy = 1 * u.TeV
import numpy as np
import time
from gammapy.scripts.cta_utils import CTAObservationSimulation
from gammapy.scripts import CTAPerf
from gammapy.spectrum.models import TableModel
from gammapy.spectrum.models import LogParabola
from gammapy.spectrum.models import PowerLaw
from gammapy.spectrum.models import ExponentialCutoffPowerLaw
from gammapy.scripts.cta_utils import Target
from gammapy.scripts.cta_utils import ObservationParameters
from gammapy.spectrum import SpectrumFit

# Observation parameters
alpha = 0.2 * u.Unit('')
livetime = 10. * u.h
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
lambda_= 0.07 * u.Unit('TeV-1')            ### HESS: 0.07  (1/Ecutoff)      ### Needed with: 'ExponentialCutoffPowerLaw'
amplitude= 3.23e-11 * u.Unit('cm-2 s-1 TeV-1') ### HEGRA: 2.83e-11 HESS: 3.76e-11 MAGIC: 3.23e-11  HESSII: 1.79e-10 ### Needed for: All models
reference= 1.0 * u.TeV     ### HEGRA, HESS, MAGIC: 1        HESSII: 0.521    ### Needed for: All models
alphapar=2.47 * u.Unit('')  ### MAGIC: 2.47 HESSII: 2.1      ### Needed for: 'LogParabola'
beta=0.104 * u.Unit('')   ### MAGIC: 0.24/np.log(10) HESSII: 0.24          ### Needed for: 'LogParabola'

######## Define Flare model

flare_index = 3.669 * u.Unit('')
flare_amplitude = 1.597e-12 * u.Unit('cm-2 s-1 TeV-1')
flare_reference = 1.25 * u.TeV
lambda_flare = 0.0 * u.Unit('TeV-1')

flare_model = ExponentialCutoffPowerLaw(index=flare_index, amplitude=flare_amplitude, reference=flare_reference,lambda_ = lambda_flare)
#flare_model = PowerLaw(index=flare_index, amplitude=flare_amplitude, reference=flare_reference)




#######################################################################

# Model

if spectype == 'PowerLaw':
    neb_model = PowerLaw(index=index, amplitude=amplitude, reference=reference)    ### HEGRA

elif spectype == 'LogParabola':
    neb_model = LogParabola(amplitude=amplitude, reference=reference, alpha=alphapar, beta=beta)    ###MAGIC   HESSII

elif spectype == 'ExponentialCutoffPowerLaw':
    neb_model = ExponentialCutoffPowerLaw(index=index, amplitude=amplitude, reference=reference, lambda_=lambda_)   ####HESS

else:
    raise ValueError('Spectra Model must be either "LogParabola", "PowerLaw", or "ExponentialCutoffPowerLaw"')


energy_array = np.linspace(emin.value, emax.value, 300) * u.TeV
flare_plus_neb_model = np.zeros(len(energy_array)) * u.Unit('cm-2 s-1 TeV-1')
for i in range(len(energy_array)):
    flare_plus_neb_model.value[i] = (neb_model.evaluate(energy_array[i], amplitude=amplitude, reference=reference, alpha=alphapar, beta=beta).value + flare_model.evaluate(energy_array[i], index = flare_index, amplitude = flare_amplitude, reference = flare_reference,lambda_ = lambda_flare).value) * 1e11


print(flare_plus_neb_model)
sum_model = TableModel(energy_array,flare_plus_neb_model,scale=1.0e-11)
#sum_model.parameters['amplitude'].value = 1.0e-11
#sum_model.parameters['amplitude'].unit = u.Unit('cm-2 s-1 TeV-1')
#print sum_model.parameters


# No EBL model needed
target = Target(name='Crab', model=sum_model)

# Simulation
t_start = time.clock()
simu = CTAObservationSimulation.simulate_obs(perf=cta_perf,
                                             target=target,
                                             obs_param=obs_param)
t_end = time.clock()
print(simu)
print('\nsimulation done in {} s'.format(t_end-t_start))

#flux_points = CTAObservationSimulation.plot_simu(simu, target) #* u.Unit('cm-2 s-1 TeV-1')
#flux_points = CTAObservationSimulation.plot_simu(simu, target, filename, livetime) * u.Unit('cm-2 s-1 TeV-1')
#print(flux_points)

#plt.show()

#print(simu.on_vector.data.data.value)

model2fit1 = LogParabola(amplitude=1.00e-11 * u.Unit('cm-2 s-1 TeV-1') , reference=1.25 * u.TeV, alpha=2.5 * u.Unit(''),beta=0.1 * u.Unit('')  )
model2fit2 = ExponentialCutoffPowerLaw(index = 2.5 * u.Unit(''), amplitude = 5.0e-11 * u.Unit('cm-2 s-1 TeV-1'), reference= 1.0 * u.TeV ,  lambda_= 0.1 * u.Unit('TeV-1') )
model2fit3 = PowerLaw(index=2.5 * u.Unit(''), amplitude= 1.0e-11 * u.Unit('cm-2 s-1 TeV-1') , reference= 1.25 * u.TeV)


model2fit1.parameters['alpha'].parmin = 0.1
model2fit1.parameters['alpha'].parmax = 5.0
model2fit1.parameters['beta'].parmin = 0.0
model2fit1.parameters['beta'].parmax = 10.0
model2fit1.parameters['amplitude'].parmin = 1e-14
model2fit1.parameters['amplitude'].parmax = 1e-5


model2fit2.parameters['index'].parmin = 0.1
model2fit2.parameters['index'].parmax = 5.0
model2fit2.parameters['lambda_'].parmin = 0.001
model2fit2.parameters['lambda_'].parmax = 100
model2fit2.parameters['amplitude'].parmin = 1.0e-14
model2fit2.parameters['amplitude'].parmax = 1.0e-3


model2fit3.parameters['index'].parmin = 1.0
model2fit3.parameters['index'].parmax = 7.0
model2fit3.parameters['amplitude'].parmin = 1.0e-14
model2fit3.parameters['amplitude'].parmax = 1.0e-4

models2fit = [model2fit3]#,model2fit2,model2fit3]

for k in range(len(models2fit)):
    fit_crab = SpectrumFit(obs_list = simu, model=models2fit[k], fit_range=(1.0 * u.TeV,55 * u.TeV), stat = "cash")
    fit_crab.fit()
    fit_crab.est_errors()
    results = fit_crab.result
    #ax0, ax1 = results[0].plot(figsize=(8,8))
    #plt.show()
    print(results[0])


