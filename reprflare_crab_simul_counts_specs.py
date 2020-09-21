import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
from scipy.stats import norm
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
from matplotlib import rc
from scipy.stats import chi2


lo_energ = np.array([1.25892544e-02,   1.99526232e-02,   3.16227786e-02,   5.01187257e-02,
                     7.94328228e-02,   1.25892550e-01,   1.99526235e-01,   3.16227764e-01,
                     5.01187205e-01,   7.94328213e-01,   1.25892544e+00,   1.99526227e+00,
                     3.16227746e+00,   5.01187229e+00,   7.94328213e+00,   1.25892534e+01,
                     1.99526215e+01,   3.16227741e+01,   5.01187172e+01,   7.94328156e+01,
                     1.25892525e+02]) * u.TeV


hi_energ = np.array([1.99526232e-02,   3.16227786e-02,   5.01187257e-02,   7.94328228e-02,
                     1.25892550e-01,   1.99526235e-01,   3.16227764e-01,   5.01187205e-01,
                     7.94328213e-01,   1.25892544e+00,   1.99526227e+00,   3.16227746e+00,
                     5.01187229e+00,   7.94328213e+00,  1.25892534e+01,  1.99526215e+01,
                     3.16227741e+01,   5.01187172e+01,   7.94328156e+01,   1.25892525e+02,
                     1.99526215e+02]) * u.TeV


###### 10 HOURS


#### MAGIC LogParabola


y_spec2 = np.array([8.00048280e+02, 6.10063590e+03, 1.09622779e+04, 7.96988340e+03,
                    4.92487646e+03, 4.22223664e+03, 4.72902524e+03, 3.43176270e+03,
                    2.70397268e+03, 1.97044036e+03, 1.36165088e+03, 8.74479840e+02,
                    5.02390680e+02, 2.51993300e+02, 1.04199420e+02, 3.99342800e+01,
                    1.48004600e+01, 5.74004000e+00, 1.99542000e+00, 5.25800000e-01,
                    1.27800000e-01]) * u.ct

y_spec2_err = np.array([108.05642217, 100.00467051, 150.73972798, 95.23024471, 72.68907685,
                        66.06913084, 69.4632497, 59.2546783, 51.45236474, 44.38569674,
                        37.20387188, 29.7749452, 22.78836741, 15.9679158, 10.24450075,
                        6.34337898, 3.83970569, 2.39233376, 1.42372435, 0.72328028,
                        0.35730542])


#### Tests
'''
y_specflare  = np.array([1.08315820e+02, 6.77341200e+02, 1.69961622e+03, 2.05603528e+03,
                         2.15679390e+03, 2.72031260e+03, 3.73855390e+03, 3.07761214e+03,
                         2.61400628e+03, 1.96375938e+03, 1.36357044e+03, 8.74258060e+02,
                         5.01831120e+02, 2.52078280e+02, 1.04165220e+02, 3.98621200e+01,
                         1.48627200e+01, 5.76646000e+00, 1.97396000e+00, 5.29600000e-01,
                         1.16700000e-01]) * u.ct


y_specflare_err = np.array([104.01908112,  67.27347672, 115.69675427,  57.78911042,
                            48.71828822, 53.95348079, 62.06656722, 55.21764762,
                            51.9332896, 44.21969301, 36.88986699, 29.52325038,
                            22.43794856, 15.84870317, 10.09753843, 6.24048052,
                            3.87037908, 2.38008552, 1.40314287, 0.71380939,
                            0.33894116])


'''
##### Neb + flare counts


'''
## Power-law model Index = 3.0
y_specflare  = np.array([1.33603020e+02, 1.25330480e+03, 3.00327574e+03, 3.17877368e+03,
                         2.89112860e+03, 3.27394766e+03, 4.22955908e+03, 3.33883404e+03,
                         2.75623504e+03, 2.03868144e+03, 1.40363112e+03, 8.96606940e+02,
                         5.13077260e+02, 2.56934860e+02, 1.05949100e+02, 4.04390600e+01,
                         1.50797800e+01, 5.84322000e+00, 2.01486000e+00, 5.41000000e-01,
                         1.25800000e-01]) * u.ct


y_specflare_err = np.array([104.72347333,  72.38809118, 121.83370661,  65.72962148,
                            55.40601554,  58.51101011,  65.82369268,  57.90593014,
                            53.11486003,  45.14789155,  37.50706813,  29.56015541,
                            22.72491256,  15.99484344,  10.16617987,   6.27515596,
                            3.8665984 ,   2.40693831,   1.43069185,   0.72973899,
                            0.35010621])

'''

'''
## Power-law model Index = 3.5
y_specflare  = np.array([7.71686000e+01, 7.36268660e+02, 1.91658012e+03, 2.25233620e+03,
                         2.28274992e+03, 2.80638614e+03, 3.80399596e+03, 3.10423206e+03,
                         2.62203706e+03, 1.96519716e+03, 1.36385396e+03, 8.74967420e+02,
                         5.02165860e+02, 2.52190420e+02, 1.04248740e+02, 3.97990200e+01,
                         1.48131600e+01, 5.74372000e+00, 1.96134000e+00, 5.14000000e-01,
                         1.17500000e-01]) * u.ct


y_specflare_err = np.array([103.75734572,  68.76335741, 117.01469746,  59.0680097 ,
                            49.79233439,  54.23182678,  62.55097377,  55.85086141,
                            51.81763058,  44.68965697,  37.02765356,  29.66106563,
                            22.2857897 ,  15.8458512 ,  10.11355113,   6.31659743,
                            3.79566105,   2.41077593,   1.41614314,   0.71708019,
                            0.34769779])
'''

'''
## Power-law model Index = 2.5
y_specflare  = np.array([5.25465780e+02, 4.73039882e+03, 1.08407014e+04, 1.07347869e+04,
                         8.98375198e+03, 9.40803656e+03, 1.15242645e+04, 8.73091214e+03,
                         6.99614302e+03, 5.11892464e+03, 3.56360788e+03, 2.35564076e+03,
                         1.43435892e+03, 7.87785120e+02, 3.69101080e+02, 1.66484340e+02,
                         7.64725200e+01, 3.83727000e+01, 1.77814600e+01, 6.70180000e+00,
                         2.37420000e+00]) * u.ct
    
    
y_specflare_err = np.array([105.80003394,  92.55555976, 149.38350516, 109.31489485,
                            96.19530863,  97.19077067, 107.58663407,  93.31464724,
                            84.05917919,  71.64653031,  59.98202666,  48.62138002,
                            37.64015442,  27.99155949,  19.26370408,  12.93931755,
                            8.68677183,   6.16659409,   4.24705313,   2.60228299,
                            1.52897821])
'''

y_specflare  = np.array([2.99932600e+02, 2.53032428e+03, 5.27453102e+03, 4.67022022e+03,
                         3.55264558e+03, 3.54854078e+03, 4.30624864e+03, 3.28131976e+03,
                         2.66428244e+03, 1.96651270e+03, 1.36219368e+03, 8.74376020e+02,
                         5.02100000e+02, 2.51836480e+02, 1.04033200e+02, 3.98640200e+01,
                         1.48030600e+01, 5.70250000e+00, 1.95666000e+00, 5.32800000e-01,
                         1.22300000e-01]) * u.ct


y_specflare_err = np.array([106.43652475, 79.29176504, 129.93017992, 76.44073937, 61.62438127,
                            61.25432113, 66.14193842, 57.65262028, 51.60801538, 45.11377136,
                            36.91767642, 29.65320308, 22.27693965, 15.90465583, 10.12384442,
                            6.35925542, 3.87401221, 2.41273657, 1.39049834, 0.7313851,
                            0.34977523])


#############
make_plot = 'False'

if make_plot == 'True':
    x = lo_energ.value
    width = (hi_energ.value - lo_energ.value)

    fig, axs = plt.subplots(nrows=1, ncols=2, sharex=True)
    fig.text(0.5, 0.01, 'Energy [TeV]', ha='center', fontsize=18)
    fig.text(0.025, 0.5, 'Mean number of events', va='center', rotation='vertical', fontsize=18)

    ax = axs[0]
    ax.bar(x, y_spec2.value, yerr=y_spec2_err, width = width, color =(0.255, 0.412, 0.882, 0.7), linewidth = 0, align='edge')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.tick_params(labelsize=16)
    #ax.set_ylabel('Mean number of events')
    ax.set_title('SPEC2')

    ax = axs[1]
    ax.bar(x, y_specflare.value, yerr=y_specflare_err, width = width, color = (0.18, 0.545, 0.341, 0.7),linewidth = 0,align='edge')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.tick_params(labelsize=16)
    ax.set_title('SPEC2 +  Flare')


    plt.show()


N_simul = 10000

# Simulation
t_start = time.clock()

spec_models = [y_spec2.value]
chisquares = []
#sd_chisquares = []

init_bin = 6 #1
finit_bin = 18 #5

#init_bin = 12
#finit_bin = 16


for i in range(N_simul):
    
    chisquare_i = np.zeros(len(spec_models))
    #sd_chisquare_i = np.zeros(len(spec_models))
    for j in range(len(spec_models)):
        chisquare_val = 0
        #sd_chisquares_val = 0
        for k in range(len(y_spec2[init_bin:finit_bin])):
            expc_i = np.random.normal(loc=y_spec2[k+init_bin].value, scale=y_spec2_err[k+init_bin])
            obs_i = np.random.normal(loc=y_specflare[k+init_bin].value, scale=y_specflare_err[k+init_bin])
            data_i = obs_i - expc_i
            chisquare_val =  chisquare_val + (data_i**2/expc_i)
            #sd_chisquares_val = sd_chisquares_val + np.sqrt((2*data_i*y_specflare_err[k+init_bin]/expc_i)**2 + ((2*data_i*y_spec2_err[k+init_bin]*expc_i-data_i**2*y_spec2_err[k+init_bin])/(expc_i**2))**2)
        chisquare_i[j] = chisquare_val
        #sd_chisquare_i[j] = sd_chisquares_val

    chisquares.append(chisquare_i)
    #sd_chisquares.append(sd_chisquare_i)

#print(chisquares)
#print(sd_chisquares)

t_end = time.clock()
print('\nsimulation done in {} s'.format(t_end-t_start))

dof = len(y_spec2.value[init_bin:finit_bin])-1
print('Degrees of freedom: ', dof)


mean_chisquare = np.zeros(N_simul)
for l in range(N_simul):
    mean_chisquare[l] = chisquares[l]


p_crit = np.asarray([chi2.isf(q=0.05, df=dof), chi2.isf(q=0.01, df=dof), chi2.isf(q=0.001, df=dof)])
print(p_crit)

plt.hist(mean_chisquare,bins=20,density=True, alpha = 0.75)
plt.xlabel(r'$ \chi^{2} $', size = 18)
plt.ylabel('Density', size = 18)
plt.xticks(size = 14)
plt.yticks(size = 14)
plt.axvline(x=p_crit[0],color = 'k', linestyle = '--', linewidth = 2.0, label= r'$ \alpha = 0.05 $')
plt.axvline(x=p_crit[1], color = 'k', linestyle = ':', linewidth =2.0, label= r'$ \alpha = 0.01 $')
plt.axvline(x=p_crit[2], color = 'k',linewidth=2.0, label = r'$ \alpha = 0.001 $')
plt.legend(loc='upper right',fontsize = 16)
#plt.xlim(0, np.max(mean_chisquare)*1.05)
plt.tight_layout()
plt.show()


pcrit_i = np.zeros(len(p_crit))
for i in range(len(p_crit)):
    pcrit_i[i] = len(mean_chisquare[np.where(mean_chisquare < p_crit[i])])/float(N_simul)

print('Probability[chi-squares < critical_value]: ', pcrit_i, ' for p_value: 0.05, 0.01, 0.001')

'''
sd_chisquare = np.zeros(len(spec_models))
for z in range(len(chisquares[0])):
    sd_arr = np.zeros(N_simul)
    for m in range(N_simul):
        sd_arr[m] = chisquares[m][z]
    sd_chisquare[z] = abs(np.sort(sd_arr)[int(0.05*N_simul)]-np.mean(sd_arr))

print('95% CL chi-squares: ', sd_chisquare)

mean_sd_chisquare = np.zeros(len(spec_models))
for l in range(N_simul):
    mean_sd_chisquare += sd_chisquares[l]
mean_sd_chisquare = mean_sd_chisquare/N_simul

print('sigma CL chi-squares: ', mean_sd_chisquare)
'''


#init_energ = 1.0
#finit_energ = 10.0

#init_bin = np.where(abs(lo_energ.value - np.repeat(init_energ, len(lo_energ.value))) == min(abs(lo_energ.value - np.repeat(init_energ, len(lo_energ.value)))))[0][0]
#finit_bin = np.where(abs(lo_energ.value - np.repeat(finit_energ, len(lo_energ.value))) == min(abs(lo_energ.value - np.repeat(finit_energ, len(lo_energ.value)))))[0][0]


#delt_AIC_mat = np.zeros((4,4))
#Lik_mat = np.zeros((4,4))
#specs = [y_spec1.value,y_spec2.value,y_spec3.value,y_spec4.value]
#specs_err = [y_spec1_err,y_spec2_err,y_spec3_err,y_spec4_err]

#dof_specs = np.asarray([2,3,3,3])

#for i in range(4):
#    Lik_mat[i][i] = 1.0
#    for v in range(finit_bin-init_bin):
        #Lik_mat[i][i] = Lik_mat[i][i]*abs(norm.cdf(specs[i][v+init_bin]-specs_err[i][v+init_bin], loc = specs[i][v+init_bin], scale = specs_err[i][v+init_bin])-norm.cdf(specs[i][v+init_bin]+specs_err[i][v+init_bin], loc = specs[i][v+init_bin], scale = specs_err[i][v+init_bin]))
        #        Lik_mat[i][i] = Lik_mat[i][i]*abs(1-norm.cdf(specs[i][v+init_bin], loc = specs[i][v+init_bin], scale = specs_err[i][v+init_bin]))
    

#    for j in range(4):
#        if j != i:
#            Lik_mat[i][j] = 1.0
#            for v in range(finit_bin-init_bin):
#                E_v = np.sqrt(lo_energ.value[v+init_bin]*hi_energ.value[v+init_bin])
#                #Lik_mat[i][j] = Lik_mat[i][j]*abs(norm.cdf(specs[i][v+init_bin]-specs_err[i][v+init_bin], loc = specs[j][v+init_bin], scale = specs_err[j][v+init_bin])-norm.cdf(specs[i][v+init_bin]+specs_err[i][v+init_bin], loc = specs[j][v+init_bin], scale = specs_err[j][v+init_bin]))
#                Lik_mat[i][j] = Lik_mat[i][j]*abs(1-norm.cdf(specs[i][v+init_bin], loc = specs[j][v+init_bin], scale = specs_err[j][v+init_bin]))
                

#        delt_AIC_mat[i][j] = 2*(dof_specs[i]-dof_specs[j]) - 2*np.log(Lik_mat[i][i]/Lik_mat[i][j])

#print(lo_energ[init_bin],lo_energ[finit_bin], finit_bin - init_bin)
#print(delt_AIC_mat)
#print(Lik_mat)












