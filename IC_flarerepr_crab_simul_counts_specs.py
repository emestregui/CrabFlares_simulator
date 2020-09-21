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


y_spec2 = np.array([1.08315820e+02, 6.77341200e+02, 1.69961622e+03, 2.05603528e+03,
                    2.15679390e+03, 2.72031260e+03, 3.73855390e+03, 3.07761214e+03,
                    2.61400628e+03, 1.96375938e+03, 1.36357044e+03, 8.74258060e+02,
                    5.01831120e+02, 2.52078280e+02, 1.04165220e+02, 3.98621200e+01,
                    1.48627200e+01, 5.76646000e+00, 1.97396000e+00, 5.29600000e-01,
                    1.16700000e-01]) * u.ct

y_spec2_err = np.array([104.01908112,  67.27347672, 115.69675427,  57.78911042,
                        48.71828822, 53.95348079, 62.06656722, 55.21764762,
                        51.9332896, 44.21969301, 36.88986699, 29.52325038,
                        22.43794856, 15.84870317, 10.09753843, 6.24048052,
                        3.87037908, 2.38008552, 1.40314287, 0.71380939,
                        0.33894116])


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
## B = 100 Index = 2.5
y_specflare  = np.array([2.51894623e+04, 2.02133506e+05, 3.83009827e+05, 2.84952345e+05,
                         1.64344979e+05, 1.18051987e+05, 1.08110883e+05, 6.23940805e+04,
                         3.78343785e+04, 2.13204339e+04, 1.15878902e+04, 6.08546736e+03,
                         2.98418314e+03, 1.33328144e+03, 5.08429880e+02, 1.85910460e+02,
                         6.87879200e+01, 2.72035600e+01, 9.88316000e+00, 2.91480000e+00,
                         7.87800000e-01]) * u.ct


y_specflare_err = np.array([192.12335513, 453.34801541, 635.46331014, 533.5191457 ,
                            405.71891071, 344.77717911, 327.82302502, 248.85504675,
                            193.449173  , 146.92561862, 108.00490137,  77.82861828,
                            54.69544383,  36.53291173,  22.68095631,  13.72280447,
                            8.2004828 ,   5.1805223 ,   3.16337485,   1.73122528,
                            0.88043805])
'''
'''
## B = 500 Index = 2.5
y_specflare  = np.array([5.25562380e+02, 4.33789474e+03, 8.77886782e+03, 7.42216536e+03,
                         5.29378862e+03, 4.94537004e+03, 5.69188726e+03, 4.13155274e+03,
                         3.19841854e+03, 2.26662190e+03, 1.51980510e+03, 9.52319600e+02,
                         5.38654320e+02, 2.67624040e+02, 1.10032760e+02, 4.18048600e+01,
                         1.56085600e+01, 6.05170000e+00, 2.09260000e+00, 5.46800000e-01,
                         1.34400000e-01]) * u.ct


y_specflare_err = np.array([106.09029213,  90.02223657, 142.81100262,  92.4652313 ,
                            74.33959743,  70.85373028,  76.01970148,  64.92671052,
                            56.53921567,  47.1712245 ,  38.57162836,  30.94451072,
                            23.29085076,  16.28619434,  10.62062554,   6.5027739 ,
                            3.95232472,   2.46005917,   1.46016754,   0.73634894,
                            0.36950865])


'''
'''
## B = 100 Index = 2.0
y_specflare  = np.array([6.60617200e+01, 6.48391660e+02, 1.72834840e+03, 2.09656892e+03,
                         2.19016062e+03, 2.75029124e+03, 3.77264934e+03, 3.10279016e+03,
                         2.63350356e+03, 1.97714614e+03, 1.37253746e+03, 8.80895260e+02,
                         5.06472320e+02, 2.54258060e+02, 1.05279760e+02, 4.05388600e+01,
                         1.51240400e+01, 5.81622000e+00, 2.03978000e+00, 5.42200000e-01,
                         1.34500000e-01]) * u.ct


y_specflare_err = np.array([103.70977701,  66.65807743, 113.64762738,  57.7515156 ,
                            48.89124328,  53.36006427,  62.41426095,  56.21575287,
                            51.49899435,  44.10363086,  36.54700213,  29.75892487,
                            22.43994443,  16.09027859,  10.17873854,   6.35960517,
                            3.91091576,   2.39851223,   1.41547644,   0.74378704,
                            0.36906605])
'''

'''
## B = 50 Index = 2.0
y_specflare  = np.array([7.04969200e+01, 7.08743260e+02, 1.86674622e+03, 2.23037068e+03,
                         2.29488484e+03, 2.85177130e+03, 3.88385714e+03, 3.17807562e+03,
                         2.68912496e+03, 2.01618306e+03, 1.39948014e+03, 8.99230020e+02,
                         5.17162040e+02, 2.61018680e+02, 1.08372880e+02, 4.20303200e+01,
                         1.58227600e+01, 6.28880000e+00, 2.25908000e+00, 6.22580000e-01,
                         1.60400000e-01]) * u.ct


y_specflare_err = np.array([104.77263363,  68.52090584, 115.90832377,  58.20944532,
                            50.23261954,  55.07315749,  65.03014553,  56.46578643,
                            52.00341074,  44.71568759,  37.41081209,  30.20218679,
                            22.5953811 ,  16.41399997,  10.35941198,   6.50720898,
                            3.98186112,   2.53852606,   1.50159301,   0.79195842,
                            0.40530463])

'''
'''
## B = 10 Index = 2.0

y_specflare  = np.array([2.08699680e+02, 1.92227966e+03, 4.60445626e+03, 4.85089742e+03,
                         4.35963652e+03, 4.85807100e+03, 6.19259404e+03, 4.83434372e+03,
                         3.96056690e+03, 2.92582340e+03, 2.03342398e+03, 1.32571506e+03,
                         7.86006720e+02, 4.15126400e+02, 1.84596540e+02, 7.83015400e+01,
                         3.36662000e+01, 1.56739000e+01, 6.79332000e+00, 2.36790000e+00,
                         8.06200000e-01]) * u.ct


y_specflare_err = np.array([104.34034391,  75.51301309, 126.78084057,  77.2089628 ,
                            67.75645217,  70.61162731,  80.06553858,  69.5417439 ,
                            62.8644707 ,  54.24616894,  44.9940618 ,  36.74240054,
                            27.97236398,  20.14812465,  13.60601514,   8.83070357,
                            5.85446168,   3.97328161,   2.61999454,   1.53315022,
                            0.90113349])
'''
'''
## B = 10 Index = 1.5
y_specflare  = np.array([6.52235000e+01, 6.32877720e+02, 1.69207500e+03, 2.06104460e+03,
                         2.16193610e+03, 2.72340110e+03, 3.74199228e+03, 3.08040624e+03,
                         2.61571854e+03, 1.96521688e+03, 1.36557184e+03, 8.77189580e+02,
                         5.04624700e+02, 2.54397440e+02, 1.06055560e+02, 4.13926000e+01,
                         1.60323600e+01, 6.74730000e+00, 2.80184000e+00, 1.04100000e+00,
                         4.35200000e-01]) * u.ct


y_specflare_err = np.array([104.42075497,  67.02532643, 115.29369585,  57.11720108,
                            48.20865591,  53.63532197,  62.05029564,  55.72475379,
                            52.05832214,  44.24916798,  36.57244459,  29.67303448,
                            22.33445611,  15.94920793,  10.3050035 ,   6.44632432,
                            4.01599214,   2.61434862,   1.66362875,   1.01267912,
                            0.65802808])

'''
'''
## B = 10 Index = 1.8

y_specflare  = np.array([6.65589600e+01, 6.46755100e+02, 1.73202258e+03, 2.10574836e+03,
                         2.20524096e+03, 2.77602296e+03, 3.81112080e+03, 3.13661788e+03,
                         2.66733708e+03, 2.00891594e+03, 1.40159336e+03, 9.06309260e+02,
                         5.26479940e+02, 2.69546100e+02, 1.14632780e+02, 4.58431000e+01,
                         1.84466400e+01, 8.04112000e+00, 3.36686000e+00, 1.18038000e+00,
                         4.32000000e-01]) * u.ct


y_specflare_err = np.array([103.06366244,  67.51887443, 114.9651226 ,  57.67426141,
                            49.48562404,  53.95388009,  62.88349093,  56.45978215,
                            52.15423903,  44.6493864 ,  37.63865508,  30.04751228,
                            22.86075847,  16.28542351,  10.77731476,   6.7461831 ,
                            4.2746837 ,   2.82924392,   1.84817362,   1.08794626,
                            0.6561829])
                            
'''

'''
## B = 10 Index = 1.9

y_specflare  = np.array([8.28931200e+01, 8.06513680e+02, 2.10261638e+03, 2.48230184e+03,
                         2.51626902e+03, 3.09117418e+03, 4.18206642e+03, 3.40999854e+03,
                         2.88088010e+03, 2.16595612e+03, 1.51327886e+03, 9.81983360e+02,
                         5.73993940e+02, 2.96322960e+02, 1.27562420e+02, 5.17238400e+01,
                         2.09374400e+01, 9.23842000e+00, 3.76940000e+00, 1.27700000e+00,
                         4.19100000e-01]) * u.ct


y_specflare_err = np.array([103.94577922,  68.08731562, 116.7664237 ,  60.42835643,
                            51.9360828 ,  56.87134014,  66.15026355,  58.08566489,
                            53.63976387,  46.58919817,  39.03080826,  31.18505275,
                            24.0608796 ,  17.16477162,  11.35665055,   7.19317841,
                            4.59578571,   3.03793086,   1.92684811,   1.13581292,
                            0.64222674])
'''
'''
## B = 50 Index = 2.1

y_specflare  = np.array([1.17866360e+02, 1.10485384e+03, 2.74598858e+03, 3.04944782e+03,
                         2.90255418e+03, 3.39478584e+03, 4.46104570e+03, 3.55806516e+03,
                         2.95451964e+03, 2.19257502e+03, 1.51440772e+03, 9.71296020e+02,
                         5.59340520e+02, 2.83663380e+02, 1.18896020e+02, 4.65598400e+01,
                         1.79624800e+01, 7.27378000e+00, 2.67616000e+00, 8.03700000e-01,
                         2.16000000e-01]) * u.ct


y_specflare_err = np.array([103.7339166 ,  71.16737532, 120.34153292,  65.12224853,
                            55.6598552 ,  60.12486833,  68.16413576,  59.27706587,
                            54.12988282,  47.25223851,  38.67295309,  31.05740105,
                            23.81454207,  16.87053298,  10.98207777,   6.92838244,
                            4.23454274,   2.67828611,   1.62292318,   0.89608387,
                            0.46940814])
'''
'''

## B = 1000 Index = 2.5

y_specflare  = np.array([1.42768340e+02, 1.31653522e+03, 3.11561374e+03, 3.24607344e+03,
                         2.90620062e+03, 3.25520968e+03, 4.18456040e+03, 3.29551810e+03,
                         2.71899134e+03, 2.01093170e+03, 1.38683650e+03, 8.86910300e+02,
                         5.07908400e+02, 2.54471420e+02, 1.04904140e+02, 4.01845000e+01,
                         1.49202000e+01, 5.79602000e+00, 1.96914000e+00, 5.35400000e-01,
                         1.30900000e-01]) * u.ct


y_specflare_err = np.array([104.26998165,  71.73020825, 120.6775947 ,  66.87974778,
                            55.97154159,  58.16366325,  65.65283384,  57.13898872,
                            52.14497208,  44.67974175,  37.32709   ,  29.54627743,
                            22.85903028,  16.02865207,  10.30977705,   6.32356385,
                            3.8398927 ,   2.4034617 ,   1.41229022,   0.71620307,
                            0.36022936])
'''
'''
## B = 1000 Index = 2.6

y_specflare  = np.array([3.10063780e+02, 2.66769150e+03, 5.70980044e+03, 5.21172356e+03,
                         4.05447166e+03, 4.06891144e+03, 4.90154856e+03, 3.68563142e+03,
                         2.93710614e+03, 2.12597132e+03, 1.44664796e+03, 9.16808240e+02,
                         5.21731320e+02, 2.60488020e+02, 1.07253800e+02, 4.10447600e+01,
                         1.51493400e+01, 5.83994000e+00, 2.01680000e+00, 5.62600000e-01,
                         1.25700000e-01]) * u.ct


y_specflare_err = np.array([104.63595661,  80.53843639, 131.87979715,  80.62673475,
                            65.86331899,  65.00797352,  70.69989919,  60.83661411,
                            54.41511277,  46.1060214 ,  38.00806951,  30.91420806,
                            22.89500267,  16.21017879,  10.36883762,   6.44144243,
                            3.88096864,   2.45431066,   1.44540297,   0.75065387,
                            0.3491411])
'''


y_specflare  = np.array([8.59615182e+03, 6.15635529e+04, 9.87426656e+04, 5.91021549e+04,
	2.73197428e+04, 1.70052105e+04, 1.48307042e+04, 8.54203852e+03,
	5.36093636e+03, 3.21493280e+03, 1.89840494e+03, 1.09263556e+03,
	5.85932800e+02, 2.80997440e+02, 1.12861300e+02, 4.24069400e+01,
	1.56099200e+01, 5.95122000e+00, 2.04030000e+00, 5.43800000e-01,
	1.22900000e-01]) * u.ct


y_specflare_err = np.array([137.49723259 ,256.12126641 ,333.31342065 ,244.51317234 ,164.21222218,
	131.98705396, 121.57745185,  93.57509214 , 73.13635352 , 56.77727683,
	43.57218447,  32.63871136  ,24.28356539 , 16.76577375 , 10.71500435,
  	6.46635321 ,  3.96741498  , 2.46976608 ,  1.42690291   ,0.73285849,
  	0.35213008])

#############
make_plot = 'True'

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
    plt.xlim(left=1.25)

    ax = axs[1]
    ax.bar(x, y_specflare.value, yerr=y_specflare_err, width = width, color = (0.18, 0.545, 0.341, 0.7),linewidth = 0,align='edge')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.tick_params(labelsize=16)
    ax.set_title('SPEC2 +  Flare')
    plt.xlim(left=1.25)


    plt.show()


N_simul = 10000

# Simulation
t_start = time.clock()

spec_models = [y_spec2.value]
chisquares = []
#sd_chisquares = []

#init_bin = 1
#finit_bin = 5

init_bin = 10
finit_bin = 17
print('Energy nodes: ', lo_energ[init_bin:finit_bin], hi_energ[finit_bin-1])

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
#plt.xlim(left=0.0)
#plt.xlim(right=np.sort(mean_chisquare)[int(0.99*N_simul)])
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












