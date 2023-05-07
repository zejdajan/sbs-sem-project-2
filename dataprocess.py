import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
import scipy.stats as sst
import statistics


#This python file is used for computations for the semesters work of SBS
#some parts of the code are used from Moodle inspiration file Jupyter Notebook:
#https://moodle.fel.cvut.cz/mod/resource/view.php?id=259425

# Show empirical CDF and compare with Rayleigh and Rice distributions
def rice_cdf( r, a, sigma ): # adjusting sst.rice implementation
    return sst.rice.cdf(r, a/sigma, scale=sigma)
def rice_median( a, sigma ):
    return sst.rice.median( a/sigma, scale=sigma)
def rayleigh_cdf(r, sigma): # the same as sst.rayleigh.cdf for sigma=1
    return 1 - np.exp(-r**2/(2*sigma**2))

def ecdf(data, normalize=-999.0):
    '''
    generates CDF from empirical data 
    data - empirical data as a numpy vector
    normalize - percentil (%) to set zero (no normalization if outside <0,100>)
    returns - (x, y) vectors in tuple    
    '''
    x = np.sort(data)
    if 0 <= normalize <= 100.0:
        x -= np.percentile(data, normalize)
    y = np.arange(1, len(x)+1)/float(len(x))
    return (x, y)

def small_scale_crowd(measuredRx_df, time_s):
    max_dB = list()
    for row in measuredRx_df.iterrows():
        max_dB.append(max(row[1]))
    t = np.linspace(0, time_s, len(max_dB))


    print("Max: ", max(max_dB))
    print("Min: ", min(max_dB))
    print("Median: ", statistics.median(max_dB))

    plt.plot(t, max_dB)
    plt.xlabel('time (s)')
    plt.ylabel('Measured RSS (dB)')
    plt.title('Crowd scenario')
    plt.grid()
    plt.savefig('figures/Crowd_scenario1_Measured')
    plt.show()
    max_dB = max_dB - np.median(max_dB) # normalize to median




    plt.plot(t, max_dB)
    plt.xlabel('time (s)')
    plt.ylabel('normalized RSS (dB)')
    plt.title('Crowded scenario')
    plt.grid()
    plt.savefig('figures/Crowd_scenario1')

    plt.show()

    xcdf, ycdf = ecdf(max_dB, 50)
    plt.semilogy(xcdf, 100*ycdf, label='Measured')
    xr = np.arange(np.power(10.0, -30/20), np.power(10.0, 10/20), 0.1)
    yr = sst.rayleigh.cdf(xr)
    xr = 20.0*np.log10(xr/sst.rayleigh.median())
    plt.semilogy(xr,100*yr, label='Rayleigh')
    plt.plot([0,0],[0.01,100], label='no fading')
    plt.plot([-30,10],[50,50], label='50%')
    plt.ylim(0.1,100)
    plt.xlim(-30,10)
    plt.xlabel('Relative loss (dB), normalized (0 dB at 50%)')
    plt.ylabel('Cumulative probability (%)')
    plt.title('CDF')
    plt.legend()
    plt.grid()
    plt.savefig('figures/Crowd_scenario2')

    plt.show()

    plt.figure(figsize=(9, 7))
    plt.ylim(0.1,100)
    plt.xlim(-30,10)
    plt.title('CDF')
    plt.xlabel('Relative loss (dB), normalized (0 dB at 50%)')
    plt.ylabel('Cumulative probability (%)')
    plt.plot([0,0],[0.01,100], '--', label='no fading')
    plt.plot([-30,10],[50,50], '--', label='50%')

    # Rayleigh distribution normalized to median
    x_lin = np.arange(np.power(10.0, -60/20), np.power(10.0, 10/20), 0.01)
    x_log_rayleigh_norm = 20.0*np.log10(x_lin/sst.rayleigh.median())
    y_rayleigh = sst.rayleigh.cdf(x_lin)
    plt.semilogy(x_log_rayleigh_norm, 100*y_rayleigh, label='Rayleigh (Rice k=0)')

    # Rice distribution normalized to median
    for k in [2,5,10,40]: # rice k-factor k=a**2/(2*sigma**2)
        a = 1
        sigma = np.sqrt(a**2/(2*k))
        y_rice =  rice_cdf( x_lin, a, sigma )
        x_rice_norm = 20.0*np.log10(x_lin/rice_median(a, sigma))
        plt.semilogy(x_rice_norm, 100*y_rice, ':', label='Rice k='+str(k))

    # empirical CDF for measured data normalized to median
    xcdf, ycdf = ecdf(max_dB, 50)
    plt.semilogy(xcdf, 100*ycdf, label='Measured', linewidth=3)

    plt.legend()
    plt.grid()
    plt.savefig('figures/Crowd_scenario3')

    plt.show()

def small_scale_dense(measuredRx_df, time_s):
    max_dB = list()
    for row in measuredRx_df.iterrows():
        max_dB.append(max(row[1]))
    t = np.linspace(0, time_s, len(max_dB))
    max_dB = max_dB[500:700]
    #1100-we want 3x bigger interval
    t = t[500:1100] - t[500]

    max_dB=max_dB+max_dB+max_dB


    print("Max: ", max(max_dB))
    print("Min: ", min(max_dB))
    print("Median: ", statistics.median(max_dB))

    plt.plot(t, max_dB)
    plt.xlabel('time (s)')
    plt.ylabel('Measured RSS (dB)')
    plt.title('Dense scenario')
    plt.grid()
    plt.savefig('figures/Dense_scenario1_Measured')
    plt.show()
    max_dB = max_dB - np.median(max_dB) # normalize to median

    plt.plot(t, max_dB)
    plt.xlabel('time (s)')
    plt.ylabel('normalized RSS (dB)')
    plt.title('Dense crowd scenario')
    plt.grid()
    plt.savefig('figures/Dense_scenario1')

    plt.show()

    xcdf, ycdf = ecdf(max_dB, 50)
    plt.semilogy(xcdf, 100*ycdf, label='Measured')
    xr = np.arange(np.power(10.0, -30/20), np.power(10.0, 10/20), 0.1)
    yr = sst.rayleigh.cdf(xr)
    xr = 20.0*np.log10(xr/sst.rayleigh.median())
    plt.semilogy(xr,100*yr, label='Rayleigh')
    plt.plot([0,0],[0.01,100], label='no fading')
    plt.plot([-30,10],[50,50], label='50%')
    plt.ylim(0.1,100)
    plt.xlim(-30,10)
    plt.xlabel('Relative loss (dB), normalized (0 dB at 50%)')
    plt.ylabel('Cumulative probability (%)')
    plt.title('CDF')
    plt.legend()
    plt.grid()
    plt.savefig('figures/Dense_scenario2')

    plt.show()

    plt.figure(figsize=(9, 7))
    plt.ylim(0.1,100)
    plt.xlim(-30,10)
    plt.title('CDF')
    plt.xlabel('Relative loss (dB), normalized (0 dB at 50%)')
    plt.ylabel('Cumulative probability (%)')
    plt.plot([0,0],[0.01,100], '--', label='no fading')
    plt.plot([-30,10],[50,50], '--', label='50%')

    # Rayleigh distribution normalized to median
    x_lin = np.arange(np.power(10.0, -60/20), np.power(10.0, 10/20), 0.01)
    x_log_rayleigh_norm = 20.0*np.log10(x_lin/sst.rayleigh.median())
    y_rayleigh = sst.rayleigh.cdf(x_lin)
    plt.semilogy(x_log_rayleigh_norm, 100*y_rayleigh, label='Rayleigh (Rice k=0)')

    # Rice distribution normalized to median
    for k in [2,5,10,40]: # rice k-factor k=a**2/(2*sigma**2)
        a = 1
        sigma = np.sqrt(a**2/(2*k))
        y_rice =  rice_cdf( x_lin, a, sigma )
        x_rice_norm = 20.0*np.log10(x_lin/rice_median(a, sigma))
        plt.semilogy(x_rice_norm, 100*y_rice, ':', label='Rice k='+str(k))

    # empirical CDF for measured data normalized to median
    xcdf, ycdf = ecdf(max_dB, 50)
    plt.semilogy(xcdf, 100*ycdf, label='Measured', linewidth=3)

    plt.legend()
    plt.grid()
    plt.savefig('figures/Dense_scenario3')

    plt.show()

def small_scale_static(data, time_s):
    max_dB = list()
    for d in data:
        max_dB.append(max(d))
    t = np.linspace(0, time_s, len(max_dB))


    print("Max: ", max(max_dB))
    print("Min: ", min(max_dB))
    print("Median: ", statistics.median(max_dB))

    plt.plot(t, max_dB)
    plt.xlabel('time (s)')
    plt.ylabel('Measured RSS (dB)')
    plt.title('Static scenario')
    plt.grid()
    plt.savefig('figures/Static_scenario1_Measured')
    plt.show()
    max_dB = max_dB - np.median(max_dB) # normalize to median

    plt.plot(t, max_dB)
    plt.xlabel('time (s)')
    plt.ylabel('normalized RSS (dB)')
    plt.title('Static scenario')
    plt.grid()
    plt.savefig('figures/Static_scenario1')

    plt.show()

    xcdf, ycdf = ecdf(max_dB, 50)
    plt.semilogy(xcdf, 100*ycdf, label='Measured')
    xr = np.arange(np.power(10.0, -30/20), np.power(10.0, 10/20), 0.1)
    yr = sst.rayleigh.cdf(xr)
    xr = 20.0*np.log10(xr/sst.rayleigh.median())
    plt.semilogy(xr,100*yr, label='Rayleigh')
    plt.plot([0,0],[0.01,100], label='no fading')
    plt.plot([-30,10],[50,50], label='50%')
    plt.ylim(0.1,100)
    plt.xlim(-30,10)
    plt.xlabel('Relative loss (dB), normalized (0 dB at 50%)')
    plt.ylabel('Cumulative probability (%)')
    plt.title('CDF')
    plt.legend()
    plt.grid()
    plt.savefig('figures/Static_scenario2')

    plt.show()

    plt.figure(figsize=(9, 7))
    plt.ylim(0.1,100)
    plt.xlim(-30,10)
    plt.title('CDF')
    plt.xlabel('Relative loss (dB), normalized (0 dB at 50%)')
    plt.ylabel('Cumulative probability (%)')
    plt.plot([0,0],[0.01,100], '--', label='no fading')
    plt.plot([-30,10],[50,50], '--', label='50%')

    # Rayleigh distribution normalized to median
    x_lin = np.arange(np.power(10.0, -60/20), np.power(10.0, 10/20), 0.01)
    x_log_rayleigh_norm = 20.0*np.log10(x_lin/sst.rayleigh.median())
    y_rayleigh = sst.rayleigh.cdf(x_lin)
    plt.semilogy(x_log_rayleigh_norm, 100*y_rayleigh, label='Rayleigh (Rice k=0)')

    # Rice distribution normalized to median
    for k in [2,5,10,40]: # rice k-factor k=a**2/(2*sigma**2)
        a = 1
        sigma = np.sqrt(a**2/(2*k))
        y_rice =  rice_cdf( x_lin, a, sigma )
        x_rice_norm = 20.0*np.log10(x_lin/rice_median(a, sigma))
        plt.semilogy(x_rice_norm, 100*y_rice, ':', label='Rice k='+str(k))

    # empirical CDF for measured data normalized to median
    xcdf, ycdf = ecdf(max_dB, 50)
    plt.semilogy(xcdf, 100*ycdf, label='Measured', linewidth=3)

    plt.legend()
    plt.grid()
    plt.savefig('figures/Static_scenario3')

    plt.show()

def large_scale(name,measuredRx_df, time_df, distance, d_start=20):
    max_dB = list()
    for row in measuredRx_df.iterrows():
        max_dB.append(max(row[1]))
    distances = np.linspace(1, distance, time_df.shape[1])[:len(max_dB)]
    start = (d_start * len(distances) // distance)
    distances = distances[start:]

    # measured PL
    measuredPL = (max_dB[start] - np.array(max_dB[start:])) + (max_dB[start] - max_dB[len(max_dB) - 1])
    win=50 # slide window should consider the wavelength and environment
    measuedPLf=[]
    for i in range(len(measuredPL)):
        imin = max(0, i-win)
        imax = min(len(measuredPL),i+win)
        measuedPLf.append(sum(measuredPL[j] for j in range(imin, imax))/(imax-imin))

    logd = np.log10(distances)
    n, L1, _, _, _ = sst.linregress(logd, measuredPL) # fit the model

    PLpredict = L1 + n * logd # prediction by the derived model
    err = PLpredict - measuredPL
    sigma = np.std(err)

    logging.info(f"sigma = {sigma} dB, L1 = {L1} dB")

    plt.plot(distances, max_dB[start:])
    plt.title('Signal Strength vs. Distance')
    plt.xlabel('distance [m]')
    plt.ylabel('signal [dB]')
    plt.grid()
    plt.savefig('figures/Signal_StrengthVDistance_'+name)
    plt.show()

    plt.semilogx(distances, PLpredict, label=f"n={0.1 * n}")
    plt.semilogx(distances, measuredPL)
    plt.title('Mean Path Loss Model vs. Measured Path Loss')
    plt.xlabel('distance')
    plt.ylabel('PL [dB]')
    plt.legend()
    plt.grid()
    plt.savefig('figures/Mean_Path_LossVMeasured_'+name)

    plt.show()

    # PDF
    plt.figure()
    plt.hist(err, bins=40, density=True, label='Empirical PDF')
    # comparison with Log-Normal distribution
    pdfx = np.linspace(-30,30)
    ppdf = sst.norm.pdf(pdfx, scale=sigma)
    plt.plot(pdfx, ppdf, label='Normal distribution fit')
    plt.legend()
    plt.grid()
    plt.savefig('figures/Statistics_'+name)

    plt.show()

    plt.figure(figsize=(9, 7))
    xcdf, ycdf = ecdf(err, 50)
    plt.semilogy(xcdf, 100*ycdf, label='Empirical CDF')
    plt.semilogy(xcdf,100*sst.norm.cdf(xcdf,scale=sigma), label='Normal CDF fit')
    plt.ylim(0.1,100)
    plt.xlabel('Normalized predicted RSS (dB)')
    plt.ylabel('Percentage of locations RSS < predicted RSS (%)')
    plt.legend()
    plt.grid()
    plt.savefig('figures/Prediction1_'+name)

    plt.show()

    Pn = 20
    plt.figure(figsize=(9, 9))
    plt.subplot(211)
    plt.semilogx(distances, Pn-measuredPL, '+')
    plt.plot(distances, Pn-PLpredict, lw =4, label='50 % location exceeding this' )
    plt.plot(distances, Pn-PLpredict-9.5, lw =4, label='90 % location exceeding this' )
    plt.plot(distances, Pn-PLpredict+9.5, lw =4, label='10 % location exceeding this' )
    plt.xlabel('Distance (m)')
    plt.ylabel('RSS (dBm)')
    plt.title('Predicted RSS (Distance in log and linear scale)')
    plt.legend()
    plt.grid()
    plt.subplot(212)
    plt.plot(distances, Pn-measuredPL, '+')
    plt.plot(distances, Pn-PLpredict, lw =4, label='50 % location exceeding this' )
    plt.plot(distances, Pn-PLpredict-9.5, lw =4, label='90 % location exceeding this' )
    plt.plot(distances, Pn-PLpredict+9.5, lw =4, label='10 % location exceeding this' )
    plt.xlabel('Distance (m)')
    plt.ylabel('RSS (dBm)')
    plt.grid()
    plt.savefig('figures/Prediction2'+name)

    plt.show()


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
    partyzanu_down_time = pd.read_csv('./data/partyzanu_dolu_casy.csv', delimiter=';')
    partyzanu_down_data = pd.read_csv('./data/partyzanu_dolu_data.csv', delimiter=';')
    partyzanu_down_dist = 350  # meters

    terronska_down_time = pd.read_csv('./data/terronska_dolu_casy.csv', delimiter=';')      #pd.read_csv('./data/partyzanu_dolu_casy.csv', delimiter=';')
    terronska_down_data = pd.read_csv('./data/terronska_dolu_data.csv', delimiter=';')  # pd.read_csv('./data/partyzanu_dolu_data.csv', delimiter=';')
    terronska_down_dist = 350  # meters

    large_scale("partyzanu",partyzanu_down_data, partyzanu_down_time, partyzanu_down_dist, d_start=10)
    large_scale("terronska",terronska_down_data, terronska_down_time, terronska_down_dist, d_start=10)


    crowd1_data = pd.read_csv('./data/crowd_1_data.csv', delimiter=';')
    crowd1_time_s = 104

    crowd2_data = pd.read_csv('./data/crowd_2_data.csv', delimiter=';')
    crowd2_time_s = 117

    static1_data = pd.read_csv('./data/static_1_data.csv', delimiter=';').values.tolist()
    static2_data = pd.read_csv('./data/static_2_data.csv', delimiter=';').values.tolist()
    static3_data = pd.read_csv('./data/static_3_data.csv', delimiter=';').values.tolist()
    static_data = static1_data + static2_data + static3_data
    static_time_s = 58
    print("Static")
    small_scale_static(static_data, static_time_s)

    print("Crowd")
    small_scale_crowd(crowd2_data, crowd2_time_s)

    print("Dense")
    small_scale_dense(crowd1_data, crowd1_time_s)