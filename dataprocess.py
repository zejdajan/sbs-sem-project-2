import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
import scipy.stats as sst

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

    start = (70 * len(t) // time_s)
    max_dB = max_dB[start:]
    t = t[start:]

    logging.info(f"Max: {np.max(max_dB)} dB")
    logging.info(f"Min: {np.min(max_dB)} dB")
    logging.info(f"Median: {np.median(max_dB)} dB")

    plt.plot(t, max_dB)
    plt.xlabel('Time (s)')
    plt.ylabel('Measured RSS (dB)')
    plt.title('Crowd scenario')
    plt.grid()
    plt.savefig('figures/Crowd_scenario1_Measured')
    plt.show()
    max_dB = max_dB - np.median(max_dB) # normalize to median

    plt.plot(t, max_dB)
    plt.xlabel('Time (s)')
    plt.ylabel('Normalized RSS (dB)')
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
    percentile = [xcdf[np.floor(ycdf*100) == 1][0], xcdf[np.floor(ycdf*100) == 10][0], xcdf[np.floor(ycdf*100) == 50][0], 
                  xcdf[np.floor(ycdf*100) == 90][0]]
    logging.info(f"Percentile:")
    for i, p in enumerate([1, 10, 50, 90]):
        logging.info(f"{p} %: {percentile[i]} dB")
    plt.semilogy(xcdf, 100*ycdf, label='Measured', linewidth=3)
    plt.legend()
    plt.grid()
    plt.savefig('figures/Crowd_scenario3')
    plt.show()

    return xcdf, ycdf

def small_scale_dense(measuredRx_df, time_s):
    max_dB = list()
    for row in measuredRx_df.iterrows():
        max_dB.append(max(row[1]))
    max_dB = max_dB[0:1200]
    t = np.linspace(0, time_s, len(max_dB))

    start = (20 * len(t) // time_s)
    stop = (40 * len(t) // time_s)
    max_dB1 = max_dB[:start]
    max_dB2 = max_dB[start:stop]
    max_dB3 = max_dB[stop:]

    mean1 = np.mean(max_dB1)
    mean2 = np.mean(max_dB2)
    mean3 = np.mean(max_dB3)
    norm1 = mean1 - mean2
    norm2 = mean3 - mean2
    max_dB_tmp = [max_dB1 - norm1, max_dB2, max_dB3 - norm2]

    max_dB = list()
    for m in max_dB_tmp:
        for val in m:
            max_dB.append(val)

    logging.info(f"Max: {np.max(max_dB)} dB")
    logging.info(f"Min: {np.min(max_dB)} dB")
    logging.info(f"Median: {np.median(max_dB)} dB")

    plt.plot(t, max_dB)
    plt.xlabel('Time (s)')
    plt.ylabel('Measured RSS (dB)')
    plt.title('Dense scenario')
    plt.grid()
    plt.savefig('figures/Dense_scenario1_Measured')
    plt.show()
    max_dB = max_dB - np.median(max_dB) # normalize to median

    plt.plot(t, max_dB)
    plt.xlabel('Time (s)')
    plt.ylabel('Normalized RSS (dB)')
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
    percentile = [xcdf[np.floor(ycdf*100) == 1][0], xcdf[np.floor(ycdf*100) == 10][0], xcdf[np.floor(ycdf*100) == 50][0], 
                  xcdf[np.floor(ycdf*100) == 90][0]]
    logging.info(f"Percentile:")
    for i, p in enumerate([1, 10, 50, 90]):
        logging.info(f"{p} %: {percentile[i]} dB")
    plt.semilogy(xcdf, 100*ycdf, label='Measured', linewidth=3)

    plt.legend()
    plt.grid()
    plt.savefig('figures/Dense_scenario3')

    plt.show()
    return xcdf, ycdf

def small_scale_static(data, time_s):
    meas_dB = [[], [], []]
    for i, lst in enumerate(data):
        for d in lst:
            meas_dB[i].append(max(d))

    max_dB = list()
    for m in meas_dB:
        for val in m:
            max_dB.append(val)
    t = np.linspace(0, time_s, len(max_dB))

    start = (5 * len(t) // time_s)
    stop = (10 * len(t) // time_s)
    stop2 = (20 * len(t) // time_s)
    stop3 = (30 * len(t) // time_s)
    max_dB1 = max_dB[:start]
    max_dB2 = max_dB[start:stop]
    max_dB3 = max_dB[stop:stop2]
    max_dB4 = max_dB[stop2:stop3]
    max_dB5 = max_dB[stop3:]

    mean1 = np.mean(max_dB1)
    mean2 = np.mean(max_dB2)
    mean3 = np.mean(max_dB3)
    mean4 = np.mean(max_dB4)
    mean5 = np.mean(max_dB5)
    norm1 = mean1 - mean2
    norm2 = mean1 - mean3
    norm3 = mean1 - mean4
    norm4 = mean1 - mean5
    max_dB_tmp = [max_dB1, max_dB2 + norm1, max_dB3 + norm2, max_dB4 + norm3, max_dB5 + norm4]

    max_dB = list()
    for m in max_dB_tmp:
        for val in m:
            max_dB.append(val)

    logging.info(f"Max: {np.max(max_dB)} dB")
    logging.info(f"Min: {np.min(max_dB)} dB")
    logging.info(f"Median: {np.median(max_dB)} dB")

    plt.plot(t, max_dB)
    plt.xlabel('Time (s)')
    plt.ylabel('Measured RSS (dB)')
    plt.title('Static scenario')
    plt.grid()
    plt.savefig('figures/Static_scenario1_Measured')
    plt.show()
    max_dB = max_dB - np.median(max_dB) # normalize to median

    plt.plot(t, max_dB)
    plt.xlabel('Time (s)')
    plt.ylabel('Normalized RSS (dB)')
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
    percentile = [xcdf[np.floor(ycdf*100) == 1][0], xcdf[np.floor(ycdf*100) == 10][0], xcdf[np.floor(ycdf*100) == 50][0], 
                  xcdf[np.floor(ycdf*100) == 90][0]]
    logging.info(f"Percentile:")
    for i, p in enumerate([1, 10, 50, 90]):
        logging.info(f"{p} %: {percentile[i]}")
    plt.semilogy(xcdf, 100*ycdf, label='Measured', linewidth=3)
    plt.legend()
    plt.grid()
    plt.savefig('figures/Static_scenario3')
    plt.show()
    return xcdf, ycdf

def large_scale(name,measuredRx_df, time_df, distance, d_start=20, d_stop=None, flip=0):
    max_dB = list()
    for row in measuredRx_df.iterrows():
        max_dB.append(max(row[1]))

    #pokud jdmeme k antene-chceme odriznout to u anteny, ne na konci-proto je tu ten flip
    if(flip==1):
        max_dB=np.flip(max_dB)

    distances = np.linspace(1, distance, time_df.shape[1])[:len(max_dB)]
    start = (d_start * len(distances) // distance)
    if d_stop is None:
        stop = (distance * len(distances) // distance)
    else:
        stop = (d_stop * len(distances) // distance)
    distances = distances[start:stop]

    # measured PL
    fspl = 20 * np.log10(1E-3) + 20 * np.log10(1) + 92.45  # free space path lost at 1 m (far field)
    measuredPL = (max_dB[start] - np.array(max_dB[start:stop])) + fspl 
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

    logging.info(f"n = {0.1 * n}")
    logging.info(f"sigma = {sigma} dB")
    logging.info(f"L1 = {L1} dB")

    plt.plot(distances, max_dB[start:stop])
    plt.title('Signal Strength vs. Distance')
    plt.xlabel('Distance [m]')
    plt.ylabel('RSS [dB]')
    plt.grid()
    plt.savefig('figures/Signal_StrengthVDistance_'+name)
    plt.show()

    plt.semilogx(distances, PLpredict, label=f"n={0.1 * n}")
    plt.semilogx(distances, measuredPL)
    plt.title('Mean Path Loss Model vs. Measured Path Loss')
    plt.xlabel('Distance [m]')
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


    #DOLU
    partyzanu_down_time = pd.read_csv('./data/partyzanu_dolu_casy.csv', delimiter=';')
    partyzanu_down_data = pd.read_csv('./data/partyzanu_dolu_data.csv', delimiter=';')
    partyzanu_down_dist = 350  # meters

    terronska_down_time = pd.read_csv('./data/terronska_dolu_casy.csv', delimiter=';')      #pd.read_csv('./data/partyzanu_dolu_casy.csv', delimiter=';')
    terronska_down_data = pd.read_csv('./data/terronska_dolu_data.csv', delimiter=';')  # pd.read_csv('./data/partyzanu_dolu_data.csv', delimiter=';')
    terronska_down_dist = 350  # meters

    logging.info("Large Scale - Partyzanu dolu")
    large_scale("partyzanu_dolu",partyzanu_down_data, partyzanu_down_time, partyzanu_down_dist, d_start=40)
    logging.info("Large Scale - Terronska dolu")
    large_scale("terronska_dolu",terronska_down_data, terronska_down_time, terronska_down_dist, d_start=40, d_stop=200)



    #NAHORU
    partyzanu_up_time = pd.read_csv('./data/partyzanu_nahoru_casy.csv', delimiter=';')
    partyzanu_up_data = pd.read_csv('./data/partyzanu_nahoru_data.csv', delimiter=';')
    partyzanu_up_dist = 350  # meters

    terronska_up_time = pd.read_csv('./data/terronska_nahoru_casy.csv',delimiter=';')
    terronska_up_data = pd.read_csv('./data/terronska_nahoru_data.csv',delimiter=';')
    terronska_up_dist = 350  # meters

    logging.info("Large Scale - Partyzanu nahoru")
    large_scale("partyzanu_nahoru", partyzanu_up_data, partyzanu_up_time, partyzanu_up_dist, d_start=40,flip=1)

    logging.info("Large Scale - Terronska nahoru")
    large_scale("terronska_nahoru", terronska_up_data, terronska_up_time, terronska_up_dist, d_start=40, d_stop=200, flip=1)


    #MERENI UNIKU
    crowd1_data = pd.read_csv('./data/crowd_1_data.csv', delimiter=';')
    crowd1_time_s = 60

    crowd2_data = pd.read_csv('./data/crowd_2_data.csv', delimiter=';')
    crowd2_time_s = 117

    static1_data = pd.read_csv('./data/static_1_data.csv', delimiter=';').values.tolist()
    static2_data = pd.read_csv('./data/static_2_data.csv', delimiter=';').values.tolist()
    static3_data = pd.read_csv('./data/static_3_data.csv', delimiter=';').values.tolist()
    static_data = [static1_data, static2_data, static3_data]
    static_time_s = 58

    logging.info("Bez zastínení")
    xcdf, ycdf = small_scale_static(static_data, static_time_s)

    logging.info("Dynamické zastínění")
    xcdf1, ycdf1 = small_scale_crowd(crowd2_data, crowd2_time_s)

    logging.info("Úplné zastínění")
    xcdf2, ycdf2 = small_scale_dense(crowd1_data, crowd1_time_s)

    x_lin = np.arange(np.power(10.0, -60/20), np.power(10.0, 10/20), 0.01)
    x_log_rayleigh_norm = 20.0*np.log10(x_lin/sst.rayleigh.median())
    y_rayleigh = sst.rayleigh.cdf(x_lin)
    plt.semilogy(x_log_rayleigh_norm, 100*y_rayleigh, label='Rayleigh (Rice k=0)')
    plt.xlabel('Relative loss (dB), normalized (0 dB at 50%)')
    plt.ylabel('Cumulative probability (%)')
    plt.semilogy(xcdf, 100*ycdf, label='Bez zastínení', linewidth=3)
    plt.semilogy(xcdf1, 100*ycdf1, label='Dynamické zastínění', linewidth=3)
    plt.semilogy(xcdf2, 100*ycdf2, label='Úplné zastínění', linewidth=3)
    plt.axvline(x=0)
    plt.axhline(y=50)
    plt.ylim([10E-1, 100])
    plt.xlim([-30, 10])
    plt.legend()
    plt.grid()
    plt.savefig('figures/all_scenario')
    plt.show()