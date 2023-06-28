# Lab 1: Telescope Proposal

## Lab 1 Calculations

#For this lab, we'll do the calculations for items 5 (signal to noise and exposure time) and 6 (total time) in this notebook.

# Exposure time to reach a desired signal-to-noise ratio

#In this calculation we have the experience of a reference observation to draw upon, which constraints the relationships beteween source brightness, exposure time, and signal-to-noise ratio.

#First you'll have to do some algebraic manipulation.  Then fill in the code below.

# define variables for reference observation
m_ref = 12.41 # [mag]
t_ref = 700. # [s]   NOTE  here we use a period to indicate the number is not necessarily an integer.  This is a good habit.
SNR_ref = 151.
# constant for flux-magnitude relationship
F0 = 3.530e-20 # [erg/s/cm2]

F_ref = F0 * 10.**(-m_ref/2.5) # [erg/s/cm2]

# parameters for our observation
SNR_targ = 10.
m_targ = 11.16 # [mag]  NOTE here you need to fill in the expression, or there will be an error here.
F_targ = F0 * 10.**(-m_targ/2.5)
t_targ = ((SNR_targ**2)*(F_ref**2)*(t_ref))/((SNR_ref**2)*(F_targ**2)) # [s]  NOTE here you need to fill in the expression, or there will be an error here.

# Let's print the results.  Notice the funny syntax.  Every instance of {} gets replaced with the argument of the format statement.
print("Target magnitude: {}".format(m_targ))
print("Exposure time to reach SNR={}: {}".format(SNR_targ, t_targ))


### Total time for observations

#Here we're going to calculate the total time needed.  We're getting data in 3 filters ($B$, $V$, and $R$), with 10 s of overhead per exposure.  We need to calculate a couple of quantities, namely
 #* the time for obtaining 3 exposures, one in each filter, and
 #* once we know the pulsation period, how many exposures in each filter we expect to obtain.
 

t_over = 10. # [s] overhead time
n_filt = 3 # number of filters
t_pulse = 88*60  # [s] pulsation period  NOTE here you need to fill in the expression, or there will be an error here.  Also note the units!

t_3exp = n_filt * (t_targ+t_over)  # [s] time for one exposure in each filter.  NOTE here you need to fill in the expression, or there will be an error here.
print() #  NOTE  print out the time

import numpy as np
# the np.round() function rounds the value to the nearest integer value (though it still returns a floating point value, e.g. 5.0)
# the int() function converts the value to an integer
n_meas = int(np.round(t_pulse / t_3exp )) # number of measurements in each filter.  NOTE here you need to fill in the expression, or there will be an error here.
print(n_meas) #  NOTE  print out the number of measurements

