# This python script accompanies the 2020 Science Advances study of England, Wagner and Eisenman entitled Modeling the Breakup of Tabular Icebergs
# This is based off of the analytical model of iceberg drift of Wagner et al 2017.
# Our addition is incorporating a probabilistic breakup scheme, in which child icebergs calve from the parent iceberg according to a Poisson distribution each day
# In addition to tracking the large iceberg, this script also keeps track of the bulk properties of the small child icebergs.
# We use ECCO2 for the ocean variables and ERA5 for the atmospheric variables, however the user can load their desired datasets.
# The user must also specify the starting dimensions of the parent iceberg as well as its starting location

# ============================================================================================
# IMPORT PACKAGES
# ============================================================================================
import scipy.io as sio # this is for reading matlab files
import numpy as np # used for numeric databases
import netCDF4 as nc
import os
import Nio
import random
from scipy.stats import norm

# ============================================================================================
# DEFINE NEW FUNCTION
# ============================================================================================

def closest_node(node, nodes):
    nodes = np.asarray(nodes)
    dist_2 = (nodes - node)**2
    return np.argmin(dist_2)

# ============================================================================================
# LOAD THE VARIABLES
# ============================================================================================

# Directories (local hard drive)
SST_dir = '/Users/markengland/Documents/Research/Icebergs_2019/WDE17v2/run_variables/'
sic_dir = '/Users/markengland/Documents/Research/Icebergs_2019/WDE17v2/run_variables/'
u_ocn_dir = '/Users/markengland/Documents/Research/Icebergs_2019/WDE17v2/run_variables/'
v_ocn_dir = '/Users/markengland/Documents/Research/Icebergs_2019/WDE17v2/run_variables/'
u_atm_dir = '/Users/markengland/Documents/Research/Icebergs_2019/WDE17v2/run_variables/'
v_atm_dir = '/Users/markengland/Documents/Research/Icebergs_2019/WDE17v2/run_variables/'

fname = SST_dir + "SST_1992_2019.nc"
f = Nio.open_file(fname)
sst = f.variables["SST"][:,:,:]
sst = np.where(sst<-4.0,-4.0,sst) # There is an issue where crazy values were being read as SSTs
time = f.variables["time"][:]
latitude = f.variables["lat"][:]
longitude = f.variables["lon"][:]
msk = f.variables["msk"][:,:]

fname = u_ocn_dir + "u200_ocn_1992_2019.nc"
f2 = Nio.open_file(fname)
u_ocn = f2.variables["u_ocn"][:,:,:]
del(fname)

fname = sic_dir + "sic_1992_2019.nc"
f6 = Nio.open_file(fname)
sic = f6.variables["sic"][:,:,:]
del(fname)

fname = v_ocn_dir + "v200_ocn_1992_2019.nc"
f3 = Nio.open_file(fname)
v_ocn = f3.variables["v_ocn"][:,:,:]
del(fname)

fname = u_atm_dir + "u_atm_1992_2019.nc"
f4 = Nio.open_file(fname)
u_atm = f4.variables["u_atm"][:,:,:]
del(fname)
u_atm_clim = np.mean(u_atm,axis=0)

fname = v_atm_dir + "v_atm_1992_2019.nc"
f5 = Nio.open_file(fname)
v_atm = f5.variables["v_atm"][:,:,:]
del(fname)
v_atm_clim = np.mean(v_atm,axis=0)

# Pick the desired time period
u_ocn = u_ocn[0:3286,:,:]
v_ocn = v_ocn[0:3286,:,:]
u_atm = u_atm[0:3286,:,:]
v_atm = v_atm[0:3286,:,:]
sst = sst[0:3286,:,:]
sic = sic[0:3286,:,:]
time = time[0:3286]

# ============================================================================================
# CONSTANTS
# ============================================================================================

R = 6378e3           # Radius of earth [m]
Om = 7.2921e-5       # Rotation rate of earth [rad/s]
g = 9.81               # Acceleration due to gravity [m/s]

# Densities
rhow = 1027             # Density of water [kg/m^3]
rhoa = 1.2              # Density of air [kg/m^3]
rhoi = 850              # Density of shelf ice [kg/m^3] from Silva et al (2006)
drho = rhow - rhoi
nu = 0.33               # Poisson ratio of ice

# Bulk coefficient
Cw = 0.9                # Bulk coefficient of water from Bigg et al (1997)
Ca = 1.3                # Bulk coefficient of air  from Bigg et al (1997)

# Melt parameters
Ti = -4
a1 = 8.7e-6
a2 = 5.8e-7
b1 = 8.8e-8
b2 = 1.5e-8
c = 6.7e-6

# ============================================================================================
# ADDITIONAL FUNCTIONS
# ============================================================================================

ff = lambda lat: 2*Om*(np.sin(lat*np.pi/180)) 		# Coriolis
ga = np.sqrt(rhoa*drho/rhow/rhoi*Ca/Cw)             # gamma equation 7
S = lambda l,w: l*w/(l+w)                           # harmonic mean length
La = lambda u,lat,S: Cw*ga/ff(lat)*u/S/np.pi 		# Lambda equation 9
a = lambda La: 1/(2*La**3)*(np.sqrt(1+4*La**4)-1)	# alpha equation 8
b = lambda La,lat: ff(lat)/abs(ff(lat))*1/(np.sqrt(2)*La**3)*np.sqrt(np.abs((1+La**4)*np.sqrt(1+4*La**4)-3*La**4-1))		# beta equation 8
b_approx = lambda La,lat: ff(lat)/abs(ff(lat))*(La**3-3/2*La**7 )            # To avoid numerical issues when lambda is small
B = lambda h0: (E*h0**3)/12/(1-nu**2)				# Bending stiffness of the beam
lw = lambda Bb: (Bb/g/rhow)**0.25					# Buoyancy length (Wagner et al 2014, GRL)

# ============================================================================================
# SPACE AND TIME
# ============================================================================================

lat_diff = latitude[1] - latitude[0]				# Latitude step
lon_diff = longitude[1] - longitude[0]				# Longitude step
minlat = np.min(latitude)							# Minimum latitude
maxlat = np.max(latitude)							# Maximum latitude
minlon = np.min(longitude)							# Minimum latitude
maxlon = np.max(longitude)							# Maximum latitude
DT = 3												# Data is every 3 days
final_t = 3287*4									# Loop through 4 times
startrange = np.round(np.linspace(1,3287,trajnum))  # evenly spaced seeding
dt = 24*3600*tstep                                  # model timestep in seconds
dtR = dt/R*180/np.pi                                # ratio for distances
t = np.arange(1,final_t+tstep,tstep)                # input time
nt = int(len(t) * DT/tstep) 
tt = np.linspace(1,len(t),nt) *tstep                # Model time
t1 = t-tstep

# ============================================================================================
# DEFINE MAIN ICEBERG FUNCTION
# ============================================================================================


def iceberg_trajectory_parent(bergdims,traj_number,start_location,start_time,p):
    # set output arrays (of unknown dimensions)
    XIL = np.empty([nt,1]) 
    YIL = np.empty([nt,1]) 
    VOL = np.empty([nt,1])
    LENGTH = np.empty([nt,1])
    WIDTH = np.empty([nt,1])
    HEIGHT = np.empty([nt,1])
    MELT = np.empty([nt,1])
    num_breaks = [0] * nt 
    iceberg_endtime = np.empty([1])
    iceberg_time = np.empty([1])
    iceberg_start_time = np.empty([1])
    iceberg_no = np.empty([1])
    kk = traj_number
    
    # Initialise the iceberg dimensions
    L = bergdims[0]
    W = bergdims[1]
    H = bergdims[2]
    
    # Run drift and melt
    mm=0 # melted icebergs
    ss=0 # survived icebergs
    ob=0 # escaped icebergs
            
    ts = startrange[kk] # pick seeding time of input field
    tts = ts*DT/tstep # trajectory start time of model
    lt = nt-tts # trajectory run length
    
    # temporary variables
    xil = np.empty([lt.astype(int)])
    yil = np.empty([lt.astype(int)])
    v = np.empty([lt.astype(int)])
    melt = np.empty([lt.astype(int)])    
    xil[start_time] = start_location[0]
    yil[start_time] = start_location[1]       
    # Initial berg dimensions
    l = np.ones([lt.astype(int)])*L
    w = np.ones([lt.astype(int)])*W
    h = np.ones([lt.astype(int)])*H
    v[start_time] = L*W*H # Initial volume
        
    #integrate as long as the iceberg is in the domain and not melted and over the time period
    i = start_time
    outofbound = 0
    melted = 0
    fractured = 0
    ti2 = 0

    while outofbound ==0 and melted ==0 and i<(lt.astype(int)-2) and ti2<13143:
        i += 1 
        # ==============================================
        # Computes iceberg drift component WDE17 sec. 3
        # ==============================================
            
        # Find nearest neighbour of surface condition field
        XI = closest_node(xil[i-1],longitude) 
        YI = closest_node(yil[i-1],latitude)
            
        # interpolate fields linearly between timesteps
        timestep = tt[tts.astype(int)+i-1]
        ti1  = np.floor(timestep)
        ti2 = ti1+1
        dt1 = timestep-ti1
        dt2 = ti2-timestep
        ua = u_atm[(ti1.astype(int)-1)%3286,YI,XI]*dt1+u_atm[(ti2.astype(int)-1)%3286,YI,XI]*dt2 # Atmospheric u wind
        va = v_atm[(ti1.astype(int)-1)%3286,YI,XI]*dt1+v_atm[(ti2.astype(int)-1)%3286,YI,XI]*dt2 # Atmospheric v wind
        uw = u_ocn[(ti1.astype(int)-1)%3286,YI,XI]*dt1+u_ocn[(ti2.astype(int)-1)%3286,YI,XI]*dt2 # Upper ocean u currents
        vw = v_ocn[(ti1.astype(int)-1)%3286,YI,XI]*dt1+v_ocn[(ti2.astype(int)-1)%3286,YI,XI]*dt2 # Upper ocean v currents
        SST= sst[(ti1.astype(int)-1)%3286,YI,XI]*dt1+sst[(ti2.astype(int)-1)%3286,YI,XI]*dt2     # SST
        SIC= sic[(ti1.astype(int)-1)%3286,YI,XI]*dt1+sic[(ti2.astype(int)-1)%3286,YI,XI]*dt2     # sea ice concentration

        # compute wind speed and Lambda at location (for given iceberg size)
        Ua = np.sqrt(ua**2+va**2);
        LA = La(Ua,yil[i-1],S(l[i-1],w[i-1]));
            
        # Compute the iceberg velocity, equation 6, possibly uses approximation for beta due to numerical issues
        if abs(LA)>0.2:
            ui = uw + ga*(a(LA)*va+b(LA,yil[i-1])*ua);
            vi = vw + ga*(-a(LA)*ua+b(LA,yil[i-1])*va);
        else:
            ui = uw + ga*(a(LA)*va+b_approx(LA,yil[i-1])*ua);
            vi = vw + ga*(-a(LA)*ua+b_approx(LA,yil[i-1])*va);
            
        # iceberg translation (convert from m to deg lat/lon)
        dlon = ui*dtR;
        dlat = vi*dtR;

        # advance the iceberg
        yil[i] = yil[i-1] + dlat;
        xil[i] = xil[i-1] + dlon/np.cos((yil[i]+yil[i-1])/2*np.pi/180);
            
        # Check you haven't gone out of bounds
        if yil[i]>maxlat or yil[i]<minlat:
            outofbound = 1
            ob+=1
        else:
            yi2 = closest_node(yil[i],latitude) # Find closest latitude coordinate
            if xil[i]>maxlon: 					# Check if iceberg has looped out of bounds
                xil[i] = xil[i]-360
            elif xil[i]<minlon:
                xil[i] = xil[i]+360
            xi2 = closest_node(xil[i],longitude)
            
            if msk[yi2,xi2] == 1: 				# Don't let iceberg run into the continent, move it back to original position
                yil[i] = yil[i-1]
                xil[i] = xil[i-1]
                    
        # =================================================
        # Computes iceberg melting component WDE17 appendix
        # =================================================
            
        # Compute melt terms in equations 2, 3 of England et al 2020
        Me = (a1*Ua**0.5 + a2*Ua)*(0.5+0.5*np.cos(3.1416*SIC**3))*(0.33333*(SST+2)) # This adds in a cosine reliance on sea ice conc.
        Mv = b1*max(SST,0.0) + b2*max(SST,0.0)**2 # This ensures Mv does not go negative
        Mb = (c*(np.sqrt((ui-uw)**2+(vi-vw)**2))**0.8)*(SST-Ti)*(l[i-1])**-0.2
            
        # Apply melt rates, reducing iceberg size
        dldt = -Mv -Me
        dhdt = -Mb
        l[i] = l[i-1]+dldt*dt
        w[i] = w[i-1] +dldt*dt
        h[i] = h[i-1] +dhdt*dt
        melt[i] = l[i]*w[i]*h[i] - l[i-1]*w[i-1]*h[i-1] 
            
        # Make sure the iceberg is not negative size
        if l[i]<=0 or w[i]<=0 or h[i]<=0:
            l[i] = 0
            w[i] = 0
            h[i] = 0
            melted = 1
            mm +=1
                
        # Rollovers in equation A2
        if w[i]>0 and h[i]>0:
            if w[i]/h[i] < np.sqrt(6*rhoi/rhow*(1-rhoi/rhow)):
                w[i], h[i] = h[i], w[i]
            
        # Make sure length is greater than width
        if w[i]>l[i]:
            w[i], l[i] = l[i], w[i]

        break_number = 0
        iceberg_no[0] =  1

        # Random number to see if iceberg breaks up
        # Set up the poisson probabilities
        # Only release child icebergs every 'break day', this is to save on computation power, currently this is set at 1day so has no impact
        xmax = 3.14/(2**1.5)*lw(B(h[i]))
        if p>0.0 and l[i]>(3*xmax) and SIC<0.5 and ti1%break_days==0.0 and tt[tts.astype(int)+i-2]<ti1: # Check probability of fracture isn't zero and iceberg isnt too small
    
            pfactor = float(break_days*p)
            if pfactor<20:
            	# Poisson distribution
                s = np.arange(0,DT*20)
                sfactorial = np.append([1],np.cumprod(s[1:]),axis=0)
                f1 = np.exp(-float(break_days*p))
                f2 = ((float(break_days)*p)**s)
                f3 = sfactorial
                p_dist = f1*f2/f3 # Equation 4 of England et al 2020
                aa = np.where(p_dist<1e-8)[0][0]
                p_dist[aa.astype(int):] = 0.00
                p_dist[0] = 1.0-np.sum(p_dist[np.arange(1,DT*20)])
                break_number = np.random.choice(s,1,p=p_dist) # Pick number of child icebergs to break off using a poisson distribution
                num_breaks[start_time+i]=break_number[0]
            else:
            	# Normal distribution
                s = np.arange(0,DT*300)
                p_dist = norm.pdf(s,pfactor,np.sqrt(pfactor))
                p_dist[0] = 1.0-np.sum(p_dist[np.arange(1,DT*300)])
                if p_dist[0] <0:
                    p_dist[0]=0
                break_number = np.random.choice(s,1,p=p_dist) # Pick number of child icebergs to break off using a normal distribution to approximate a poisson distribution
                num_breaks[start_time+i]=break_number[0]
#             break_number = [break_days*p, 0]
            if break_number[0]>0: # If a break event occurs
                iceberg_dummy = np.empty([1])
                iceberg_dummy[0] = break_number[0]
                iceberg_no = np.append(iceberg_no,iceberg_dummy,axis=0)
                del(iceberg_dummy)
                fractured+=1
                xmax = 3.142/(2**1.5)*lw(B(h[i])) # Calculate the break off length from Wagner et al 2014

                # Dimensions of child iceberg
                l2 = xmax
                w2 = epsilon * xmax
                h2 = h[i]
                A2 =l2*w2
                # Calculate dimensions of parent iceberg
                lnew = l[i]-break_number[0]*A2/w[i]
                while lnew<(3*xmax): # Stop breaking the parent iceberg up once it gets too small
                    break_number[0] += -1
                    lnew = l[i]-break_number[0]*A2/w[i]
                l[i] = lnew
                num_breaks[start_time+i]=break_number[0]

                # Make sure length is greater than width
                if w2>l2:
                    w2, l2 = l2, w2 # else switch

                # Rollovers in equation A2
                if w2/h2 < np.sqrt(6*rhoi/rhow*(1-rhoi/rhow)):
                    w2, h2 = h2, w2 # else switch

                xil2 = xil[i]
                yil2 = yil[i]

                # Generate and track child icebergs (this uses a nested version of the original function, INCEPTION)
                XIL2,YIL2,VOL2,LENGTH2,WIDTH2,HEIGHT2,MELT2,iceberg_endtime2,iceberg_time2,iceberg_start_time2,num_breaks2,iceberg_no2 = iceberg_trajectory_parent([l2,w2,h2],kk,[xil2,yil2],i,0)
                
                # Append the child iceberg to the parent iceberg
                XIL = np.append(XIL,XIL2,axis=1)
                YIL = np.append(YIL,YIL2,axis=1)
                VOL = np.append(VOL,VOL2,axis=1)
                LENGTH = np.append(LENGTH,LENGTH2,axis=1)
                WIDTH = np.append(WIDTH,WIDTH2,axis=1)
                HEIGHT = np.append(HEIGHT,HEIGHT2,axis=1)
                MELT = np.append(MELT,MELT2,axis=1)
                iceberg_endtime = np.append(iceberg_endtime,iceberg_endtime2,axis=0)
                iceberg_time = np.append(iceberg_time,iceberg_time2,axis=0)
                iceberg_start_time = np.append(iceberg_start_time,iceberg_start_time2,axis=0)
                del(xil2)
                del(yil2)

                    
        # Make sure length is greater than width
        if w[i]>l[i]:
            w[i], l[i] = l[i], w[i]  
        # Compute new volume
        v[i] = l[i] * h[i] * w[i]
    
    # Store the iceberg location and dimensions            
    ind = np.arange(start_time,i+1,1)
    XIL[ind,0] = xil[ind] # Store trajectory
    YIL[ind,0] = yil[ind] # Store trajectory
    VOL[ind,0] = v[ind]   # Store volume
    LENGTH[ind,0] = l[ind]   # Store length
    WIDTH[ind,0] = w[ind]   # Store length
    HEIGHT[ind,0] = h[ind]   # Store length
    MELT[ind,0] = melt[ind]   # Store melt
    iceberg_time[0] = len(ind)
    iceberg_start_time[0] = start_time

    if v[ind[-1]]>(min_volume): # Only track to a certain volume
        iceberg_endtime[0] = iceberg_start_time[0] + iceberg_time[0]
    else: # Track to end if it isn't small enough
        iceberg_endtime[0] = iceberg_start_time[0] + np.argmax(v[ind]<(min_volume))

    # Clear variables
    XI = None
    YI = None
    lt = None
    xil = None
    yil = None
    v = None
    h = None
    w = None
    l = None
    return[XIL,YIL,VOL,LENGTH,WIDTH,HEIGHT,MELT,iceberg_endtime,iceberg_time,iceberg_start_time,num_breaks,iceberg_no]

# ============================================================================================
# SEEDING AND STARTING ICEBERG DIMENSION
# ============================================================================================
# Seeding Group
seedgroup = "Group4"
sizegroup = "f_p4_200m"
tstep = 1

prob = 4 # Average number of breakups per day
trajnum = 1000 # The number of trajectories of parent icebergs to simulate
E = 0.1e9# Youngs modulus
epsilon = 3
break_days = int(1) # You break every x days
# START_VOL = [0.69e3,0.46e3,175] # Group a
# START_VOL = [1.22e3,0.82e3,200] # Group b
# START_VOL = [2.18e3,1.45e3,225] # Group c
# START_VOL = [3.87e3,2.58e3,250] # Group d
# START_VOL = [6.89e3,4.59e3,275] # Group e
START_VOL = [12.25e3,8.16e3,300] # Group f
# START_VOL = [21.78e3,14.52e3,325] # Group g
# START_VOL = [38.73e3,25.82e3,350] # Group h

# Biggs distribution
# START_VOL = [60,40,40] # Group A
# START_VOL = [100,67,67] # Group B
# START_VOL = [200,133,133] # Group C
# START_VOL = [350,175,175] # Group D
# START_VOL = [500,333,250] # Group E
# START_VOL = [700,467,250] # Group F
# START_VOL = [900,600,250] # Group G
# START_VOL = [1200,800,250] # Group H
# START_VOL = [1600,1067,250] # Group I
# START_VOL = [2200,1467,250] # Group J

ndays = int(5000)
min_volume = 6e6 # This is the minimum volume we track until

# The release locations are loaded. The user will need to specify the location
seed_input = sio.loadmat("/Users/markengland/Documents/Research/Icebergs_2019/WDE17_Antarctic/python_notebook/"+seedgroup+"_Seed")
# cycle through locations 500 times
seed_X = np.tile(np.ndarray.flatten(np.transpose(seed_input['Seed_X'])),500)
seed_Y = np.tile(np.ndarray.flatten(np.transpose(seed_input['Seed_Y'])),500)

# ============================================================================================
# PRE-ALLOCATE VARIABLES
# ============================================================================================

FW_lat = np.arange(-90.0,91.0,1.0)
FW_lon = np.arange(-360,360,1.0) # This is a trick so don't have to worry about which longitude to use
FW_parents = np.zeros((len(FW_lon),len(FW_lat)))
FW_children = np.zeros((len(FW_lon),len(FW_lat)))
V_parents = np.zeros((len(FW_lon),len(FW_lat)))
V_children = np.zeros((len(FW_lon),len(FW_lat)))
parents_number = 0
children_number = 0
nbreaks = np.zeros((trajnum, int(ndays/tstep)))
VOL_total_ndays = np.zeros((trajnum, int(ndays/tstep)))
XIL_total_ndays = np.zeros((trajnum, int(ndays/tstep)))
YIL_total_ndays = np.zeros((trajnum, int(ndays/tstep)))
LENGTH_total_ndays = np.zeros((trajnum, int(ndays/tstep)))
WIDTH_total_ndays = np.zeros((trajnum, int(ndays/tstep)))
HEIGHT_total_ndays = np.zeros((trajnum, int(ndays/tstep)))
start_time_total_ndays = np.zeros((trajnum))
end_time_total_ndays = np.zeros((trajnum))

# ============================================================================================
# RUN THE ICEBERGS!!
# ============================================================================================

for pp in np.arange(0,trajnum,1): # Loop over number of trajectories
	# Use the function to simulate the parent iceberg and its children
    XIL_parent,YIL_parent,VOL_parent,LENGTH_parent,WIDTH_parent,HEIGHT_parent,MELT_parent,iceberg_endtime_parent,iceberg_time_parent,iceberg_start_time_parent, child_breaks, iceberg_no_plot =\
    iceberg_trajectory_parent(START_VOL,pp,[longitude[seed_X[pp]],latitude[seed_Y[pp]]],0,prob)
    for gg in np.arange(np.shape(XIL_parent)[1]-1,-1,-1):
    	# Start and end time
        ind1 = iceberg_start_time_parent[gg].astype(int)
        ind2 = min((iceberg_endtime_parent[gg]).astype(int),int(ndays/tstep))
        if gg==0: # Only store exact trajectories of parent iceberg
            VOL_total_ndays[pp,ind1:ind2] = VOL_parent[ind1:ind2,gg]
            XIL_total_ndays[pp,ind1:ind2] = XIL_parent[ind1:ind2,gg]
            YIL_total_ndays[pp,ind1:ind2] = YIL_parent[ind1:ind2,gg]
            LENGTH_total_ndays[pp,ind1:ind2] = LENGTH_parent[ind1:ind2,gg]
            WIDTH_total_ndays[pp,ind1:ind2] = WIDTH_parent[ind1:ind2,gg]
            HEIGHT_total_ndays[pp,ind1:ind2] = HEIGHT_parent[ind1:ind2,gg]
            start_time_total_ndays[pp] = iceberg_start_time_parent[gg]
            nbreaks[pp,ind1:ind2] = child_breaks[ind1:ind2]
            end_time_total_ndays[pp] = iceberg_endtime_parent[gg]
            parents_number +=1
            print(pp)
            children_number+=np.sum(child_breaks)
        # Calculate the 
        for nn in np.arange(ind1,ind2,1,int):
            XI_FW = closest_node(XIL_parent[nn,gg],FW_lon) 
            YI_FW = closest_node(YIL_parent[nn,gg],FW_lat)
            if gg==0: # Parent iceberg
                FW_parents[XI_FW,YI_FW] += MELT_parent[nn,gg] # FW of parent icebergs
                V_parents[XI_FW,YI_FW] += VOL_parent[nn,gg] # Volume of parent icebergs
            else: # Save bulk properties of child icebergs
                FW_children[XI_FW,YI_FW] += MELT_parent[nn,gg]*iceberg_no_plot[gg] # FW of child icebergs
                V_children[XI_FW,YI_FW] += VOL_parent[nn,gg]*iceberg_no_plot[gg] # Volume of child icebergs

# Trick to align the freshwater grid for plotting and the output of the iceberg model
FW_A = FW_parents[180:540,:]
FW_B = FW_children[180:540,:]
FW_A[0:180,:] = FW_A[0:180,:] + FW_parents[540:720,:]
FW_B[0:180,:] = FW_B[0:180,:] + FW_children[540:720,:]
FW_A[180:360,:] = FW_A[180:360,:] + FW_parents[0:180,:]
FW_B[180:360,:] = FW_B[180:360,:] + FW_children[0:180,:]
V_A = V_parents[180:540,:]
V_B = V_children[180:540,:]
V_A[0:180,:] = V_A[0:180,:] + V_parents[540:720,:]
V_B[0:180,:] = V_B[0:180,:] + V_children[540:720,:]
V_A[180:360,:] = V_A[180:360,:] + V_parents[0:180,:]
V_B[180:360,:] = V_B[180:360,:] + V_children[0:180,:]

# ============================================================================================
# SAVE NC FILE
# ============================================================================================

trajectories = np.linspace(1,trajnum,trajnum)
#-- open new netCDF file
os.system("rm -rf Output_version2/output_"+seedgroup+sizegroup+".nc") #-- delete file
outf = Nio.open_file("Output_version2/output_"+seedgroup+sizegroup+".nc","c")
#-- create dimensions time, lat and lon
outf.create_dimension('time',None)
outf.create_dimension('lat',181)
outf.create_dimension('lon',360)
outf.create_dimension('trajectories',trajnum)
#-- create dimension variables
outf.create_variable('time','d',('time',))
outf.create_variable('lon','d',('lon',))
outf.create_variable('lat','d',('lat',))
outf.create_variable('trajectories','d',('trajectories',))
#-- create variable t
outf.create_variable('Volume','d',('time','trajectories'))
outf.create_variable('X','d',('time','trajectories'))
outf.create_variable('Y','d',('time','trajectories'))
outf.create_variable('Width','d',('time','trajectories'))
outf.create_variable('Height','d',('time','trajectories'))
outf.create_variable('Length','d',('time','trajectories'))
outf.create_variable('Start_Time','d',('trajectories',))
outf.create_variable('End_Time','d',('trajectories',))
outf.create_variable('FW_Parent','d',('lon','lat',))
outf.create_variable('FW_Child','d',('lon','lat',))
outf.create_variable('Vol_Parent','d',('lon','lat',))
outf.create_variable('Vol_Child','d',('lon','lat',))
outf.create_variable('N_breaks','d',('time','trajectories',))
#-- write data to new file (assign values)
st = int(1/tstep)
outf.variables['lat'].assign_value(np.arange(-90.0,91.0,1.0))
outf.variables['lon'].assign_value(np.arange(-180.0,180.0,1.0))
outf.variables['time'].assign_value(t[0:ndays])
outf.variables['trajectories'].assign_value(trajectories)
outf.variables['Volume'].assign_value(VOL_total_ndays[:,::st].transpose())
outf.variables['X'].assign_value(XIL_total_ndays[:,::st].transpose())
outf.variables['Y'].assign_value(YIL_total_ndays[:,::st].transpose())
outf.variables['Width'].assign_value(WIDTH_total_ndays[:,::st].transpose())
outf.variables['Height'].assign_value(HEIGHT_total_ndays[:,::st].transpose())
outf.variables['Length'].assign_value(LENGTH_total_ndays[:,::st].transpose())
outf.variables['N_breaks'].assign_value(nbreaks[:,::st].transpose())
outf.variables['Start_Time'].assign_value(start_time_total_ndays.transpose())
outf.variables['End_Time'].assign_value(end_time_total_ndays.transpose())
outf.variables['FW_Parent'].assign_value(FW_A)
outf.variables['FW_Child'].assign_value(FW_B)
outf.variables['Vol_Parent'].assign_value(V_A)
outf.variables['Vol_Child'].assign_value(V_B)
#-- close output stream
outf.close()