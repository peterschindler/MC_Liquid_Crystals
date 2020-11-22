from __future__ import print_function
import time as time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation
from matplotlib import rc
from IPython.display import HTML, Image
from ipywidgets import interact, interactive, fixed, interact_manual
import glob
import math

############################################################################
# Util functions
############################################################################

# Get logarithmic spaced interval for the simulation if num_points > 100
def get_logaritmic_spacing(num_pts):
  log_intervals = []
  idx = 30
  LOG = 1.5

  if num_pts > 50:
    for i in range(1,idx,2):
      log_intervals.append(i)
    while idx <= num_pts:
      log_intervals.append(idx)
      idx = round(idx*LOG)
      if idx >= num_pts*0.01:
        LOG=1.1
  else:
    for i in range(1, num_pts, 1):
      log_intervals.append(i)

  for i in range(30):
    log_intervals.append(num_pts)

  #print(len(log_intervals))
  return log_intervals

# Calculate approximate area of Pi based on Monte Carlo estimate
def estimate_area_sim(x_pts, y_pts):
  radius_array = np.sqrt(x_pts**2 + y_pts**2)
  inside_circle = (radius_array < 1.0)
  inside_circle_sum = np.sum(inside_circle)
  pi_approx = (inside_circle_sum / len(x_pts)) * 4.0
  return pi_approx, inside_circle


# Calculate value of Pi for each interval
def calculate_pi_approx(num_pts, x_pts, y_pts, log_intervals):
  steps, pi =[], []
  x_pts[0], y_pts[0] = 0.25,0.25 #Init first point inside the circle

  RED_COLOUR = [1,0,0]
  BLUE_COLOUR = [52/255, 152/255, 219/255]
  color_array = np.ones((num_pts,3))*BLUE_COLOUR
  color_array[np.where(np.sqrt(x_pts[:]**2 + y_pts[:]**2) < 1)] = RED_COLOUR

  for i in log_intervals:
    pi_approx, inside_circle = estimate_area_sim(x_pts[:i], y_pts[:i])
    steps.append(i)
    pi.append(pi_approx)
  return steps, pi, color_array


# Function to run the Monte Carlo simulation
def run_mc_simulation(num_pts, x_pts, y_pts):
  log_intervals = get_logaritmic_spacing(num_pts)
  steps, pi, color_array = calculate_pi_approx(num_pts, x_pts, y_pts, log_intervals)
  return  log_intervals, steps, pi, color_array

############################################################################
# Helper plotting functions
############################################################################

# Init plot layout
def init_plots(num_pts, steps, pi, color_array):
  fig, (ax01,ax02) = plt.subplots(1,2,  figsize=(14,7), gridspec_kw={'width_ratios': [1, 2]})
  plt.tight_layout()
  plt.close()
  rc('animation', html='jshtml')

  SCATTER_SIZE = 2 if num_pts < 500 else 1

  x, y, colorMap = [],[],[],
  x1, y1 = [0,0], [0,0]


  # Set scatter plot for plotting the random points in each interval
  sc = ax01.scatter(x,y, s=SCATTER_SIZE)
  ax01.set(xlim=(-1, 1),ylim=(-1,1))
  ax01.set_aspect('equal')

  if num_pts <= 10000:
    circle_plot = plt.Circle( ( 0, 0 ), 1, color='lightgray', linewidth=0.8, alpha=.8, fill=False)
  else:
    circle_plot = plt.Circle( ( 0, 0 ), 1, color='red', linewidth=0.8, alpha=.8, fill=False)

  ax01.add_artist(circle_plot)


  # Set plot for displaying value of Pi with each iteration
  PLOT, = ax02.plot([],[])
  x1[0],x1[1], y1[0], y1[1] = steps[0], steps[-1],np.pi, np.pi
  ax02.plot(x1, y1, 'r')
  if num_pts <= 10000:
    ax02.set(xlim=(steps[0], steps[-1]),ylim=(min(pi) -0.1, max(pi) + 0.1))
  else:
    ax02.set(xlim=(steps[0], steps[-1]),ylim=(3.0, 3.3))

  x0,x1 = ax02.get_xlim()
  y0,y1 = ax02.get_ylim()
  ax02.set_aspect(0.5*abs(x1-x0)/abs(y1-y0))

  text_time = ax01.text(0.05, 1.1,'',horizontalalignment='left',verticalalignment='bottom', transform=ax01.transAxes, fontsize=15)
  text_pi = ax01.text(1.15, 1.1,'',horizontalalignment='left',verticalalignment='bottom', transform=ax01.transAxes, fontsize=15)
  return fig, text_time, text_pi, sc, PLOT

# Update the scatter plot and pi-value plot with each linear/logarithmic interval
def animate(i, x_pts, y_pts, num_pts, log_intervals, steps, pi, color_array, fig, text_time, text_pi, sc, PLOT):
    pts_range = range(log_intervals[i]) if i==0 else range(log_intervals[i-1],log_intervals[i],1)

    for pts in pts_range:
      text_time.set_text('Iteration = %.1d' % (pts+1))

    x = x_pts[:log_intervals[i]]
    y = y_pts[:log_intervals[i]]
    colorMap = color_array[:log_intervals[i]]
    sc.set_offsets(np.c_[x,y])
    if num_pts <= 10000:
      sc.set_color(colorMap)
    text_pi.set_text('Pi = %.3f' % (pi[i]))

    PLOT.set_data(steps[:i], pi[:i])

    return PLOT, sc, text_time, text_pi


# Function to visualize and update plots
def visualize_simulation(x_pts, y_pts, num_pts, fig, log_intervals, steps, pi, color_array):
  fig, text_time, text_pi, sc, PLOT = init_plots(num_pts, steps, pi, color_array)
  # Finally call the plotting function to display the Monte Carlo simulation
  ani = matplotlib.animation.FuncAnimation(fig, animate,
                  frames=len(log_intervals), fargs=(x_pts, y_pts, num_pts, log_intervals, steps, pi, color_array, fig, text_time, text_pi, sc, PLOT,), interval=150, repeat=True, blit=True)
  start = time.time();
  print('Running simulation and generating plots ...')
  vid = ani.to_html5_video()
  print('Took %.1f seconds' % (time.time()-start))
  return vid


############################################################################
# Util functions
############################################################################

# Parse cooling_info file
def init_temp(dirname, newfmt=0):
  fname='MC_Liquid_Crystals/runs/{}/lqs_2_coolinfo.txt'.format(dirname)

  t,  energy , temp, count = [],[],[],0
  file = open(fname)
  for lines in file:
    if(count==0):
        count = count+1
        continue
    lines = lines.replace('[', '')
    lines = lines.replace(']', '')
    if newfmt or (len(lines.split('\t'))==6):
      time, T, e, i,j,k  = lines.split('\t')
    else:
      time, T, e, i  = lines.split('\t')
    t.append(float(time))
    temp.append(float(T))
    energy.append(float(e))
    count=count+1

  t0=t[0]
  for i in range(len(t)):
    t[i] = (t[i] - t0)/1000

  return t, energy, temp

# Function to get order parameter of crystal at a given state
from numpy import linalg as LA

def compute_order_parameter(vecs):
  i = 0
  mat = np.zeros((len(vecs),3,3))
  for v in vecs:
    x,y,z = v
    mat[i] = np.array([[x*x, x*y, x*z],[y*x, y*y, y*z],[z*x, z*y, z*z]])
    i = i+1
  mat_avg = np.mean(mat, axis = 0)
  eig_values, eig_vectors = LA.eig(mat_avg)
  max_idx = np.argmax(eig_values)
  max_eig = eig_vectors[:,max_idx]

# Get order parameter based on variance of structure from it's eigen direction
  var = 0
  for v in vecs:
    dir = np.dot(v, max_eig)
    var += dir*dir

  S = 1.5*var/len(vecs) - 0.5
  return S, max_eig

############################################################################
# Helper plotting functions
############################################################################


# Plot liquid crystal structure
def plot_liquid_crystal(fname, fnum, max_eig, temp, do_plt = 1):
  file=open(fname)
  if do_plt:
    fig = plt.figure(1, figsize=(20,20))
    ax1 = plt.subplot(131)

  locs, vecs = [], []
  for lines in file:
    if 'Sp ' in lines:
      loc, vec = [0,0,0], [0,0,0]
      arr=lines.split(' ')
      locs.append([float(arr[1]),float(arr[2]),float(arr[3])])
      vecs.append([float(arr[4]),float(arr[5]),float(arr[6])])

  N = math.ceil(pow(len(locs),0.33)) if '3d' in fname else pow(len(locs),0.5)

  if do_plt:
    ax1.set_aspect('equal')
    ax1.set(xlim=(-1, N), ylim=(-1, N))
    ax1.patch.set_facecolor((48/255, 10/255, 36/255))

    # Plot crystal points
    for i in range(len(locs)):
      x,y=[0,0],[0,0]
      x[0],y[0]=locs[i][1],locs[i][2]
      x[1],y[1]=locs[i][1]+vecs[i][1]/2,locs[i][2]+vecs[i][2]/2
      ax1.plot(x,y, 'lime')


  # Plot principal eigen vector or prominent direction of the crystal
  if do_plt==0:
    S, max_eig = compute_order_parameter(vecs)
    return S, max_eig
  if do_plt:
    x[0],y[0]=N*0.82, N*0.87
    x[1],y[1]=N*0.82 + max_eig[1] ,N*0.87 + max_eig[2]
    ax1.plot(x,y, 'yellow', linewidth=3)

    ax1.set_xlabel('T = %.2f' % (temp[fnum]), fontsize = 12, c = 'b')
    ax1.xaxis.set_label_position('top')
  return [], []


# Plot state of crystal i.e. Energy and Order parameter vs Temperature(time)
def plot_cystal_state(fnum, t, energy, temp, S):
  do_color=1
  MAP= 'viridis' #viridis #plasma #winter #cool #hot
  bg = None #(48/255, 10/255, 36/255)
  cm = plt.get_cmap(MAP)

  # Plot Energy versus Temperature
  ax3 = plt.subplot(132)

  if do_color:
    if bg!=None:
        ax3.patch.set_facecolor(bg)
    ax3.set_prop_cycle('color',[cm(1.*i/(fnum-1)) for i in range(fnum-1)])

  plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=None)
  x0,x1 = min(temp), max(temp)
  y0,y1 = min(energy)*1.8, max(energy)*1.5
  ax3.set(xlim=(x1, x0),ylim=(y0,y1))
  ax3.set_aspect(abs(x1-x0)/abs(y1-y0))
  if do_color:
    for i in range(fnum-1):
      ax3.plot(temp[i:i+2], energy[i:i+2], linewidth = 2)
  else:
      ax3.plot(temp[:fnum], energy[:fnum])
  ax3.set_xlabel('Temperature', fontsize=15)
  ax3.set_ylabel('Energy', fontsize=15)


  # Plot Order Parameter versus Temperature
  ax2 = plt.subplot(133)
  if do_color:
    if bg!=None:
      ax2.patch.set_facecolor(bg)
    ax2.set_prop_cycle('color',[cm(1.*i/(fnum-1)) for i in range(fnum-1)])

  plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=None)
  x0,x1 = min(t), max(t)
  y0,y1 = min(S)*1.8, max(S)*1.5
  ax2.set(xlim=(x0, x1),ylim=(0,1))
  x0,x1 = ax2.get_xlim()
  y0,y1 = ax2.get_ylim()
  ax2.set_aspect(abs(x1-x0)/abs(y1-y0))

  if do_color:
    for i in range(fnum-1):
      ax2.plot(t[i:i+2], S[i:i+2], linewidth = 2)
  else:
      ax2.plot(t[:fnum], S[:fnum])

  ax2.set_xlabel('Time', fontsize=15)
  ax2.set_ylabel('Order parameter', fontsize=15)


# Piecing it together
def plot_lc_simulation(dir, newfmt=0):

  files = sorted(glob.glob("MC_Liquid_Crystals/runs/{}/*.xyz".format(dir)))
  l=len(files)
  t, energy, temp = init_temp(dir, newfmt)
  S_all, eigen_all = [], []
  for f in files:
    S, max_eig = plot_liquid_crystal(f, None, None, None, do_plt = 0)
    S_all.append(S)
    eigen_all.append(max_eig)
  plt.show()

  def f(Frame=0):
      S = plot_liquid_crystal(files[Frame], Frame, eigen_all[Frame], temp)
      plot_cystal_state(Frame, t, energy, temp, S_all)
      plt.show()

  interact(f, Frame=(0,l-2), value=0);
