## Learning trade-off in MGM
'''
Here we try to show that for any given sequence of observations 
$$ x_0,x_1,x_2,...,x_t $$ we have a spectrum of explanatory models.

Stochasticity in the observations is explained by different processes 
in different models for which the likelihood of the observations may be tied.

Specifically we describe an axis in model space defined by two aspects: 
i) the number of models that may be needed to be combined in order to 
explain the osbervations (implied in this are the dynamics of how the 
different models are combined in time) and 
ii) the stochasticity of any one of the model componens (whether a model 
needs to account for all observations).

At the two extremes of this axis, for explaining a stochastic sequence, 
we have on one extreme 
i) a bunch of fully deterministic models that  together explain all 
observations (i.e. stochasticity is absorbed in the switches of model 
from any time-point to the following time-point) and on the other extreme, 
we have ii) a single model that has limited predictivity power (i.e. some 
observations are irreducibly stochastic).
'''

# import libraries
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy as sp
import numpy.random as rnd

## Create all dominant patterns within the CTW generative family
'''
Briefly, for any one model we can compute a transition probability matrix,
 $T_m$ , on the basis of its parameters:

$ T_{\theta^m}  =  \alpha_1*(W_p) + \alpha_2*D^{-1}(E - I - W_p) + \alpha_3*(I)$ 

, where $W_p$ is one of six possible permutation matrices, 
$E$ is a matrix of ones, and $I$ is the identity matrix, 
and $ D^{-1} $ is a diagonal matrix that ensures the rows 
add up to 1, with the constraints that 
$ \alpha_1 + \alpha_2 + \alpha_3 = 1$ and $ \alpha_1>\alpha_2> \alpha_3>0 $. 

More concretely, if  $ \alpha_1 = 0.66 $ , $ \alpha_2 = 0.33 $ and $ \alpha_3 = 0.01 $,
then for the following permutation matrix,

$\begin{equation} 
W^m_p = \begin{bmatrix}  0  &  0  & 1 & 0 \\
0 &  0  &   0  &  1 \\
0  &  1  &  0  &   0 \\
1  &   0  &  0 & 0 \end{bmatrix} 
\end{equation}$

we obtain the following transition probability matrix:

$ T_{\theta^m}   =   \alpha_1\begin{bmatrix}  0  &  0  & 1 & 0 \\
0 &  0  &   0  &  1 \\
0  &  1  &  0  &   0 \\
1  &   0  &  0 & 0 \end{bmatrix} + \alpha_2\begin{bmatrix}  0  &  \tfrac{1}{2}  &  0 & \tfrac{1}{2} \\
\tfrac{1}{2} &  0  &   \tfrac{1}{2}  &  0 \\
\tfrac{1}{2}  &  0  & 0  &   \tfrac{1}{2} \\
0  &   \tfrac{1}{2}  &  \tfrac{1}{2} &  0 \end{bmatrix} + \alpha_3\begin{bmatrix}  1  & 0  &  0 & 0 \\
0 &  1  &  0  &  0 \\
0  &  0  &  1  &   0 \\
0  &   0  &  0 &  1 \end{bmatrix}  = \begin{bmatrix} 0.01  & 0.16  &  0.66  &  0.16 \\
0.16  &  0.01  &  0.16  &  0.66 \\
0.16  &  0.66  &  0.01  &  0.16 \\
0.66  &  0.16  &  0.16  &  0.01 \end{bmatrix}  $ 

Below we create all the permutation matrices within the CTW generative family
'''

# start with a vector of zeros (of with 6*4*4 elements)
Wps = np.zeros([6*4*4])
# manually define all the cells that contain a dominant transition
wh_dom = np.array([3,6,8,13,17,23,24,30,34,36,43,45,50,55,57,60,65,70,75,76,83,84,89,94])
Wps[wh_dom] = 1
# reshape into a model*valuein*valueout array
Wps = np.reshape(Wps,[6,4,4])
# select the fourth pattern model as the dominant pattern model
Wp = Wps[3]

'''
Here, we define the true generative model, as a matrix of transition probabilities from one state to another. We make use of the (already defined) true dominant pattern and we define and use the true dominant transition probability value:

$ \alpha = 0.75 $
'''

alpha = 0.75
E = np.ones([4,4])
I = np.identity(4)
D = E-I-(Wp)
D = D/np.sum(D,0)
#print(D)
tGM = Wp*alpha + D*(1-alpha)
#print(tGM)

# we can already create the D matrix for all dominant patterns
aE = np.reshape(E,[1,4,4])
aI = np.reshape(I,[1,4,4])
aD = aE-aI-Wps
aD = aD/np.reshape(np.sum(aD,1),[6,1,4])

'''
## Make a simulated sequence of observations
Now, we create a random sequence of 500 transitions as sampled from the true generative model
'''

num_t = 500
in_st = rnd.randint(4)
seq = []
seq_d = []
for it in range(0,num_t):
  p_vec  = tGM[in_st]
  p_cum  = np.cumsum(p_vec)
  rnd_dr = rnd.random_sample()
  is_high = rnd_dr > np.hstack((0, p_cum[0:-1]))
  is_low  = rnd_dr < np.hstack((p_cum[0:-1],1))
  out_st   = is_high&is_low
  out_st_d =  np.argwhere(out_st)
  seq = np.append(seq,out_st)
  seq_d = np.append(seq_d,out_st_d)
  in_st = out_st_d
seq = np.reshape(seq,[num_t,4])

'''
Now we want to obtain the likelihood of each transition under each of the possible models within the generative family.
We define a transition as the combination of a pair of consecutive observations indexed by trial $ tr_t = \{ s_{t-1},s_{t} \}$.
We want to obtain the likelihood of an observed transition given a model as parameterised by a permutation matrix and a dominant transition probability value, $\alpha$:

$P(tr_t|Wp_x,\alpha)$
'''

# define transitions as pairs of consecutive observations
x_in  = seq[:-1,:] # from first to second-to-last
x_out = seq[1:,:]  # from second to last

# take care of dimensions to allow for element-wise multiplication
x_in_5d = np.reshape(x_in.T,[1,4,1,(num_t-1),1])
x_out_5d = np.reshape(x_out.T,[1,1,4,(num_t-1),1])

#W = x_in_5d*x_out_5d
n_alph = 101
alphas = np.linspace((1/3),1,n_alph)
alphas_5d = np.reshape(alphas,[1,1,1,1,n_alph])

Wps_5d = np.reshape(Wps,[6,4,4,1,1])
aD_5d  = np.reshape(aD,[6,4,4,1,1])

like_dom_tr = Wps_5d*x_in_5d*x_out_5d*alphas_5d
like_ndom_tr = aD_5d*x_in_5d*x_out_5d*(1-alphas_5d)
like_tr = like_dom_tr+like_ndom_tr

'''
Compute the likelihoods of transitions under models
'''
like_tr_m = np.nansum(like_tr,1);
like_tr_m = np.nansum(like_tr_m,1);

loglike_tr = np.log(like_tr_m)
sum_ll_2d  = np.nansum(loglike_tr,1)
#sum_ll_2d.shape
#print(sum_ll_2d)


plt.imshow(sum_ll_2d, extent=[0,100,0,6], aspect='auto',cmap='viridis')
plt.title('sum of log likelihood across all trials')
plt.xlabel('alphas')
plt.ylabel('model index')
plt.show()

