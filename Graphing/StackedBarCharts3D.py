from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
ax = fig.add_subplot(111, projection = "3d")

ax.set_xlabel("Dataset")
ax.set_ylabel("Algorithm")
ax.set_zlabel("Evaluation Metric")
ax.set_xlim3d(0.5,6.5)
ax.set_ylim3d(0.5,6.5)

xNames = ['tf-idf', 'doc2vec (500)', 'doc2vec (1000)', 'hdp (bow)', 'hdp (tf-idf)', 'lda as lsi']
yNames = ['RF', 'MLP', 'KNN', 'log. reg. (Ovs.R)', 'KNN (Ovs.R)', 'SVM (Ovs.R)']

ticksx = np.arange(1, 7, 1)
plt.xticks(ticksx, xNames)

ticksy = np.arange(1, 7, 1)
plt.yticks(ticksy, yNames)

xpos = [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6]
ypos = [1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6]
zpos = np.zeros(36)

dx = np.array([0.3] * 36)
dy = np.array([0.3] * 36)
dz = [[ 20.559,20.559,20.561,20.536,20.557,20.554, 
        19.676,19.80,19.837,20.341,20.341,19.881,
        19.903,20.287,20.360,20.190,20.285,19.950,
        0.00,19.900,19.850,20.268,20.355,20.013,
        0.00,20.257,20.336,20.215,20.286,19.965,
        0.00,20.280,20.145,20.481,20.532,20.472], 
       [0.541,0.545,0.541,0.537,0.533,0.554,
        0.573,0.564,0.539,0.542,0.534,0.594, 
        0.586,0.548,0.529,0.511,0.506,0.571,
       0.000,0.585,0.579,0.542,0.538,0.592, 
       0.00,0.540,0.523,0.504,0.499,0.562,
       0.000,0.583,0.588,0.538,0.532,0.570], 
      [ 0.560,0.564,0.568,0.563,.573,0.551,
       0.458,0.487,0.512,0.533,0.539,0.460,
       0.470,0.532,0.553,0.544,0.552,0.479,
       0.000,0.468,0.468,0.523,0.544,0.471,
       0.000,0.531,0.552,0.550,0.559,0.485,
       0.000,0.505,0.487,0.550,0.568,0.535,]
     ]

_zpos = zpos
colors = ['b', 'g', 'y']
#colors = ['#5E35B1', '#B39DDB', '#D1C4E9']
for i in range(3):
    ax.bar3d(xpos, ypos, _zpos, dx, dy, dz[i], color=colors[i], alpha=0.6)
    _zpos += dz[i]

plt.gca().invert_xaxis()
plt.show()
