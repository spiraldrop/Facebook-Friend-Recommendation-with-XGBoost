# Facebook-Friend-Recommendation-with-XGBoost

## About this project
This assignment uses singular value decomposition (SVD) with XGBoost to recommend Facebook friends. Most of the feature engineering has been included in the FB_featurization file, but we add two additional features and run the model to examine whether the new features add any substantial weight, which can be seen in the FB_all_features file.  In the FB_all_features file, we supplement features based on Preferential attachment[http://www.cs.cornell.edu/home/kleinber/link-pred.pdf] for followers and followees as well as [Supervised link features](https://ieeexplore.ieee.org/abstract/document/6033365) called ```svd_dot```. We then determine the best parameters to run our XGBoost model including learning rate and maximum tree depth using ```RandomizedSearchCV```. As a performance metric, we use the confusion matrix and later on find the feature importances. 

## Link to the data 
https://drive.google.com/drive/folders/1c50Q5RcmdpMYj1jCPc3ShOE2y4G8G2ez

## Libraries needed
```import pandas as pd
import numpy as np
import xgboost as xgb
import networkx as nx
from pandas import HDFStore
,DataFrame
from pandas import read_hdf
import os
import warnings
warnings
.filterwarnings
("ignore"
)
import csv
import pandas as pd#pandas to create small dataframes 
import datetime #Convert to unix time
import time #Convert to unix time
# if numpy is not installed already : pip3 install numpy
import numpy as np#Do aritmetic operations on arrays
# matplotlib: used to plot graphs
import matplotlib
import matplotlib.pylab as plt
import seaborn as sns#Plots
from matplotlib import rcParams#Size of plots 
from sklearn.cluster import MiniBatchKMeans
, KMeans#Clustering
import math
import pickle
import os
from sklearn.metrics import roc_auc_score
, f1_score
# to install xgboost: pip3 install xgboost
import xgboost as xgb
import warnings
import networkx as nx
import pdb
import pickle
from pandas import HDFStore
,DataFrame
from pandas import read_hdf
from scipy.sparse.linalg import svds, eigs
import gc
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from xgboost import XGBClassifier
```

### Link to the course
https://www.appliedaicourse.com/course/11/Applied-Machine-learning-course 
