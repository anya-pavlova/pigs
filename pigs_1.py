import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

%matplotlib inline

data_cam_df = pd.read_csv('xcms_pigs_camera.csv', sep = ',', index_col=0)

service_cols = ["mz", "mzmin", "mzmax", "rt", "rtmin", "rtmax", ".", "npeaks", "isotopes", "adduct", "pcgroup"]
assert len(set(service_cols) - set(data_cam_df.columns))==0
samples_cols = list(set(data_cam_df.columns) - set(service_cols))

data_cam_df[data_cam_df==0]=np.nan
data_cam_df[samples_cols] = np.log(data_cam_df[samples_cols])

#standards
#TAG
new_mz_palm = np.abs((data_cam_df['mz'] - 811.765)) / 811.765 
new_mz_palm = new_mz_palm * 1000000
sunf_ppm_palm = data_cam_df[new_mz_palm<13]

sunf_ppm_palm.head()
sunf_ppm_palm.index

RO=pd.Series([i.split('_')[3][1:] if 'S' in i else i for i in data_cam_df.columns],index=data_cam_df.columns)

for i in RO.index:
    try:
        RO[i]=int(RO[i])
    except:
        RO[i]=np.nan
RO=RO.dropna()


#LPS
plt.scatter(RO,data_cam_df.loc[1,RO.index], color = 'rebeccapurple')

#filtration by rt
#rt(sec) -> rt(min)

rt_in_min = data_cam_df['rt']/60
data_cam_df['rt'] = rt_in_min
data_cam_filt_rt_df = data_cam_df[(data_cam_df['rt'] > 0.6) & (data_cam_df['rt'] < 19)]
data_cam_filt_rt_df.head(4)

plt.gcf().set_size_inches(24,19)
plt.scatter(data_cam_filt_rt_df['rt'], data_cam_filt_rt_df['mz'], color = 'rebeccapurple', s = 50)

del_isotopes = data_cam_filt_rt_df['isotopes'].str.match(r'\[\d+\]\[M\+\d+\]\+').fillna(False)
#[m][M+n]+ где n от 1, m from 1
data_cam_filt_rt_iso_df = data_cam_filt_rt_df[~del_isotopes]

plt.gcf().set_size_inches(25,20)
plt.scatter(data_cam_filt_rt_iso_df['rt'], data_cam_filt_rt_iso_df['mz'], s = 50)

#PCA
lm1_samples = data_cam_filt_rt_iso_df[[col for col in samples_cols if 'LM1' in col]]
lm2_samples = data_cam_filt_rt_iso_df[[col for col in samples_cols if 'LM2'  in col]]
lm3_samples = data_cam_filt_rt_iso_df[[col for col in samples_cols if 'LM3' in col]]
bf1_samples = data_cam_filt_rt_iso_df[[col for col in samples_cols if 'BF1' in col]]
bf2_samples = data_cam_filt_rt_iso_df[[col for col in samples_cols if 'BF2' in col]]
bf3_samples = data_cam_filt_rt_iso_df[[col for col in samples_cols if 'BF3' in col]]
scat1_samples = data_cam_filt_rt_iso_df[[col for col in samples_cols if 'scat1' in col]]
scat2_samples = data_cam_filt_rt_iso_df[[col for col in samples_cols if 'scat2' in col]]

samples_filled_nan = data_cam_filt_rt_iso_df[samples_cols].fillna(0).transpose().as_matrix()

pca = PCA(2)

transformed_samples = pca.fit_transform(samples_filled_nan)

transformed_samples_df = pd.DataFrame(transformed_samples.T, columns = samples_cols)

plt.gcf().set_size_inches(15,10)
#plt.scatter(transformed_samples[:,0], transformed_samples[:,1], s = 200)
plt.scatter(transformed_samples_df[scat1_samples.columns].loc[0], transformed_samples_df[scat1_samples.columns].loc[1],
             s = 150, facecolors='none', edgecolors='#FF00FF',  label="scat1")
plt.scatter(transformed_samples_df[scat2_samples.columns].loc[0], transformed_samples_df[scat2_samples.columns].loc[1],
             s = 150, facecolors='none', edgecolors='black',  label="scat2")
plt.legend() 



plt.gcf().set_size_inches(15,10)
#plt.scatter(transformed_samples[:,0], transformed_samples[:,1], s = 200)
plt.scatter(transformed_samples_df[lm1_samples.columns].loc[0], transformed_samples_df[lm1_samples.columns].loc[1],
             s = 150, facecolors='none', edgecolors='#FF0000',  label="LM1")
plt.scatter(transformed_samples_df[lm2_samples.columns].loc[0], transformed_samples_df[lm2_samples.columns].loc[1],
             s = 150, facecolors='none', edgecolors='#FFFF00',  label="LM2")
plt.scatter(transformed_samples_df[lm3_samples.columns].loc[0], transformed_samples_df[lm3_samples.columns].loc[1],
             s = 150, facecolors='none', edgecolors='#00FF00',  label="LM3")
plt.scatter(transformed_samples_df[bf1_samples.columns].loc[0], transformed_samples_df[bf1_samples.columns].loc[1],
             s = 150, facecolors='none', edgecolors='#D6BCC0',  label="bf1")
plt.scatter(transformed_samples_df[bf2_samples.columns].loc[0], transformed_samples_df[bf2_samples.columns].loc[1],
             s = 150, facecolors='none', edgecolors='#00FFFF',  label="bf2")
plt.scatter(transformed_samples_df[bf3_samples.columns].loc[0], transformed_samples_df[bf3_samples.columns].loc[1],
             s = 150, facecolors='none', edgecolors='#0000FF',  label="bf3")
plt.legend() 


plt.gcf().set_size_inches(15,10)
#plt.scatter(transformed_samples[:,0], transformed_samples[:,1], s = 200)
plt.scatter(transformed_samples_df[lm1_samples.columns].loc[0], transformed_samples_df[lm1_samples.columns].loc[1],
             s = 150, facecolors='none', edgecolors='#FF0000',  label="LM1")
plt.scatter(transformed_samples_df[lm2_samples.columns].loc[0], transformed_samples_df[lm2_samples.columns].loc[1],
             s = 150, facecolors='none', edgecolors='#FFFF00',  label="LM2")
plt.scatter(transformed_samples_df[lm3_samples.columns].loc[0], transformed_samples_df[lm3_samples.columns].loc[1],
             s = 150, facecolors='none', edgecolors='#00FF00',  label="LM3")
plt.legend() 

plt.gcf().set_size_inches(20,15)
#plt.scatter(transformed_samples[:,0], transformed_samples[:,1], s = 200)
plt.scatter(transformed_samples_df[lm1_samples.columns].loc[0], transformed_samples_df[lm1_samples.columns].loc[1],
             s = 150, facecolors='none', edgecolors='#FF0000',  label="LM1")
plt.scatter(transformed_samples_df[lm2_samples.columns].loc[0], transformed_samples_df[lm2_samples.columns].loc[1],
             s = 150, facecolors='none', edgecolors='#FFFF00',  label="LM2")
plt.scatter(transformed_samples_df[lm3_samples.columns].loc[0], transformed_samples_df[lm3_samples.columns].loc[1],
             s = 150, facecolors='none', edgecolors='#00FF00',  label="LM3")
plt.scatter(transformed_samples_df[bf1_samples.columns].loc[0], transformed_samples_df[bf1_samples.columns].loc[1],
             s = 150, facecolors='none', edgecolors='#D6BCC0',  label="bf1")
plt.scatter(transformed_samples_df[bf2_samples.columns].loc[0], transformed_samples_df[bf2_samples.columns].loc[1],
             s = 150, facecolors='none', edgecolors='#00FFFF',  label="bf2")
plt.scatter(transformed_samples_df[bf3_samples.columns].loc[0], transformed_samples_df[bf3_samples.columns].loc[1],
             s = 150, facecolors='none', edgecolors='#0000FF',  label="bf3")
plt.scatter(transformed_samples_df[scat1_samples.columns].loc[0], transformed_samples_df[scat1_samples.columns].loc[1],
             s = 150, facecolors='none', edgecolors='#FF00FF',  label="scat1")
plt.scatter(transformed_samples_df[scat2_samples.columns].loc[0], transformed_samples_df[scat2_samples.columns].loc[1],
             s = 150, facecolors='none', edgecolors='black',  label="scat2")
plt.legend() 


def add_mz_rt_cols(df):
    return data_cam_filt_rt_iso_df[df.columns.tolist() + ['mz', 'rt']]

lm1_samples_with_mz_rt = add_mz_rt_cols(lm1_samples)

#after filtering

filtering_res = np.load('cleanedpeaks.npy')
filtering_res_indices = sorted(set(data_cam_filt_rt_iso_df.index) & set(filtering_res))
data_filtering = data_cam_filt_rt_iso_df.loc[filtering_res_indices]
type(data_filtering)

plt.gcf().set_size_inches(25,20)
plt.scatter(data_filtering['rt'], data_filtering['mz'])

#annotation
annot_data = pd.read_csv('xcms_pigs_camera.csv.ann.txt', sep = ',', index_col=0)
#loading data
lmfa_index = annot_data['lm_id'].str.contains('LMFA0103').fillna(False)
#строки в annot_data которые содержать LMFA0101 + что-то
lmfa_annot_data = annot_data[lmfa_index]
lmfa_annot_data = annot_data[lmfa_index]
#беру из annot_data строки которые соответствуют строкам в которых есть LMFA0101 + что-то

annot_data_no_null = annot_data[~annot_data['lm_id'].isnull()]

annot_groups = []

for index, data in annot_data_no_null.groupby(annot_data_no_null.index):
    annot_groups.append({
        "index": index,
        "lm_id": ";".join(data.lm_id),
        "adduct_annot": ";".join(data.adduct)
    })
    
annot_groups = pd.DataFrame(annot_groups)
annot_groups.set_index('index', inplace=True)

data_cam_filt_rt_iso_df_annot = pd.merge(data_cam_filt_rt_iso_df, annot_groups, left_index=True, right_index=True)

plt.gcf().set_size_inches(25,20)
plt.scatter(data_cam_filt_rt_iso_df_annot['rt'], data_cam_filt_rt_iso_df_annot['mz'])

lmfa_index3 = annot_data['lm_id'].str.contains('LMFA0103').fillna(False)
#строки в annot_data которые содержать LMFA0101 + что-то
lmfa_annot_data3 = annot_data[lmfa_index3]
lmfa_annot_data3 = annot_data[lmfa_index3]
#беру из annot_data строки которые соответствуют строкам в которых есть LMFA0101 + что-то
data_cam_filt_rt_iso_df_annot.rename(columns={"X171208_pigs_LM3_1_11_1.50_pos": "X171208_pigs_LM3_11_1.50_pos"}, inplace=True)

import re
part_pattern = re.compile(r"_pigs_(\S+?)_")

def get_pig_part(col_name):
    return part_pattern.search(col_name).group(1)
    
    
parts_mapping = {}
for col in data_cam_filt_rt_iso_df_annot.columns:
    if "QC" in col:
        continue
    
    try:
        parts_mapping.setdefault(get_pig_part(col), []).append(col) 
    except:
        pass
    
    
for part, columns in parts_mapping.items():

    f, axarr = plt.subplots(len(columns), sharex=True, figsize=(15,20))
    
    for ax, col in zip(axarr, columns):
        ax.scatter(data_cam_filt_rt_iso_df_annot.mz, data_cam_filt_rt_iso_df_annot[col])
        ax.set_title(col)
    
    f.suptitle(part)
    f.subplots_adjust(top=0.95)
    
    plt.show()
    
pigs_pattern = re.compile(r"_pigs_\S+?_(\S+?)_")


def get_pig_num(col_name):
    return int(pigs_pattern.search(col_name).group(1))
    
    
pigs_mapping = {}
for col in data_cam_filt_rt_iso_df_annot.columns:
    if "QC" in col or "scat1" in col:
        continue
    
    try:
        pigs_mapping.setdefault(get_pig_num(col), []).append(col) 
    except:
        pass

    
for pig, columns in sorted(pigs_mapping.items()):
    
    columns = sorted(columns)

    f, axarr = plt.subplots(len(columns), sharex=True, figsize=(15,20))
    
    for ax, col in zip(axarr, columns):
        ax.scatter(data_cam_filt_rt_iso_df_annot.mz, data_cam_filt_rt_iso_df_annot[col])
        ax.set_title(col)
    
    f.suptitle("Pig {0}".format(pig))
    f.subplots_adjust(top=0.95)
    
    plt.show()
    
parts_dfs = {part_name: data_cam_filt_rt_iso_df_annot[cols].T for part_name, cols in parts_mapping.items() }
parts_dfs = list(parts_dfs.items())

from rpy2.robjects import r, pandas2ri

pandas2ri.activate()
from rpy2.robjects.packages import importr

Hotelling = importr('Hotelling')

def hotelling_test(df1, df2):
    df1_r = pandas2ri.py2ri(df1)
    df2_r = pandas2ri.py2ri(df2)
    return Hotelling.hotelling_test(df1_r, df2_r)

Hotelling.hotelling_test()

for i in range(len(parts_dfs)):
    for j in range(i+1, len(parts_dfs)):
        part1, df1 = parts_dfs[i]
        part2, df2 = parts_dfs[j]
        
        test_result = hotelling_test(df1, df2)
        print(test_result)
        break
    break
    
    
    
samples = data_cam_filt_rt_iso_df_annot[[col for col in data_cam_filt_rt_iso_df_annot.columns if col.startswith("X") and "QC" not in col]]

del samples['X171206_pigs_scat1_3_15_pos_1.100']
del samples['X171208_pigs_scat1_2_15_1.100_pos']
del samples['X171208_pigs_scat1_1_15_1.100_pos']

import seaborn

plt.gcf().set_size_inches(30, 30)

seaborn.heatmap(samples.corr(), cmap="BuPu")

samples.corr()

def pigs_vectors(variables, dependents):
    X = []
    Y = []
    
    
    for pig, cols in pigs_mapping.items():
        if pig == 15:
            continue
        
        x = []
        y = []
        
        for var_name in variables:
            for col in cols:
                if var_name in col:
                    x.append(data_cam_filt_rt_iso_df_annot[col])
                             
        for dep_name in dependents:
            for col in cols:
                if dep_name in col:
                    y.append(data_cam_filt_rt_iso_df_annot[col])
                    
        
        X.append(np.concatenate(x))
        Y.append(np.concatenate(y))
        
    return np.array(X), np.array(Y)

X, Y = pigs_vectors(["BF1", "BF2", "BF3", "LM1", "LM2", "LM3"], ["scat2"])
    
    
    
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import LeaveOneOut

loo = LeaveOneOut()
pigs_mse = []

for train_index, test_index in loo.split(X):
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]
    
    linreg = Ridge(alpha=1e6)
    linreg.fit(X_train, Y_train)
    square_errors =np.abs(Y_test - linreg.predict(X_test))
    mse = np.mean(square_errors)
    pigs_mse.append(mse)
    print("MSE", mse)
    plt.plot(Y_test.ravel(), label='True', c='b')
    plt.plot(linreg.predict(X_test).ravel(), label='Prediction', alpha=0.5, c='r')
    plt.legend()
    plt.show()

print("Average MSE", np.mean(pigs_mse))
print("Std MSE", np.std(pigs_mse))

parts_mse = []

for part1, part2 in zip(*pigs_vectors(["BF1"], ["BF2"])):
    square_errors =np.abs(part1.ravel() - part2.ravel())
    mse = np.mean(square_errors)
    parts_mse.append(mse)
    
print("MSE", np.mean(parts_mse))

for bf, scat in zip(*pigs_vectors(["BF1"], ["scat2"])):
    plt.plot(bf.ravel(), label='BF1')
    plt.plot(scat.ravel(), label='scat2')
    plt.legend()
    plt.show()

data_cam_filt_rt_iso_df_annot.iloc[0]['lm_id']

#MDS

print(__doc__)
import numpy as np

from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection

from sklearn import manifold
from sklearn.metrics import euclidean_distances
from sklearn.decomposition import PCA

data_cam_filt_rt_iso_df_annot.drop(["X171208_pigs_scat1_1_15_1.100_pos", "X171208_pigs_scat1_2_15_1.100_pos"], axis = 1, inplace = True)
data_cam_filt_rt_iso_df_annot.rename(columns={"X171208_pigs_LM3_1_11_1.50_pos": "X171208_pigs_LM3_11_1.50_pos",
                                             "X171206_pigs_scat1_3_15_pos_1.100": "X171208_pigs_scat1_15_1.100_pos"}, inplace=True)

all_columns = data_cam_filt_rt_iso_df_annot.columns.tolist()
samples_columns = all_columns[all_columns.index('X171208_pigs_scat1_15_1.100_pos')
                              :all_columns.index('X171208_pigs_scat2_9_1.100_pos')+1]

import numpy as np
import pandas as pd
from sklearn import manifold
from sklearn.metrics import euclidean_distances

seed = np.random.RandomState(seed=3)
data = pd.read_csv('data/big-file.csv')

#  start small dont take all the data, 
#  its about 200k records
subset = data[:10000]
similarities = euclidean_distances(subset)

mds = manifold.MDS(n_components=2, max_iter=3000, eps=1e-9, 
      random_state=seed, dissimilarity="precomputed", n_jobs=1)

pos = mds.fit(similarities).embedding_
    
    
    
    
    
    
    
    
    
