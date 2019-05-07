import pandas as pd
import numpy as np
import matplotlib.colors
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import LeaveOneOut
from collections import Counter
import seaborn
import math
from sklearn.manifold import MDS


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

rt_in_min = data_cam_df['rt']/60
data_cam_df['rt'] = rt_in_min
data_cam_filt_rt_df = data_cam_df[(data_cam_df['rt'] > 0.6) & (data_cam_df['rt'] < 19)]
data_cam_filt_rt_df.head(4)

del_isotopes = data_cam_filt_rt_df['isotopes'].str.match(r'\[\d+\]\[M\+\d+\]\+').fillna(False)
#[m][M+n]+ где n от 1, m from 1
data_cam_filt_rt_iso_df = data_cam_filt_rt_df[~del_isotopes]

#after filtering
filtering_res = np.load('cleanedpeaks.npy')
filtering_res_indices = sorted(set(data_cam_filt_rt_iso_df.index) & set(filtering_res))
data_filtering = data_cam_filt_rt_iso_df.loc[filtering_res_indices]

annot_data = pd.read_csv('xcms_pigs_camera.csv.ann.txt', sep = ',', index_col=0)
#loading data

lmfa_index = annot_data['lm_id'].str.contains('LMFA0101').fillna(False)
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
data_cam_filt_rt_iso_df_annot

data_cam_filt_rt_iso_df_annot.drop(["X171208_pigs_scat1_1_15_1.100_pos", "X171208_pigs_scat1_2_15_1.100_pos"], axis = 1, inplace = True)
data_cam_filt_rt_iso_df_annot.rename(columns={"X171208_pigs_LM3_1_11_1.50_pos": "X171208_pigs_LM3_11_1.50_pos",
                                             "X171206_pigs_scat1_3_15_pos_1.100": "X171208_pigs_scat1_15_1.100_pos"}, inplace=True)

#other

all_columns = data_cam_filt_rt_iso_df_annot.columns.tolist()
samples_columns = all_columns[all_columns.index('X171208_pigs_scat1_15_1.100_pos'):all_columns.index('X171208_pigs_scat2_9_1.100_pos')+1]

data_annot_norm = data_cam_filt_rt_iso_df_annot.copy()
data_annot_norm[samples_columns] =  data_annot_norm[samples_columns] /  data_annot_norm[samples_columns].max()

def select_columns(columns, parts):
    return [col for col in columns if any(part in col for part in parts)]

meat_columns = select_columns(samples_columns, ["BF1", "BF2", "BF3", "LM1", "LM2", "LM3"])
fat_columns = select_columns(samples_columns, ["scat1", "scat2"])

mean_meat_sample = data_annot_norm[meat_columns].mean(axis=1)
mean_fat_sample = data_annot_norm[fat_columns].mean(axis = 1)

def count_lipids(lipids_col):
    lipids = []
    for lipids_row in lipids_col:
        lipids.extend(lipids_row.split(";"))
    return Counter(lipids)

def get_top_lipids_for_sample(sample, n):
    top_indices = np.argsort(sample.values)[-n:]
    return count_lipids(data_annot_norm['lm_id'].iloc[top_indices])

def get_top_lipids_for_columns(columns, n):
    lipids_counter = Counter()
    for col in columns:
        lipids_counter.update(get_top_lipids_for_sample(data_annot_norm[col], n))
        
    lipids_counts = pd.Series(lipids_counter)
    lipids_counts /= len(columns)
    return lipids_counts.sort_values(ascending=False)
    
n_peaks = 50

top_meat_lipids = get_top_lipids_for_columns(meat_columns, n_peaks)
top_meat_lipids = list(zip(top_meat_lipids, top_meat_lipids.index))
top_meat_lipids

top_fat_lipids = get_top_lipids_for_columns(fat_columns, n_peaks)
top_fat_lipids = list(zip(top_fat_lipids, top_fat_lipids.index))
top_fat_lipids

logfc_fat_meat = np.log(mean_fat_sample/mean_meat_sample)


#cmap = seaborn.cubehelix_palette(as_cmap=True)
cmap = matplotlib.colors.ListedColormap(seaborn.color_palette("RdBu_r", 300).as_hex())
f, ax = plt.subplots()
points = ax.scatter(data_annot_norm['rt'], data_annot_norm['mz'], c=logfc_fat_meat, s=50, cmap=cmap)
f.colorbar(points)
f.set_size_inches(25,20)


count_lipids(data_annot_norm['lm_id'][logfc_fat_meat<=logfc_fat_meat.quantile(0.02)]).most_common()

mean_scat1_sample = data_annot_norm[select_columns(samples_columns, ["scat1"])].mean(axis=1)
mean_scat2_sample = data_annot_norm[select_columns(samples_columns, ["scat2"])].mean(axis=1)

logfc_scat1_scat2 = np.log(mean_scat1_sample/mean_scat2_sample)

cmap = seaborn.cubehelix_palette(as_cmap=True)
f, ax = plt.subplots()
points = ax.scatter(data_annot_norm['rt'], data_annot_norm['mz'], c=logfc_scat1_scat2, s=50, cmap=cmap)
f.colorbar(points)
f.set_size_inches(25,20)

data_annot_norm.to_csv('data_annot_pigs.csv')

#MDS

data_ms_no_nan = data_mds.fillna(0)
data_ms_no_nan.isnull().any().any()

data_ms_no_nan = data_ms_no_nan.iloc[:, 3:].corr()
data_ms_no_nan.shape

pos = MDS(dissimilarity='precomputed', max_iter=3000, random_state=13).fit_transform(1-data_ms_no_nan)
pos = pos.T
pos = pd.DataFrame(pos, columns = data_ms_no_nan.columns)

qc_cols = [col for col in pos.columns if 'QC' in col]
non_qc_cols = [col for col in pos.columns if 'QC' not in col]
scat1 = [col for col in pos.columns if 'scat1'in col]
scat2 = [col for col in pos.columns if 'scat2'in col]
LM1 = [col for col in pos.columns if 'LM1'in col]
LM1 = [col for col in pos.columns if 'LM1'in col]
LM2 = [col for col in pos.columns if 'LM2'in col]
LM3 = [col for col in pos.columns if 'LM3'in col]
BF1 = [col for col in pos.columns if 'BF1'in col]
BF2 = [col for col in pos.columns if 'BF2'in col]
BF3 = [col for col in pos.columns if 'BF3'in col]


plt.gcf().set_size_inches(10,5)
#plt.scatter(pos[non_qc_cols].iloc[0], pos[non_qc_cols].iloc[1], label='Non QC', s=100, facecolors='none', edgecolors='rebeccapurple')
plt.scatter(pos[LM1].iloc[0], pos[LM1].iloc[1], label='LM1', s=100, facecolors='none', edgecolors='Blue')
plt.scatter(pos[LM2].iloc[0], pos[LM2].iloc[1], label='LM2', s=100, facecolors='none', edgecolors='Blue')
plt.scatter(pos[LM3].iloc[0], pos[LM3].iloc[1], label='LM3', s=100, facecolors='none', edgecolors='Blue')
plt.scatter(pos[BF1].iloc[0], pos[BF1].iloc[1], label='BF1', s=100, facecolors='none', edgecolors='red')
plt.scatter(pos[BF2].iloc[0], pos[BF2].iloc[1], label='BF2', s=100, facecolors='none', edgecolors='red')
plt.scatter(pos[BF3].iloc[0], pos[BF3].iloc[1], label='BF3', s=100, facecolors='none', edgecolors='red')





#plt.scatter(pos[qc_cols].iloc[0], pos[qc_cols].iloc[1], label='QC', s=100, facecolors='none', edgecolors='black')
#plt.scatter(pos[scat1].iloc[0], pos[scat1].iloc[1], label='SCAT1', s=100, facecolors='none', edgecolors='red')
#plt.scatter(pos[scat2].iloc[0], pos[scat2].iloc[1], label='SCAT2', s=100, facecolors='none', edgecolors='Blue')

plt.legend()

qc_cols = [col for col in pos.columns if 'QC' in col]
non_qc_cols = [col for col in pos.columns if 'QC' not in col]
scat1 = [col for col in pos.columns if 'LM1'in col]
scat2 = [col for col in pos.columns if 'BF2'in col]


plt.gcf().set_size_inches(10,5)
#plt.scatter(pos[non_qc_cols].iloc[0], pos[non_qc_cols].iloc[1], label='Non QC', s=100, facecolors='none', edgecolors='rebeccapurple')
#plt.scatter(pos[qc_cols].iloc[0], pos[qc_cols].iloc[1], label='QC', s=100, facecolors='none', edgecolors='black')
plt.scatter(pos[scat1].iloc[0], pos[scat1].iloc[1], label='LM1', s=100, facecolors='none', edgecolors='red')
plt.scatter(pos[scat2].iloc[0], pos[scat2].iloc[1], label='BF2', s=100, facecolors='none', edgecolors='Blue',)
plt.title('LM1 and BF2')
plt.legend()

#FAT TAGs
data_annot_norm
data_tag_rt = data_annot_norm[(data_annot_norm['rt'] > 13) & (data_annot_norm['rt'] < 17)]
data_tag_rtmz = data_tag_rt[(data_tag_rt['mz'] > 700) & (data_tag_rt['rt'] < 1100)]

plt.gcf().set_size_inches(24,19)
plt.scatter(data_tag_rtmz['rt'], data_tag_rtmz['mz'], color = 'rebeccapurple', s = 50)

data_gk_tags = data_tag_rtmz[data_tag_rtmz.lm_id.str.contains("LMGL0301") & data_tag_rtmz.adduct_annot.str.contains("NH4")]


plt.gcf().set_size_inches(24,19)
plt.scatter(data_gk_tags['rt'], data_gk_tags['mz'], color = 'rebeccapurple', s = 50)


