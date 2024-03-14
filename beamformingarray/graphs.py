import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns




# dataset=""


















dataset = -1*np.array([[-6.010688123633101, -6.11100333450334, -6.243794597559917, -6.353587127613405, -6.42388933677584, -4.772412067522596, -6.407184307769425, -6.4197908857123596764438, -6.060722543924218, -6.388699582383895, -6.449888774516154, -6.39120517283698, -6.400202567615981, -6.177104147405201, -6.465263906008111],[-9.424527217236744, -9.175488239212678, -9.585531091313896, -9.563093541694169, -10.034977888651872, -8.939748574691485, -9.958136866617044, -9.935987032310575561875, -9.159240668691535, -9.882121265645655, -10.065974920728818, -9.92973595227665, -9.937704765406801, -9.347057359907996, -10.107159730583],[-9.472347645540951, -9.182897603550005, -9.56246931775837, -9.927244338302106, -10.01391621507058, -9.281706368995897, -9.904399923479042, -9.956931763844663601321, -9.220392421635584, -9.885867758345352, -10.079208487241658, -9.900098654207058, -9.934008927090357, -9.375825967889984, -10.1056884568959],[-10.35010925291487, -10.07672392572045, -9.734493123114946, -9.918234538988731, -10.012514344031922, -9.618868371780815, -9.93405400548719, -9.936609085916738, -10.110216996722775, -9.10208424279117, -9.570521558058292, -9.972075938648755,  -9.510208424279117, 
-9.916918399861396, -10.099641256653722]])
for i in range(4):
    print("fjadfhjakusdhfjkasdfjkasdf")
    for j in range(15):
        print(dataset[i][j])
# print(np.mean(dataset[3]))
# dataset=dataset.T

# # dataset = np.random.default_rng().uniform(60,95,(20,4))
# print(dataset)
# df = pd.DataFrame(dataset, columns=["4 Mic Linear",'8 Mic Linear','8 Mic Glasses',"8 Mic Glasses-Golomb"])
# df.head()


# vals, names, xs = [],[],[]
# for i, col in enumerate(df.columns):
#     vals.append(df[col].values)
#     names.append(col)
#     xs.append(np.random.normal(i + 1, 0.04, df[col].values.shape[0]))  # adds jitter to the data points - can be adjusted


# palette = ['r', 'g', 'y','b']
# for x, val, c in zip(xs, vals, palette):
#     plt.scatter(x, val, alpha=0.4, color=c)

# ##### Set style options here #####
# sns.set_style("darkgrid")  # "white","dark","darkgrid","ticks"
# boxprops = dict(linestyle='-', linewidth=1.5, color='#000000')
# flierprops = dict(marker='o', markersize=1,
#                   linestyle='none')
# whiskerprops = dict(color='#000000')
# capprops = dict(color='#000000')
# medianprops = dict(linewidth=1.5, linestyle='-', color='#33ce10')
# plt.boxplot(vals, labels=names, whis = 9999,medianprops=medianprops )

# # plt.boxplot(vals, labels=names, notch=False, boxprops=boxprops, whiskerprops=whiskerprops,autorange=True,capprops=capprops, flierprops=flierprops, medianprops=medianprops,showmeans=False,manage_ticks=True)

# # plt.axhline(y=0.1839131617283456, color='#ff3300', linestyle='--', linewidth=1, label='Final Design Mean STOI Improvement')

# plt.xlabel("Array Design", fontweight='normal', fontsize=14)
# plt.ylabel("Signal-to-Noise Ratio Improvement (Db)", fontweight='normal', fontsize=14)
# # sns.despine(bottom=True) # removes right and top axis lines
# # plt.axhline(y=19244, color='#ff3300', linestyle='--', linewidth=1, label='Actual Burnt Value (Acres)')


# plt.show()
































































# dataset = [[16564, 14804, 15178, 14149, 13807], [16478, 15026, 16129, 15916, 15911], [14408, 16185, 16119, 15163, 18791]]

# dataset = np.array([[14.78189365750168, 60.038212244241485, 66.69441820071805, 87.27373005128581, 42.02938599726986, 81.94140796885353, 90.48228172393821, 85.44561401632302, 55.079045775271275, 41.55363151833699, 31.76371291659244, 71.69444799330392, 29.11336041303635, 76.61049017254548, 32.666122301368375],[3.0484231422700803, 3.4859906779879357, 2.6646100256234146, 1.1449192346387764, 13.419620744967261, 3.975769013221525, 5.515917332890359, 2.57861892635015635344, 5.686457023389971, 1.4057201161171178, 3.8876120662095275, 3.6834296998545133, 0.023938824127305466, 5.522979391204906, 0.9000640348293049]])
# print(np.mean(dataset[1]))
# print(np.mean(dataset[0]))
# dataset=dataset.T

# # dataset = np.random.default_rng().uniform(60,95,(20,4))
# print(dataset)
# df = pd.DataFrame(dataset, columns=['GCC-PHAT','Novel Hybrid Source Tracker'])
# df.head()


# vals, names, xs = [],[],[]
# for i, col in enumerate(df.columns):
#     vals.append(df[col].values)
#     names.append(col)
#     xs.append(np.random.normal(i + 1, 0.04, df[col].values.shape[0]))  # adds jitter to the data points - can be adjusted

# plt.boxplot(vals, labels=names, whis = 9999)
# palette = ['g', 'r', 'y','b']
# for x, val, c in zip(xs, vals, palette):
#     plt.scatter(x, val, alpha=0.4, color=c)

# ##### Set style options here #####
# sns.set_style("darkgrid")  # "white","dark","darkgrid","ticks"
# boxprops = dict(linestyle='-', linewidth=1.5, color='#000000')
# flierprops = dict(marker='o', markersize=1,
#                   linestyle='none')
# whiskerprops = dict(color='#000000')
# capprops = dict(color='#000000')
# medianprops = dict(linewidth=1.5, linestyle='-', color='#33ce10')

# palette = ['r', 'g', 'b']
# plt.boxplot(vals, labels=names, notch=False, boxprops=boxprops, whiskerprops=whiskerprops,autorange=True,capprops=capprops, flierprops=flierprops, medianprops=medianprops,showmeans=False,manage_ticks=True)

# # plt.axhline(y=0.1839131617283456, color='#ff3300', linestyle='--', linewidth=1, label='Final Design Mean STOI Improvement')

# plt.xlabel("DOA Estimation Algorithm", fontweight='normal', fontsize=14)
# plt.ylabel("DOA Estimation Error (degrees)", fontweight='normal', fontsize=14)
# # sns.despine(bottom=True) # removes right and top axis lines
# # plt.axhline(y=19244, color='#ff3300', linestyle='--', linewidth=1, label='Actual Burnt Value (Acres)')
# plt.legend(bbox_to_anchor=(0.31, 1.06), loc=2, borderaxespad=0., framealpha=1, facecolor ='white', frameon=True)

# plt.show()











































data=np.array([0.051151484465168595, 0.10812471498541643, 0.049623166360339796, 0.046720724058639446, 0.09033882382738273, 0.03720496533322872, 0.08451455425794827, -0.16143960658514456, 0.06419549488288917, 0.18934319581056774, 0.2088677050021182, 0.07999164521169778, 0.13734770626277867, 0.0533159130399527, 0.10990266985276899])
print(data.mean())


dataset = np.array([[0.1250838896614127, 0.06953717470906642, 0.08269393118168197, 0.13383552722003372, 0.038502992817782404, 0.06468247162326086, 0.1576095168546658, 0.16823, 0.12560237880581035, 0.07918034681434749, 0.15510713568321724, 0.11267422753337109, 0.08436143117214275,  0.07264193388889663], [0.21071306985752428, 0.1345114974597848, 0.15312399392219447, 0.2015197514524586, 0.07846304086373017, 0.15675220394000303, 0.09906727637661691, 0.256052, 0.20347746815147572, 0.08977245779282389, 0.23739465774233293, 0.22256174079728763, 0.14956901556539817, 0.0467596697509413], [0.21356579960111793, 0.12448979950688865, 0.1523137886039675, 0.23306720693059196, 0.08021138742098738, 0.1556656992266448, 0.16955827102839421, 0.26014, 0.1965483787331831, 0.12247504685471, 0.2403815193052994, 0.22876370393442835, 0.1360003196953956,  0.07893384744975311],[0.205844608448928, 0.12340328141960943, 0.15080013435909048, 0.230071936824518, 0.07855683729146387, 0.13340669105200864, 0.16112545324509317, 0.257371,
0.20947956368223156, 0.12026694277506816, 0.24728307729804364, 0.22047130636213014, 0.1619817150077632, 0.26067276038923787
]])
print(np.mean(dataset[3]))
dataset=dataset.T

# dataset = np.random.default_rng().uniform(60,95,(20,4))
print(dataset)
df = pd.DataFrame(dataset, columns=['4 Mic Linear','8 Mic Linear',"8 Mic Glasses","8 Mic Glasses-Golomb"])
df.head()


vals, names, xs = [],[],[]
for i, col in enumerate(df.columns):
    vals.append(df[col].values)
    names.append(col)
    xs.append(np.random.normal(i + 1, 0.04, df[col].values.shape[0]))  # adds jitter to the data points - can be adjusted

plt.boxplot(vals, labels=names, whis = 9999)
palette = ['r', 'g', 'y','b']
for x, val, c in zip(xs, vals, palette):
    plt.scatter(x, val, alpha=0.4, color=c)

##### Set style options here #####
sns.set_style("white")  # "white","dark","darkgrid","ticks"
boxprops = dict(linestyle='-', linewidth=1.5, color='#000000')
flierprops = dict(marker='o', markersize=1,
                  linestyle='none')
whiskerprops = dict(color='#000000')
capprops = dict(color='#000000')
medianprops = dict(linewidth=1.5, linestyle='-', color='#33ce10')

palette = ['r', 'g', 'b']
plt.boxplot(vals, labels=names, notch=False, boxprops=boxprops, whiskerprops=whiskerprops,autorange=True,capprops=capprops, flierprops=flierprops, medianprops=medianprops,showmeans=False,manage_ticks=True)

plt.axhline(y=0.1839131617283456, color='#ff3300', linestyle='--', linewidth=1, label='Final Design Median STOI Improvement')

plt.xlabel("Array Design", fontweight='normal', fontsize=14)
plt.ylabel("STOI (Short Term Objective Intelligibility) Change", fontweight='normal', fontsize=14)
# sns.despine(bottom=True) # removes right and top axis lines
# plt.axhline(y=19244, color='#ff3300', linestyle='--', linewidth=1, label='Actual Burnt Value (Acres)')
plt.legend(bbox_to_anchor=(0.31, 1.06), loc=2, borderaxespad=0., framealpha=1, facecolor ='white', frameon=True)

plt.show()