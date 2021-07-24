import pandas as pd
from .module import Module
import numpy as np

filenameList = ['./data-file/KC1.csv',
               './data-file/PC3.csv',
               './data-file/PC4.csv',
               './data-file/JM1.csv',
               './data-file/PC5.csv']

projects = ["KC1","PC3","PC4","JM1","PC5"]
sampling_methods = ["SP","RE","SM"]
indicators = ["AUC","TPR","FPR","Balance","Accuracy"]
machine_learnings = ['J48','NB','LR',"SVM",'DL']
ratio = np.arange(0.5,1.01,0.05)

for i in range(len(filenameList)):
    data = pd.read_csv(filenameList[i],index_col=None)
    label = data.Defective.values
    label = np.array([0 if v == "N" else 1 for v in label])
    values = data.iloc[:, :-1]
    save_filename = "./开源结果/result-{}.xlsx".format(projects[i])
    writer = pd.ExcelWriter(save_filename)
    for sampling_method in sampling_methods:
        for machine_learning in machine_learnings:
            aucs_all = []
            tprs_all = []
            fprs_all = []
            balance_all = []
            accs_all = []
            for rate in ratio:
                modul = Module(sampling_method,machine_learning,values,label,rate)
                if machine_learning == "DL":
                    results = modul.func_in(modul.DL)(projects[i],rate)
                else:
                    results = modul.func_in(modul.module)(rate)
                aucs_all.append(results[0])
                tprs_all.append(results[1])
                fprs_all.append(results[2])
                balance_all.append(results[3])
                accs_all.append(results[4])
            all_ = np.concatenate([np.average(np.array(aucs_all), axis=1).reshape(-1, 1),
                                     np.average(np.array(tprs_all), axis=1).reshape(-1, 1),
                                     np.average(np.array(fprs_all), axis=1).reshape(-1, 1),
                                     np.average(np.array(balance_all), axis=1).reshape(-1, 1),
                                     np.average(np.array(accs_all), axis=1).reshape(-1, 1)],
                                  axis=1)
            df_tmp = pd.DataFrame(data=all_, columns=indicators, index=ratio)
            df_tmp.to_excel(writer, sheet_name=sampling_method + "-" + machine_learning)
    writer.save()
    writer.close()