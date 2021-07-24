import numpy as np
import pandas as pd
import yaml
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import roc_auc_score, recall_score, accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

from .utils import resample, buil_model, train_model, getFPR, getBalance, recall, precision
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression


class Module:
    def __init__(self, sampling_technique, m_l_algorithm,
                 data, labels, ratio):
        self.sampling_technique = sampling_technique
        self.m_l_algorithm = m_l_algorithm
        if type(data) is not pd.DataFrame:
            raise TypeError("data 必须是Dataframe类型")
        self.data = data
        if type(labels) is not np.ndarray:
            raise TypeError("labels 数据类型错误")
        if len(labels) != len(data):
            raise ValueError("data 和 label 长度不相等")
        self.labels = labels
        if type(ratio) is not np.ndarray:
            raise TypeError("ratio 数据类型错误")
        self.ratio = ratio
        self.kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)

    def func_in(self, func):
        def wrapper(*args, **kwargs):
            results = func(*args, **kwargs)
            return results
        return wrapper

    @func_in
    def DL(self, project_name, rate):
        params = self.__read_config()
        metrics_fn = [precision, recall]
        index = 1
        aucs = []
        tprs = []
        fprs = []
        balance = []
        accs = []
        for train, test in self.kfold.split(self.data, self.labels):
            train_data = self.data.iloc[train, :]
            y_train = self.labels[train]
            test_data = self.data.iloc[test, :]
            y_test = self.labels[test]
            data_sampling,label_sampling = self.get_sampling(train_data, y_train, rate)
            scaler = MinMaxScaler()
            data_sampling = scaler.fit_transform(data_sampling)
            test_data = scaler.transform(test_data)
            save_filepath = "./model/{}-DL/{}/{}_({})_{}.h5".format(self.sampling_technique,
                                                                    project_name,
                                                                    project_name,
                                                                    rate,
                                                                    index)
            model = buil_model(input_size=self.data.shape[0], lr=params["lr"],
                               d_rate=params["d_rate"], loss_fn=params["loss_fn"],
                               metrics_fn=metrics_fn)
            train_model(model, save_filepath=save_filepath, train_data=data_sampling,
                        train_label=label_sampling, epochs=params["epoch"])
            model = load_model(filepath=save_filepath,
                               custom_objects={"precision": precision, "recall": recall}, )
            predict_prob = model.predict(test_data)
            predict = [1 if v > 0.5 else 0 for v in predict_prob]
            aucs.append(roc_auc_score(y_true=y_test, y_score=predict_prob))
            tprs.append(recall_score(y_true=y_test, y_pred=predict))
            fprs.append(getFPR(y_true=y_test, predict=predict))
            balance.append(getBalance(y_true=y_test, y_score=predict_prob))
            accs.append(accuracy_score(y_true=y_test, y_pred=predict))
            index += 1
        return (aucs, tprs, fprs, balance, accs)
    @ func_in
    def module(self,rate):
        aucs = []
        tprs = []
        fprs = []
        balance = []
        accs = []
        for train, test in self.kfold.split(self.data, self.labels):
            train_data = self.data.iloc[train, :]
            y_train = self.labels[train]
            test_data = self.data.iloc[test, :]
            y_test = self.labels[test]
            data_sampling,label_sampling = self.get_sampling(train_data, y_train, rate)
            if self.m_l_algorithm == "NB":
                model = GaussianNB()
                model.fit(data_sampling, label_sampling)

            elif self.m_l_algorithm == "LR":
                model = LogisticRegression()
                model.fit(data_sampling, label_sampling)

            elif self.m_l_algorithm == "DT":
                model = DecisionTreeClassifier()
                model.fit(data_sampling, label_sampling)
            else:
                scaler = MinMaxScaler()
                data_sampling = scaler.fit_transform(data_sampling)
                test_data = scaler.transform(test_data)
                model = SVC(kernel="linear", max_iter=5000, probability=True)
                model.fit(data_sampling, label_sampling)
            predict = model.predict(test_data)
            predict_prob = model.predict_proba(test_data)[:, 1]
            aucs.append(roc_auc_score(y_true=y_test, y_score=predict_prob))
            tprs.append(recall_score(y_true=y_test, y_pred=predict))
            fprs.append(getFPR(y_true=y_test, predict=predict))
            balance.append(getBalance(y_true=y_test, y_score=predict_prob))
            accs.append(accuracy_score(y_true=y_test, y_pred=predict))
        return (aucs, tprs, fprs, balance, accs)

    def get_sampling(self, train_data, train_y, rate):

        if self.sampling_technique == "SM":
            number_zero = len(np.where(train_y == 0)[0])
            smote = SMOTE(sampling_strategy={0: number_zero, 1: int(number_zero * rate)}, random_state=111)
            data_sampling, label_sampling = smote.fit_resample(train_data, train_y)
        elif self.sampling_technique == "SP":
            number_one = len(np.where(train_y == 1)[0])
            subSampler = RandomUnderSampler(sampling_strategy={0: int(number_one / rate), 1: number_one},
                                            random_state=111, replacement=False)
            data_sampling, label_sampling = subSampler.fit_resample(train_data, train_y)
        else:
            data_sampling, label_sampling = resample(train_data, train_y, rate)
        return (data_sampling, label_sampling)


    def __read_config(self):
        conf = open(file="./config.yaml", mode='r', encoding='utf-8')
        str_conf = conf.read()
        dict_conf = yaml.load(stream=str_conf, Loader=yaml.FullLoader)
        return dict_conf
