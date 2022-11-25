# -*- coding:utf-8 -*-
import math
import os
import pickle
import time
import warnings

import matplotlib.pyplot as plot
import numpy as np
from scipy import signal
from scipy.io import loadmat
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

fs = 200
# Choose bands
""" 
band_dict = {
    "delta": [1, 4],
    "theta": [4, 8],
    "alpha": [8, 14],
    "beta": [14, 31],
    "gamma": [31, 50],
} 
"""
band_dict = {
    "delta": [1, 4],
    "theta": [4, 8],
    "alpha": [8, 14],
    "beta": [14, 31],
    "gamma": [31, 50],
}

emotion_lable_dict = {
    -1: "negative",
    0: "neutral",
    1: "positive"
}

# F7, F8, T7, T8 in SEED data set
channels = [5, 13, 23, 31]


class EmotionClassifier:
    """This is a class for emotion recognition classifier

    This class contains the reading of the dataset seed,
    the feature extraction of the data, the training and testing of the several model,
    and finding the optimal parameters of the several model

    Attributes:
        data_dir: The path of SEED dataset directory.
        feature_data_dir: The path of feature data directory.
    """

    datasets_X, datasets_y = [], []
    X_train, X_test, y_train, y_test = [], [], [], []
    y_pred = []
    SVM_params = {"C": 0.1, "kernel": "linear"}
    AdaBoost_params = {"n_estimators": 2000, "learning_rate": 0.01}
    flag = False
    usr_data_path = []

    MLP_params = {
        "activation": "tanh",
        "alpha": 0.05,
        "hidden_layer_sizes": (500, 3),
        "learning_rate": "adaptive",
        "max_iter": 1400,
        "solver": "sgd",
    }

    def __init__(
        self,  flag=False, data_dir="../SEED/Preprocessed_EEG/", feature_data_dir="../TrainingData/", usr_data_path="../TestData/BrainFlow-RAW.csv"
    ):
        """Inits EmotionClassifier Class

        Args:
            data_dir (str): The path of SEED dataset directory.
            feature_data_dir (str): The path of featured data directory.
        """
        self.usr_data_path = usr_data_path
        if not EmotionClassifier.__data_exist(feature_data_dir):
            print("/*********//* Feature data does not exit *//**********/")
            print("/****************//* creating data *//****************/")
            self.feature_extraction(data_dir, feature_data_dir)
        else:
            print("/*************//* Feature data exist *//**************/")
            print("/****************//* reading data *//*****************/")
            self.__feature_data_io(feature_data_dir, "rb")
        self.flag = flag

    def feature_extraction(self, data_dir, feature_data_dir):
        """Read the data, perform bandpass filtering on the data,
        and calculate the DE of the data

        Args:
            data_dir (str): The path of SEED dataset directory.
            feature_data_dir (str): The path of featured data directory.
        """
        label_Y = loadmat(data_dir + "label.mat")["label"][0]
        file_count = 0
        for file_name in os.listdir(data_dir):
            if file_name in ["label.mat", "readme.txt"]:
                continue
            file_data = loadmat(data_dir + file_name)
            file_count += 1
            print(
                "Currently processed to: {}，total progress: {}/{}".format(
                    file_name, file_count, len(os.listdir(data_dir)) - 2
                )
            )
            label_data = list(file_data.keys())[3:]
            for index, lable in enumerate(label_data):
                data = file_data[lable][channels]
                self.datasets_X.append(self.process_data(data, fs, channels))
                self.datasets_y.append(label_Y[index])

        self.datasets_X = np.array(self.datasets_X)
        self.datasets_y = self.datasets_y
        self.__feature_data_io(feature_data_dir, "wb")

    def process_data(self, data, fs, channels):
        dataset_X = []
        for band in band_dict.values():
            b, a = signal.butter(4, [band[0] / fs, band[1] / fs], "bandpass")
            filtedData = signal.filtfilt(b, a, data)
            filtedData_de = []
            for channel in range(len(channels)):
                filtedData_split = []
                for de_index in range(0, filtedData.shape[1] - fs, fs):
                    # Calculate DE
                    filtedData_split.append(
                        math.log(
                            2
                            * math.pi
                            * math.e
                            * np.var(
                                filtedData[channel, de_index: de_index + fs],
                                ddof=1,
                            )
                        )
                        / 2
                    )
                filtedData_split = filtedData_split[-100:]
                filtedData_de.append(filtedData_split)
            filtedData_de = np.array(filtedData_de)
            dataset_X.append(filtedData_de)
        dataset_X = np.array(dataset_X).reshape(
            (len(channels) * 100 * len(band_dict.keys()))
        )  # channels_num * 100 * band_num
        # self.print_wave(dataset_X)
        return dataset_X

    def __feature_data_io(self, feature_data_dir, method):
        """IO functions to read or write feature data

        Args:
            feature_data_dir (str): The path of featured data directory.
            method (str): read -- "rb" or write -- "wb"
        """
        with open(feature_data_dir + "datasets_X.pickle", method) as fx:
            if method == "rb":
                self.datasets_X = pickle.load(fx)
            else:
                pickle.dump(self.datasets_X, fx)
        with open(feature_data_dir + "datasets_y.pickle", method) as fy:
            if method == "rb":
                self.datasets_y = pickle.load(fy)
            else:
                pickle.dump(self.datasets_y, fy)

    def __data_exist(path):
        """Determine if the folder where the path is located exists or is empty

        Note:
            If the folder does not exist, create the folder.
            Return false if the folder does not exist or is empty
            Returns true if the folder exists and is not empty
        Args:
            path (str): The path of giving directory.
        Returns: Boolean
        """
        isExists = os.path.exists(path)
        if not isExists:
            os.makedirs(path)
            return False
        else:
            if os.path.getsize(path) < 100:
                return False
            return True

    def Init_train_test_data(self, test_size=0.2):
        """Initialize training data and test data

        Args:
            test_size : float or int, default=0.2
            If float, should be between 0.0 and 1.0 and represent the proportion
            of the dataset to include in the test split. If int, represents the
            absolute number of test samples.
        """
        print("/*********//* Initializing training data *//**********/")
        if self.flag:
            self.X_train = self.y_train = self.datasets_X
            self.X_test = self.y_test = self.datasets_y
        else:
            self.X_train, self.y_train, self.X_test, self.y_test = train_test_split(
                self.datasets_X, self.datasets_y, test_size=test_size
            )

    def SVM_model(self, find_params=False):
        """Set and train SVM model, and output the summary.

        Args:
            find_params : boolean, default=False
            If true, do find best parameters for SVM model
        """
        if find_params:
            self.model_find_best_params("SVM")
        self.model_train("SVM")
        if not self.flag:
            self.model_summary("SVM")

    def AdaBoost_model(self, find_params=False):
        """Set and train AdaBoost model, and output the summary.

        Args:
            find_params : boolean, default=False
            If true, do find best parameters for AdaBoost model
        """
        if find_params:
            self.model_find_best_params("Ada")
        self.model_train("Ada")
        if not self.flag:
            self.model_summary("Ada")

    def MLP_model(self, find_params=False):
        """Set and train MLP model, and output the summary.

        Args:
            find_params : boolean, default=False
            If true, do find best parameters for MLP model
        """
        if find_params:
            self.model_find_best_params("MLP")
        self.model_train("MLP")
        if not self.flag:
            self.model_summary("MLP")

    def model_find_best_params(self, model):
        """Find the best parameters for an SVM model

        Note:
            Class variable set default params
        """
        print("/**********//* Finding best model params *//**********/")
        SVM_gsearch_params = {
            "C": np.arange(0.1, 1, 0.1),
            "kernel": ["linear", "poly", "rbf", "sigmoid"],
        }
        AdaBoost_gsearch_params = {
            "n_estimators": [2000, 2500],
            "learning_rate": [0.01, 0.05, 0.1],
        }
        MLP_gsearch_params = {
            "max_iter": [1200, 1400, 1600],
            "alpha": [0.0001, 0.05, 0.1],
            "learning_rate": ["constant", "adaptive"],
            "activation": ["tanh", "relu"],
            "solver": ["sgd", "adam"],
            "hidden_layer_sizes": [(500, 1), (500, 2), (500, 3)],
        }

        match model:
            case "SVM":
                estimator = SVC()
                params = SVM_gsearch_params
            case "Ada":
                estimator = AdaBoostClassifier()
                params = AdaBoost_gsearch_params
            case "MLP":
                estimator = MLPClassifier()
                params = MLP_gsearch_params

        gsearch = GridSearchCV(
            estimator,
            params,
            scoring="accuracy",
            cv=5,
            n_jobs=-1,
            verbose=10,
        )
        gsearch.fit(self.X_train, self.y_train)
        match model:
            case "SVM":
                self.SVM_params = gsearch.best_params_
            case "Ada":
                self.AdaBoost_params = gsearch.best_params_
            case "MLP":
                self.MLP_params = gsearch.best_params_
        print("Best " + model + " Params is: " + str(gsearch.best_params_))
        print(model + " Score: %.3f" % gsearch.best_score_)

    def model_train(self, model):
        """Train the model

        Note:
            If the optimal parameters of the model are not calculated,
            the default params are used.
        """
        print("/*************//* Trainning " +
              model + " model *//*************/")
        start = time.time()
        match model:
            case "SVM":
                classifier = SVC(**self.SVM_params)
            case "Ada":
                classifier = AdaBoostClassifier(**self.AdaBoost_params)
            case "MLP":
                classifier = MLPClassifier(**self.MLP_params)
        classifier.fit(self.X_train, self.X_test)
        if self.flag:
            data_test = self.read_openbci_data(self.usr_data_path)
            self.y_pred = classifier.predict(data_test)
            print("The Emotion predicted by your dataset is: " +
                  emotion_lable_dict[self.y_pred[0]]
                  )
        else:
            self.y_pred = classifier.predict(self.y_train)
        end = time.time()
        print("Training time: ", end-start, "s")

    def model_summary(self, model):
        """Summary of the output model

        Note:
            Contains confusion matrix and classification report.
        """
        print("                      " + model + " Model:")
        print(confusion_matrix(self.y_test, self.y_pred))
        print(
            classification_report(
                self.y_test,
                self.y_pred,
                target_names=emotion_lable_dict.values(),
            )
        )

    def read_openbci_data(self, path):
        data = np.genfromtxt(path, delimiter=",",
                             usecols=(0, 1, 2, 3), skip_footer=1).T

        return [self.process_data(data, fs, channels)]

    def get_predicted_value(self):
        return self.y_pred[0]

    def print_wave(self, data):
        plot.plot(range(len(channels)*100*len(band_dict.keys())), data)
        plot.title('EEG')
        plot.grid(True, which='both')
        plot.axhline(y=0, color='k')
        plot.show()


if __name__ == "__main__":
    EC = EmotionClassifier(True)
    # Debug
    # EC = EmotionClassifier(True, data_dir="./SEED/Preprocessed_EEG/", feature_data_dir="./TrainingData/", usr_data_path="./TestData/positive.csv")
    EC.Init_train_test_data()

    EC.SVM_model()
    EC.AdaBoost_model()
    EC.MLP_model()
