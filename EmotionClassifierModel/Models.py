import math
import os
import pickle
import warnings

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
band_dict = {
    "delta": [1, 4],
    "theta": [4, 8],
    "alpha": [8, 14],
    "beta": [14, 31],
    "gamma": [31, 50],
}

# channels = [0, 2, 25, 29, 41, 49, 58, 60, 5, 13, 7, 11, 23, 31, 43, 47]
channels = [5, 13, 23, 31]


class EmotionClassifier:
    """This is a class for emotion recognition classifier

    This class contains the reading of the dataset seed,
    the feature extraction of the data, the training and testing of the SVM model,
    and finding the optimal parameters of the SVM model

    Attributes:
        data_dir: The path of SEED dataset directory.
        feature_data_dir: The path of feature data directory.
    """

    datasets_X, datasets_y = [], []
    X_train, X_test, y_train, y_test = [], [], [], []
    y_pred = []
    SVM_params = {"C": 0.1, "kernel": "linear"}
    AdaBoost_params = {"n_estimators": 500, "learning_rate": 1}

    MLP_params = {
        "activation": "tanh",
        "alpha": 0.05,
        "hidden_layer_sizes": (500, 3),
        "learning_rate": "adaptive",
        "max_iter": 1400,
        "solver": "sgd",
    }

    def __init__(
        self, data_dir="../SEED/Preprocessed_EEG/", feature_data_dir="../TrainingData/"
    ):
        """Inits SVMEmotionRecognition Class

        Args:
            data_dir (str): The path of SEED dataset directory.
            feature_data_dir (str): The path of feature data directory.
        """
        if not self.__data_exist(feature_data_dir):
            print("/*********//* Feature data does not exit *//**********/")
            print("/****************//* creating data *//****************/")
            self.feature_extraction(data_dir, feature_data_dir)
        else:
            print("/*************//* Feature data exist *//**************/")
            print("/****************//* reading data *//*****************/")
            self.__feature_data_io(feature_data_dir, "rb")

    def feature_extraction(self, data_dir, feature_data_dir):
        """Read the data, perform bandpass filtering on the data,
        and calculate the DE of the data

        Args:
            data_dir (str): The path of SEED dataset directory.
            feature_data_dir (str): The path of feature data directory.
        """
        label_Y = loadmat(data_dir + "label.mat")["label"][0]
        file_count = 0
        for file_name in os.listdir(data_dir):
            if file_name in ["label.mat", "readme.txt"]:
                continue
            file_data = loadmat(data_dir + file_name)
            file_count += 1
            print(
                "当前已处理到{}，总进度{}/{}".format(
                    file_name, file_count, len(os.listdir(data_dir)) - 2
                )
            )
            label_data = list(file_data.keys())[3:]
            for index, lable in enumerate(label_data):
                dataset_X = []
                data = file_data[lable][channels]
                for band in band_dict.values():
                    b, a = signal.butter(4, [band[0] / fs, band[1] / fs], "bandpass")
                    filtedData = signal.filtfilt(b, a, data)
                    filtedData_de = []
                    for channel in range(len(channels)):
                        filtedData_split = []
                        for de_index in range(0, filtedData.shape[1] - fs, fs):
                            # 计算 DE
                            filtedData_split.append(
                                math.log(
                                    2
                                    * math.pi
                                    * math.e
                                    * np.var(
                                        filtedData[channel, de_index : de_index + fs],
                                        ddof=1,
                                    )
                                )
                                / 2
                            )
                        # 这里将每个样本大小进行统一，如果想通过滑动窗口截取样本可在这一行下面自行修改
                        filtedData_split = filtedData_split[-100:]
                        filtedData_de.append(filtedData_split)
                    filtedData_de = np.array(filtedData_de)
                    dataset_X.append(filtedData_de)
                dataset_X = np.array(dataset_X).reshape(
                    (len(channels) * 100 * 5)
                )  # channels_num * 100 *5
                self.datasets_X.append(dataset_X)
                self.datasets_y.append(label_Y[index])

        self.datasets_X = np.array(self.datasets_X)
        self.datasets_y = self.datasets_y
        self.__feature_data_io(feature_data_dir, "wb")

    def __feature_data_io(self, feature_data_dir, method):
        """IO functions to read or write feature data

        Args:
            feature_data_dir (str): The path of feature data directory.
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

    def __data_exist(self, path):
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
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.datasets_X, self.datasets_y, test_size=test_size
        )

    def SVM_model(self, find_params=False):
        """Set and train SVM model, and output the summary.

        Args:
            find_params : boolean, default=False
            If true, do find best parameters for SVM model
        """
        if find_params:
            self.SVM_model_find_best_params()
        self.SVM_model_train()
        self.SVM_model_summary()

    def AdaBoost_model(self, find_params=False):
        """Set and train AdaBoost model, and output the summary.

        Args:
            find_params : boolean, default=False
            If true, do find best parameters for AdaBoost model
        """
        if find_params:
            self.AdaBoost_model_find_best_params()
        self.AdaBoost_model_train()
        self.AdaBoost_model_summary()

    def MLP_model(self, find_params=False):
        """Set and train MLP model, and output the summary.

        Args:
            find_params : boolean, default=False
            If true, do find best parameters for MLP model
        """
        if find_params:
            self.MLP_model_find_best_params()
        self.MLP_model_train()
        self.MLP_model_summary()

    def SVM_model_find_best_params(self):
        """Find the best parameters for an SVM model

        Note:
            Class variable set default value C=0.1, kernel = "linear"
        """
        print("/**********//* Finding best model params *//**********/")
        params = {
            "C": np.arange(0.1, 1, 0.1),
            "kernel": ["linear", "poly", "rbf", "sigmoid"],
        }
        gsearch = GridSearchCV(
            SVC(kernel="linear"),
            params,
            scoring="accuracy",
            cv=5,
            n_jobs=-1,
            verbose=10,
        )
        gsearch.fit(self.X_train, self.y_train)
        self.SVM_params["C"] = gsearch.best_params_["C"]
        self.SVM_params["kernel"] = gsearch.best_params_["kernel"]
        print("Best SVM Params is: " + str(gsearch.best_params_))
        print("SVM Score: %.3f" % gsearch.best_score_)

    def SVM_model_train(self):
        """Train the SVM model

        Note:
            If the optimal parameters of the SVM model are not calculated,
            the default values SVM_params are used.
        """
        print("/*************//* Trainning SVM model *//*************/")
        svclassifier = SVC(**self.SVM_params)
        svclassifier.fit(self.X_train, self.y_train)
        self.y_pred = svclassifier.predict(self.X_test)

    def SVM_model_summary(self):
        """Summary of the output model

        Note:
            Contains confusion matrix and classification report.
        """
        print("                      SVM Model")
        print(confusion_matrix(self.y_test, self.y_pred))
        print(
            classification_report(
                self.y_test,
                self.y_pred,
                target_names=["negative", "neutral", "positive"],
            )
        )

    def AdaBoost_model_find_best_params(self):
        """Find the best parameters for an AdaBoost model

        Note:
            Class variable set default AdaBoost_params"
        """
        print("/**********//* Finding best model params *//**********/")
        params = {
            "n_estimators": [500, 600, 700],
            "learning_rate": [1.0],
        }
        gsearch = GridSearchCV(
            AdaBoostClassifier(),
            params,
            scoring="accuracy",
            cv=5,
            n_jobs=-1,
            verbose=10,
        )
        gsearch.fit(self.X_train, self.y_train)
        self.AdaBoost_params["n_estimators"] = gsearch.best_params_["n_estimators"]
        self.AdaBoost_params["learning_rate"] = gsearch.best_params_["learning_rate"]
        print("Best AdaBoost Params is: " + str(gsearch.best_params_))
        print("AdaBoost Score: %.3f" % gsearch.best_score_)

    def AdaBoost_model_train(self):
        """Train the AdaBoost model

        Note:
            If the optimal parameters of the AdaBoost model are not calculated,
            the default values AdaBoost_params are used.
        """
        print("/*************//* Trainning Ada model *//*************/")
        abc = AdaBoostClassifier(**self.AdaBoost_params)
        abc.fit(self.X_train, self.y_train)
        self.y_pred = abc.predict(self.X_test)

    def AdaBoost_model_summary(self):
        """Summary of the output model

        Note:
            Contains confusion matrix and classification report.
        """
        print("                      AdaBoost Model:")
        print(confusion_matrix(self.y_test, self.y_pred))
        print(
            classification_report(
                self.y_test,
                self.y_pred,
                target_names=["negative", "neutral", "positive"],
            )
        )

    def MLP_model_find_best_params(self):
        """Find the best parameters for an MLP model

        Note:
            Class variable set default MLP_params"
        """
        print("/**********//* Finding best model params *//**********/")
        params = {
            "max_iter": [1200, 1400, 1600],
            "alpha": [0.0001, 0.05, 0.1],
            "learning_rate": ["constant", "adaptive"],
            "activation": ["tanh", "relu"],
            "solver": ["sgd", "adam"],
            "hidden_layer_sizes": [(500, 1), (500, 2), (500, 3)],
        }
        gsearch = GridSearchCV(
            MLPClassifier(), params, scoring="accuracy", cv=5, n_jobs=-1, verbose=10
        )
        gsearch.fit(self.X_train, self.y_train)
        self.MLP_params["max_iter"] = gsearch.best_params_["max_iter"]
        self.MLP_params["alpha"] = gsearch.best_params_["alpha"]
        self.MLP_params["learning_rate"] = gsearch.best_params_["learning_rate"]
        self.MLP_params["activation"] = gsearch.best_params_["activation"]
        self.MLP_params["solver"] = gsearch.best_params_["solver"]
        self.MLP_params["hidden_layer_sizes"] = gsearch.best_params_[
            "hidden_layer_sizes"
        ]
        print("Best MLP Params is: " + str(gsearch.best_params_))
        print("MLP Score: %.3f" % gsearch.best_score_)

    def MLP_model_train(self):
        """Train the MLP model

        Note:
            If the optimal parameters of the MLP model are not calculated,
            the default MLP_params are used.
        """
        print("/*************//* Trainning MLP model *//*************/")
        mlp = MLPClassifier(**self.MLP_params)
        mlp.fit(self.X_train, self.y_train)
        self.y_pred = mlp.predict(self.X_test)

    def MLP_model_summary(self):
        """Summary of the output model

        Note:
            Contains confusion matrix and classification report.
        """
        print("                      MLP Model:")
        print(confusion_matrix(self.y_test, self.y_pred))
        print(
            classification_report(
                self.y_test,
                self.y_pred,
                target_names=["negative", "neutral", "positive"],
            )
        )


if __name__ == "__main__":
    EC = EmotionClassifier()
    EC.Init_train_test_data()

    EC.SVM_model()
    EC.AdaBoost_model()
    EC.MLP_model()
