import math
import os
import pickle

import numpy as np
from scipy import signal
from scipy.io import loadmat
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC

fs = 200
band_dict = {
    "delta": [1, 4],
    "theta": [4, 8],
    "alpha": [8, 14],
    "beta": [14, 31],
    "gamma": [31, 50],
}

channels = [0, 2, 25, 29, 41, 49, 58, 60, 5, 13, 7, 11, 23, 31, 43, 47]


class SVMEmotionRecognition:
    """This is a class for an SVM emotion recognition classifier

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
    C = 0.1
    kernel = "linear"

    def __init__(
        self, data_dir="../SEED/Preprocessed_EEG/", feature_data_dir="../TrainingData/"
    ):
        """Inits SVMEmotionRecognition Class

        Args:
            data_dir (str): The path of SEED dataset directory.
            feature_data_dir (str): The path of feature data directory.
        """
        if not self.data_exist(feature_data_dir):
            print("/*********//* Feature data does not exit *//**********/")
            print("/****************//* creating data *//****************/")
            self.feature_extraction(data_dir, feature_data_dir)
        else:
            print("/*************//* Feature data exist *//**************/")
            print("/****************//* reading data *//*****************/")
            self.feature_data_io(feature_data_dir, "rb")

    def feature_extraction(self, data_dir, feature_data_dir):
        """Read the data, perform bandpass filtering on the data,
        and calculate the DE of the data

        Args:
            data_dir (str): The path of SEED dataset directory.
            feature_data_dir (str): The path of feature data directory.
        """
        label_Y = loadmat(data_dir + "label.mat")["label"][0]
        for file_name in os.listdir(data_dir):
            if file_name in ["label.mat", "readme.txt"]:
                continue
            file_data = loadmat(data_dir + file_name)
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
                self.datasets_X.append(dataset_X)
                self.datasets_y.append(label_Y[index])

        self.datasets_X = np.array(self.datasets_X)
        self.datasets_X = self.datasets_X.reshape((675, 8000))  # 16 * 100 * 15
        self.datasets_y = self.datasets_y
        self.feature_data_io(feature_data_dir, "wb")

    def feature_data_io(self, feature_data_dir, method):
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

    # 判断是否存在预处理数据
    def data_exist(self, path):
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

    def DataNoneError(self):
        """Judgment, whether to throw an error

        Note:
            Throws an error if the following variable is an empty list:
            datasets_X, datasets_y, X_train, X_test, y_train, y_test
        """
        if (
            len(self.datasets_X) == 0
            or len(self.datasets_y) == 0
            or len(self.X_train) == 0
            or len(self.y_train) == 0
            or len(self.X_test) == 0
            or len(self.y_test) == 0
        ):
            print("!!!One of data is None, please check again!!!")
            raise RuntimeError("TrainDataNoneError")

    def initialize_training_data(self, test_size=0.2):
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

    def find_best_model_params(self):
        """Find the best parameters for an SVM model

        Note:
            Class variable set default value C=0.1, kernel = "linear"
        """
        print("/**********//* Finding best model params *//**********/")
        self.DataNoneError()
        params = {
            "C": np.arange(0.1, 1, 0.1),
            "kernel": ["linear", "poly", "rbf", "sigmoid"],
        }
        gsearch = GridSearchCV(
            estimator=SVC(kernel="linear"), param_grid=params, scoring="accuracy", cv=5
        )
        gsearch.fit(self.X_train, self.y_train)
        self.C = gsearch.best_params_["C"]
        self.kernel = gsearch.best_params_["kernel"]
        print("Best Params is: " + str(gsearch.best_params_))
        print("Score: %.3f" % gsearch.best_score_)

    def train_SVM_model(self):
        """Train the SVM model

        Note:
            If the optimal parameters of the SVM model are not calculated,
            the default values C=0.1, kernel = "linear" are used.
        """
        print("/***************//* Trainning model *//***************/")
        svclassifier = SVC(C=self.C, kernel=self.kernel)
        svclassifier.fit(self.X_train, self.y_train)
        self.y_pred = svclassifier.predict(self.X_test)

    def model_summary(self):
        """Summary of the output model

        Note:
            Contains confusion matrix and classification report.
        """
        self.DataNoneError()
        print("SVM Model:")
        print(confusion_matrix(self.y_test, self.y_pred))
        print(
            classification_report(
                self.y_test,
                self.y_pred,
                target_names=["negative", "neutral", "positive"],
            )
        )


if __name__ == "__main__":
    SVM = SVMEmotionRecognition(
        "../Datasets/SEED/Preprocessed_EEG/", "../TrainingData/"
    )
    SVM.initialize_training_data()
    SVM.find_best_model_params()
    SVM.train_SVM_model()
    SVM.model_summary()
