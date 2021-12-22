import numpy as np
import os
from PIL import Image
from sklearn.svm import SVC
import joblib
from sklearn.metrics import confusion_matrix, classification_report
import glob
import time
class DataLoader(object):
    """训练前的预处理"""
    def get_files(self, fpath, fmt = "*.png"):
        """获取指定文件夹中指定格式的文件列表;
        paras:
            filepath: str, file path,
            formats: str, file format,
        return: list;"""
        tmp = os.path.join(fpath,fmt)
        fs = glob.glob(tmp)
        return fs
    def get_data_labels(self, fpath = "test"):
        paths = glob.glob(fpath + os.sep + "*")
        X = []
        y = []
        for fpath in paths:
            fs = self.get_files(fpath)
            for fn in fs:
                X.append(self.img2vec(fn))
            label = np.repeat(int(os.path.basename(fpath)), len(fs))
            y.append(label)
        labels = y[0]
        for i in range(len(y) - 1):
            labels = np.append(labels, y[i + 1])
        return np.array(X), labels
    def img2vec(self, fn):
        '''将jpg等格式的图片转为向量'''
        im = Image.open(fn).convert('L')
        im = im.resize((28,28))
        tmp = np.array(im)
        vec = tmp.ravel()
        return vec 
    def save_data(self, X_data, y_data, fn = "mnist_train_data"):
        """将数据保存到本地;"""
        np.savez_compressed(fn, X = X_data, y = y_data)
    def load_data(self, fn = "mnist_train_data.npz"):
        """从本地加载数据;"""
        data = np.load(fn)
        X_data = data["X"]
        y_data = data["y"]
        return X_data, y_data
class Trainer(object):
    '''训练器;'''
    def svc(self, x_train, y_train):
        '''构建分类器'''
        model = SVC(kernel = 'poly',degree = 4,probability= True)
        model.fit(x_train, y_train)
        return model
    def save_model(self, model, output_name):
        '''保存模型'''
        joblib.dump(model,output_name, compress = 1)
    def load_model(self, model_path):
        '''加载模型'''
        clf = joblib.load(model_path)
        return clf
class Tester(object):
    '''测试器;'''
    def __init__(self, model_path):
        trainer = Trainer()      
        self.clf = trainer.load_model(model_path)
    def clf_metrics(self,X_test,y_test):
        """评估分类器效果"""
        pred = self.clf.predict(X_test)
        cnf_matrix = confusion_matrix(y_test, pred)
        score = self.clf.score(X_test, y_test)
        clf_repo = classification_report(y_test, pred)
        return cnf_matrix, score, clf_repo
    def predict(self, fn):
        '''样本预测;'''
        loader = DataLoader()
        tmp = loader.img2vec(fn)
        X_test = tmp.reshape(1, -1)
        ans = self.clf.predict(X_test)
        return ans
def run_train():
    t0 = time.time()
    loader = DataLoader()
    trainer = Trainer()
    X, y = loader.get_data_labels()
    t1 = time.time()
    print(t1 - t0)
    clf = trainer.svc(X, y)
    print(time.time() - t1)
    joblib.dump(clf,"./mnist_svm.m", compress = 3)
    trainer.save_model(clf, "mnist_svm.m")
    X_test, y_test = loader.get_data_labels("test")
    tester = Tester("mnist_svm.m")
    mt, score, repo = tester.clf_metrics(X_test, y_test)
    return clf, X, y
 
run_train()