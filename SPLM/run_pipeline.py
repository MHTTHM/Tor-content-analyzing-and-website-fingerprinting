import subprocess
import os
import sys
from Extractor.DatasetMaker import DatasetMaker
from Train_test import run_experiment 
from itertools import product
from contextlib import contextmanager
import numpy as np
sys.path.append("./")
import yaml

@contextmanager
def tee_stdout(file_path):
    class Tee:
        def __init__(self, *files):
            self.files = files
        def write(self, obj):
            for f in self.files:
                f.write(obj)
                f.flush()
        def flush(self):
            for f in self.files:
                f.flush()

    original_stdout = sys.stdout
    with open(file_path, "w") as f:
        sys.stdout = Tee(sys.stdout, f)
        try:
            yield
        finally:
            sys.stdout = original_stdout

def envtips(kwargs):
    tm = '\n'
    if bool(kwargs.get('openworlddataset')):
        tm = 'OW '
        od = ' OWdataset: '+ os.path.basename(kwargs.get('openworldPath'))
    else:
        tm = 'CW '
        od = ''

    tm += kwargs.get('feature')

    funcname = kwargs.get('feature')
    if funcname=='TAM':
        pass
    elif funcname=='momentum':
        tm += ' threshold: '
        tm += str(kwargs.get('threshold'))
        tm += ' zscore: '
        tm += str(kwargs.get('zscore'))
        
        if kwargs.get('flat'):
            
            tm += ' flat_segment_threshold: '
            tm += str(kwargs.get('flat_segment_threshold'))
            tm += ' flat_segment_zscore: '
            tm += str(kwargs.get('flat_segment_zscore'))
            tm += ' rm_threshold: '
            tm += str(kwargs.get('rm_threshold'))
            tm += ' rm_zscore: '
            tm += str(kwargs.get('rm_zscore'))
    elif funcname=='segment':
        tm += ' percentile_threshold: '
        tm += str(kwargs.get('percentile_threshold'))
    
    tm += ' vec: '
    tm += str(kwargs.get('vector'))

    tm += ' dataset: '
    tm += os.path.basename(str(kwargs.get('input_dir')))
    tm += od

    print(tm)

def run_commands(**kwargs):
    
    readfromfolder = kwargs.get('readfromfolder')
    openworld = kwargs.get('openworlddataset')
    func = str(kwargs.get('feature'))
    
    if func == 'TAM':
        kwargs['vector'] = 1800
    vector = kwargs.get('vector', 400)
    
    envtips(kwargs)

    if readfromfolder:
        print("Running: Generating dataset from folder...")
        input_dir = kwargs.get('input_dir')
        output_file =kwargs.get('output_dir')
        print(f"Running: Generating dataset from folder {input_dir}")
        # 生成数据集
        #args = {'threshold': kwargs.get('threshold', None), 'zscore': kwargs.get('zscore', 1), 'vector': kwargs.get('vector', 400), 'flat': kwargs.get('flat', False)}
        
        dataset = DatasetMaker(input_dir, feature_func=func, **kwargs) # feature_func = "TAM" or "momentum"
        proportions = [0.8, 0.2]

        #datasets = [(dataset.features, dataset.labels)]
        #dataset.save('dataset.npy', datasets)
        print(dataset.features[0][0][:10])
        exit(0)

        
        # 开放世界
        if openworld:
            openworldPath = str(kwargs.get('openworldPath'))
            OWtrain = kwargs.get('OWtrain')
            OWtest = kwargs.get('OWtest')
            kwargs['flat'] = False

            OWdata = DatasetMaker(openworldPath, feature_func=func, **kwargs)
            openclass = max(dataset.labels)+1
            OWdata.labels[:] = openclass

            if dataset.features.shape[2] != OWdata.features.shape[2]:
                max_len = max(dataset.features.shape[2], OWdata.features.shape[2])
                dataset_padded = np.pad(dataset.features, 
                                    ((0, 0), (0, 0), (0, max_len - dataset.features.shape[2])))
                OWdata_padded = np.pad(OWdata.features, 
                                    ((0, 0), (0, 0), (0, max_len - OWdata.features.shape[2])))
                dataset.features = np.concatenate((dataset_padded, OWdata_padded), axis=0)
            else:
                dataset.features = np.concatenate((dataset.features, OWdata.features), axis=0)
            
            dataset.labels = np.concatenate((dataset.labels, OWdata.labels), axis=0)

            dataset.handle_openworld([OWtrain, OWtest])

        datasets = dataset.split_dataset(proportions, 42)

        # 打印数据集规模，保存
        for i, (X, y) in enumerate(datasets):
            print(f"Split {i+1}: Features shape {X.shape}, Labels shape {y.shape}")
        
        # try:
        #     dataset.save(output_file, datasets)
        #     #print(f"Dataset saved successfully to {output_file}")
        # except Exception as e:
        #     print(f"Save failed: {str(e)}")
    else:
        trainpath = r"D:\2025HS_dataset\3Momentum_features\Momentum_HS_sz3.5vec500_OW_train.npy"
        testpath = r"D:\2025HS_dataset\3Momentum_features\Momentum_HS_sz3.5vec500_OW_test.npy"
        print("Running: Loading dataset from files")

        # trainpath = r"D:\2025HS_dataset\3Momentum_features\Momentum_divied_OW_train.npy"
        # testpath = r"D:\2025HS_dataset\3Momentum_features\Momentum_divied_OW_test.npy"

        trainpath = kwargs.get('trainfile')
        testpath = kwargs.get('testfile')

        print(f"Train file: {trainpath}")
        trainfile = DatasetMaker(trainpath)
        testfile = DatasetMaker(testpath)
        datasets = [(trainfile.features, trainfile.labels),(testfile.features, testfile.labels)]

        for i, (X, y) in enumerate(datasets):
            print(f"Split {i+1}: Features shape {X.shape}, Labels shape {y.shape}")

    # 运行第二个命令
    print("Running: Training and testing...")
    trainfile, testfile = datasets[0], datasets[1]
    run_experiment(trainfile, testfile, num_runs=1, vector=vector)

from threading import Thread
def run_with_timeout(func, timeout, **kwargs):
    """带超时执行的函数"""
    result = [None]  # 用列表存储结果以便在嵌套函数中修改
    exception = [None]
    
    def wrapper():
        try:
            result[0] = func(**kwargs)
        except Exception as e:
            exception[0] = e
    
    thread = Thread(target=wrapper)
    thread.start()
    thread.join(timeout)
    
    if thread.is_alive():
        # 线程仍在运行，说明超时了
        thread.join(0.1)  # 再给一点时间尝试清理
        if thread.is_alive():
            # 如果仍然存活，我们无法安全终止它，只能放弃
            print(f"Timeout for params: {kwargs}")
        return None, f"Timeout after {timeout} seconds"
    return result[0], exception[0]

if __name__ == "__main__":
    with tee_stdout("output.txt"):
       

        with open('config.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        parameter_sets = config['parameter_sets']
        selected_set = parameter_sets[0]
        params = selected_set['params']

        for combination in product(*params.values()):
            current_params = dict(zip(params.keys(), combination))
            result, error = run_with_timeout(run_commands, timeout=3600, **current_params)  # 设置60秒超时

            if error:
                print(f"Failed for params: {current_params}")
                print(f"Error: {error}")
