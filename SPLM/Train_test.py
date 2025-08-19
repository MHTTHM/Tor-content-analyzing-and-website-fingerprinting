from Extractor.DatasetMaker import DatasetMaker
import numpy as np
import torch
import torch.utils.data as Data
import torch.nn as nn
from torch.autograd import Variable
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import warnings
from models.RF_bak import getRF
from sklearn.metrics import precision_recall_fscore_support, accuracy_score,confusion_matrix, balanced_accuracy_score
import argparse
import os, sys
import pandas as pd

EPOCH = 60
BATCH_SIZE = 128
LR = 0.0005
if_use_gpu = 1
num_gpus = torch.cuda.device_count()  # 自动检测GPU数量

# Suppress warnings for cleaner output
# warnings.filterwarnings('ignore')

method = "RF"

current_dir = os.path.dirname(os.path.abspath(__file__))  # 当前文件所在目录（utils）
parent_dir = os.path.dirname(current_dir)  # 项目根目录
extractor_dir = os.path.join(parent_dir, "models")
sys.path.append(extractor_dir)

try:
    from model_func import save_model
except ImportError:
    from models.model_func import save_model

def adjust_learning_rate(optimizer, echo):
    lr = LR * (0.2 ** (echo / EPOCH))
    for para_group in optimizer.param_groups:
        para_group['lr'] = lr

def train(feature_file, vector=300):
    #dataset = DatasetMaker(feature_file)
    x, y = feature_file
    #x[:,:,-1] = 0
    x = x[:, :, :vector]

    num_classes = max(y)+1

    # 加载使用的模型
    cnn = getRF(num_classes)
    print("use model: ", "RF")
    # cnn = CNNTransformerHybrid(num_classes=num_classes)
    # print("use model: ", "CNNTransformerHybrid")

    if if_use_gpu:
        cnn = cnn.cuda()

    optimizer = torch.optim.Adam(cnn.parameters(), lr=LR, weight_decay=0.001)
    # optimizer = torch.optim.Adam(cnn.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCH)

    loss_func = nn.CrossEntropyLoss()
    train_x = torch.unsqueeze(torch.from_numpy(x), dim=1).type(torch.FloatTensor)
    train_x = train_x.view(train_x.size(0), 1, 2, -1)
    train_y = torch.from_numpy(y).type(torch.LongTensor)

    train_data = Data.TensorDataset(train_x, train_y)
    train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

    cnn.train()
    with tqdm(total=EPOCH, desc="Training Progress") as pbar:
        for epoch in range(EPOCH):
            # adjust_learning_rate(optimizer, epoch)
            for step, (tr_x, tr_y) in enumerate(train_loader):
                batch_x = Variable(tr_x.cuda())
                batch_y = Variable(tr_y.cuda())
                output = cnn(batch_x)
                loss = loss_func(output, batch_y)
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(cnn.parameters(), max_norm=1.0)
                optimizer.step()
                
                
                del batch_x, batch_y, output
            
            scheduler.step()
            
            pbar.update(1)
            pbar.set_postfix({'Loss': loss.item()})
        
    save_model(cnn, save_dir="./saved_models", num_classes=num_classes)
    return cnn

def evaluate_model(model, testfile, vector):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    all_results = []

    #dataset = DatasetMaker(path)
    features, test_y = testfile
    #features[:,:,-1] = 0
    features = features[:, :, :vector]

    test_x = torch.unsqueeze(torch.from_numpy(features), dim=1).type(torch.FloatTensor)
    test_x = test_x.to(device)
    test_y = torch.squeeze(torch.from_numpy(test_y)).type(torch.LongTensor)
    test_data = Data.TensorDataset(test_x, test_y)
    test_loader = Data.DataLoader(dataset=test_data, batch_size=1, shuffle=False)
    
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for x, y in tqdm(test_loader, desc=f"Testing {1}/{len(testfile)}"):
            defense_output = model(x).cpu().squeeze().detach().numpy()
            pre = np.argmax(defense_output)
            y_true.append(y.item())
            y_pred.append(pre)
        
        # First n-1 classes metrics
        num_classes = max(y_true) + 1
        mask = np.array(y_true) < (num_classes - 1)
        if np.any(mask):
            precision_n_1, recall_n_1, f1_n_1, _ = precision_recall_fscore_support(
                np.array(y_true)[mask], np.array(y_pred)[mask], average='weighted')
            accuracy_n_1 = accuracy_score(np.array(y_true)[mask], np.array(y_pred)[mask])
        else:
            accuracy_n_1 = precision_n_1 = recall_n_1 = f1_n_1 = None
        
        # Last class metrics
        last_class = num_classes - 1
        mask_last = np.array(y_true) == last_class
        if np.any(mask_last):
            precision_last, recall_last, f1_last, _ = precision_recall_fscore_support(
                np.array(y_true)[mask_last], np.array(y_pred)[mask_last], average='weighted')
            accuracy_last = accuracy_score(np.array(y_true)[mask_last], np.array(y_pred)[mask_last])
        else:
            precision_last = recall_last = f1_last = accuracy_last = None
        
        # Overall metrics
        precision_overall, recall_overall, f1_overall, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted')
        accuracy_overall = accuracy_score(y_true, y_pred)

        balanced_acc = balanced_accuracy_score(y_true, y_pred)
        
        results = {
            'first_n-1_accuracy': accuracy_n_1,
            'first_n-1_precision': precision_n_1,
            'first_n-1_recall': recall_n_1,
            'first_n-1_f1': f1_n_1,
            'last_class_accuracy': accuracy_last,
            'last_class_precision': precision_last,
            'last_class_recall': recall_last,
            'last_class_f1': f1_last,
            'overall_accuracy': accuracy_overall,
            'overall_precision': precision_overall,
            'overall_recall': recall_overall,
            'overall_f1': f1_overall,
            'balanced_acc': balanced_acc
        }
        
        all_results.append(results)
    
    return all_results, y_true, y_pred

def run_experiment(trainfile, testfile, num_runs=10, vector=300):
    first_n1_metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
    overall_metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'balanced_acc': []}
    
    output_dir = "results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for run in range(num_runs):
        print(f"\n=== Run {run+1}/{num_runs} ===")
        
        # Train and evaluate
        model = train(trainfile, vector)
        model.eval()
        results, y_true, y_pred = evaluate_model(model, testfile, vector)

        cm = confusion_matrix(y_true, y_pred)

        cm_df = pd.DataFrame(cm)
        cm_filename = os.path.join(output_dir, f"confusion_matrix_run_{run+1}.csv")
        # index=False 和 header=False 确保输出的CSV文件内容就是纯粹的矩阵数据
        cm_df.to_csv(cm_filename, index=False, header=False)
        print(f"Confusion matrix for run {run+1} saved to {cm_filename}")
        
        # Store metrics
        for result in results:
            first_n1_metrics['accuracy'].append(result['first_n-1_accuracy'])
            first_n1_metrics['precision'].append(result['first_n-1_precision'])
            first_n1_metrics['recall'].append(result['first_n-1_recall'])
            first_n1_metrics['f1'].append(result['first_n-1_f1'])
            
            overall_metrics['accuracy'].append(result['overall_accuracy'])
            overall_metrics['precision'].append(result['overall_precision'])
            overall_metrics['recall'].append(result['overall_recall'])
            overall_metrics['f1'].append(result['overall_f1'])
            overall_metrics['balanced_acc'].append(result['balanced_acc'])
    
    # Calculate averages
    avg_first_n1 = {k: np.mean(v) for k, v in first_n1_metrics.items()}
    std_first_n1 = {k: np.std(v) for k, v in first_n1_metrics.items()}
    
    avg_overall = {k: np.mean(v) for k, v in overall_metrics.items()}
    std_overall = {k: np.std(v) for k, v in overall_metrics.items()}
    
    # Print results
    print("\n=== Final Results (Averaged over {} runs) ===".format(num_runs))
    print("\nFirst n-1 classes:")
    print("{:<12} {:<10} {:<10}".format("Metric", "Mean", "Std"))
    for metric in ['accuracy', 'precision', 'recall', 'f1']:
        print("{:<12} {:<10.4f} {:<10.4f}".format(
            metric.capitalize(), 
            avg_first_n1[metric], 
            std_first_n1[metric]
        ))
    
    print("\nOverall classes:")
    print("{:<12} {:<10} {:<10}".format("Metric", "Mean", "Std"))
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'balanced_acc']:
        print("{:<12} {:<10.4f} {:<10.4f}".format(
            metric.capitalize(), 
            avg_overall[metric], 
            std_overall[metric]
        ))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train and test WF")
    parser.add_argument('--train', help="Input train file path")
    parser.add_argument('--test', help="Input test file path")

    args = parser.parse_args()

    trainpath = 'dataset/Momentum_size+40716_train.npy'
    testpath = ['dataset/Momentum_size+40716_test.npy']

    trainpath = r'D:\2025HS_dataset\Tik_Tok\TAM_Tiktok_OW_train.npy'
    testpath = r'D:\2025HS_dataset\Tik_Tok\TAM_Tiktok_OW_test.npy'

    if args.train:
        trainpath = args.train

    if args.test:
        testpath = args.test

    traindata = DatasetMaker(trainpath, feature_func="TAM")
    testdata = DatasetMaker(testpath, feature_func="TAM")

    tr = (traindata.features, traindata.labels)
    ts = (testdata.features, testdata.labels)


    print(f"Split {1}: Features shape {tr[0].shape}, Labels shape {tr[1].shape}")
    print(f"Split {2}: Features shape {ts[0].shape}, Labels shape {ts[1].shape}")

    run_experiment(tr, ts, num_runs=1, vector=400)