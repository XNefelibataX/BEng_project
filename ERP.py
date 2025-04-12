"""
 Sample script using EEGNet to classify Event-Related Potential (ERP) EEG data
 from a four-class classification task, using the sample dataset provided in
 the MNE [1, 2] package:
     https://martinos.org/mne/stable/manual/sample_dataset.html#ch-sample-data
   
 The four classes used from this dataset are:
     LA: Left-ear auditory stimulation
     RA: Right-ear auditory stimulation
     LV: Left visual field stimulation
     RV: Right visual field stimulation

 The code to process, filter and epoch the data are originally from Alexandre
 Barachant's PyRiemann [3] package, released under the BSD 3-clause. A copy of 
 the BSD 3-clause license has been provided together with this software to 
 comply with software licensing requirements. 
 
 When you first run this script, MNE will download the dataset and prompt you
 to confirm the download location (defaults to ~/mne_data). Follow the prompts
 to continue. The dataset size is approx. 1.5GB download. 
 
 For comparative purposes you can also compare EEGNet performance to using 
 Riemannian geometric approaches with xDAWN spatial filtering [4-8] using 
 PyRiemann (code provided below).

 [1] A. Gramfort, M. Luessi, E. Larson, D. Engemann, D. Strohmeier, C. Brodbeck,
     L. Parkkonen, M. Hämäläinen, MNE software for processing MEG and EEG data, 
     NeuroImage, Volume 86, 1 February 2014, Pages 446-460, ISSN 1053-8119.

 [2] A. Gramfort, M. Luessi, E. Larson, D. Engemann, D. Strohmeier, C. Brodbeck, 
     R. Goj, M. Jas, T. Brooks, L. Parkkonen, M. Hämäläinen, MEG and EEG data 
     analysis with MNE-Python, Frontiers in Neuroscience, Volume 7, 2013.

 [3] https://github.com/alexandrebarachant/pyRiemann. 

 [4] A. Barachant, M. Congedo ,"A Plug&Play P300 BCI Using Information Geometry"
     arXiv:1409.0107. link

 [5] M. Congedo, A. Barachant, A. Andreev ,"A New generation of Brain-Computer 
     Interface Based on Riemannian Geometry", arXiv: 1310.8115.

 [6] A. Barachant and S. Bonnet, "Channel selection procedure using riemannian 
     distance for BCI applications," in 2011 5th International IEEE/EMBS 
     Conference on Neural Engineering (NER), 2011, 348-351.

 [7] A. Barachant, S. Bonnet, M. Congedo and C. Jutten, “Multiclass 
     Brain-Computer Interface Classification by Riemannian Geometry,” in IEEE 
     Transactions on Biomedical Engineering, vol. 59, no. 4, p. 920-928, 2012.

 [8] A. Barachant, S. Bonnet, M. Congedo and C. Jutten, “Classification of 
     covariance matrices using a Riemannian-based kernel for BCI applications“, 
     in NeuroComputing, vol. 112, p. 172-178, 2013.`


 Portions of this project are works of the United States Government and are not
 subject to domestic copyright protection under 17 USC Sec. 105.  Those 
 portions are released world-wide under the terms of the Creative Commons Zero 
 1.0 (CC0) license.  
 
 Other portions of this project are subject to domestic copyright protection 
 under 17 USC Sec. 105.  Those portions are licensed under the Apache 2.0 
 license.  The complete text of the license governing this material is in 
 the file labeled LICENSE.TXT that is a part of this project's official 
 distribution. 
"""
import os
import math
from sklearn.model_selection import KFold
import numpy as np
# EEGNet-specific imports
from EEGModels import EEGNet
from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras import backend as K
from sklearn.model_selection import train_test_split
# from pyriemann.utils.viz import plot_confusion_matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from tensorflow.keras.optimizers import Adam
# tools for plotting confusion matrices
from matplotlib import pyplot as plt
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# while the default tensorflow ordering is 'channels_last' we set it here
# to be explicit in case if the user has changed the default ordering
K.set_image_data_format('channels_last')
kernels, chans, samples = 1, 64, 656

############################# EEGNet portion ##################################
class EEGNet_Data(object):
    def __init__(self, file_test_datapath, file_train_datapath, file_path):
        self.file_train_datapath = file_train_datapath
        self.file_test_datapath = file_test_datapath
        self.file_path = file_path
        self.mean = []
        self.std = []

    def normalize(self, train):
        # input should be seg, chan, time/f
        # data: sample x channel x data
        for channel in range(train.shape[1]):
            train[:, channel, :] = (train[:, channel, :] - self.mean[channel]) / self.std[channel]
        return train


    def get_source_data(self):

        train_file_path = self.file_train_datapath
        test_file_path = self.file_test_datapath
        with open(train_file_path, 'r') as f:
            train_lines = f.readlines()
        train_lines_data = [np.expand_dims(np.load(os.path.join(self.file_path, line.split(' ')[0])), axis=0) for line
                            in train_lines]
        train_lines_data = np.concatenate(train_lines_data, axis=0) * 1000000
        train_lines_label = np.array([int(line.split(' ')[1].replace('\n', '')) for line in train_lines])

        with open(test_file_path, 'r') as f:
            test_lines = f.readlines()
        test_lines_data = [np.expand_dims(np.load(os.path.join(self.file_path, line.split(' ')[0])), axis=0) for line in
                           test_lines]
        test_lines_data = np.concatenate(test_lines_data, axis=0) * 1000000
        test_lines_label = np.array([int(line.split(' ')[1].replace('\n', '')) for line in test_lines])

        TOTAL_data = np.concatenate([train_lines_data, test_lines_data], axis=0)

        for channel in range(TOTAL_data.shape[1]):
            self.mean.append(np.mean(TOTAL_data[:, channel, :]))
            self.std.append(np.std(TOTAL_data[:, channel, :]))

        data_x_train = self.normalize(train_lines_data)
        data_x_test = self.normalize(test_lines_data)

        return data_x_train, train_lines_label, data_x_test, test_lines_label


def compute_accuracy(confusion_matrix):
    """
    Compute the accuracy, given the output and the target.
    Output should be a 1-D vector, and target should be a 1-D vector. They both
    have the same length.
    """
    # 打开和关闭左拳【标签为0】，打开和关闭右拳【标签为1】
    # 想象打开和关闭左拳【标签为2】，想象打开和关闭右拳【标签为3】
    n_classes = 4
    acc_per_class = []
    for i in range(n_classes):
        tp = confusion_matrix[i][i]
        fn = np.sum(confusion_matrix[i, :]) - tp
        denom = tp + fn
        acc = tp / denom if denom > 0 else 0.0  # 防止除零
        acc_per_class.append(acc)
    macro_acc = np.mean(acc_per_class)
    '''
    计算真实运行与想象的二大类准确度
    '''
    TP = np.sum(confusion_matrix[:2,:2])
    TN = np.sum(confusion_matrix[2:,2:])
    Accuracy_real_imagine = (TP + TN) / np.sum(confusion_matrix)

    return macro_acc, acc_per_class, Accuracy_real_imagine


def plot_training_history(model):
    plt.figure(figsize=(12,5))

    # Accuracy
    plt.subplot(1,2,1)
    plt.plot(model.history['accuracy'], label='Train Accuracy')
    plt.plot(model.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss
    plt.subplot(1,2,2)
    plt.plot(model.history['loss'], label='Train Loss')
    plt.plot(model.history['val_loss'], label='Validation Loss')
    plt.title('Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig("training_curve.png")
    plt.show()

#
# if __name__ == '__main__':
#
#
#     process_type = 'cross test'   # 使用交叉验证
#     test_dataset_path = r"D:\python\BCI_project\BCI_RC\label_test.txt"
#     train_dataset_path = r"D:\python\BCI_project\BCI_RC\label_train.txt"
#     file_path = r"D:\python\BCI_project\BCI_RC\process_data"
#     model_path = r'./tmp/checkpoint.h5'
#
#     # 加载数据
#     our_data = EEGNet_Data(file_test_datapath=test_dataset_path, file_train_datapath=train_dataset_path,
#                            file_path=file_path)
#     X_all_data, Y_all_labels, _, _ = our_data.get_source_data()  # combine train and test data
#
#     # 数据合并
#     all_data = X_all_data  # 样本数据尺寸为 (样本数, 通道数, 时间点数)
#     all_labels = Y_all_labels  # 标签数据为 (样本数, )
#
#     # 将标签转换为 one-hot 编码
#     all_labels_one_hot = to_categorical(all_labels)
#
#     # 调整数据格式到 NHWC（样本数，通道数，时间点，核数），准备输入 EEGNet
#     all_data = all_data.reshape(all_data.shape[0], chans, samples, kernels)
#
#     # 初始化 KFold，参数 n_splits = 10，表示 10 折交叉验证
#     kfold = KFold(n_splits=5, shuffle=True, random_state=42)
#
#     fold_no = 1
#     n_classes = 4
#     accuracies = []  # 记录每次折测试的准确率
#     macro_Acc_list = []  # 记录每次折测试的 macro_Acc
#     Accuracy_real_imagine_list = []  # 记录每次折测试的 Accuracy_real_imagine
#     wei_acc_0 = []  # 记录每次折测试的 打开和关闭左拳【标签为0】准确度
#     wei_acc_1 = []  # 记录每次折测试的 打开和关闭右拳【标签为1】准确度
#     wei_acc_2 = []  # 记录每次折测试的 想象打开和关闭左拳【标签为2】准确度
#     wei_acc_3 = []  # 记录每次折测试的 想象打开和关闭右拳【标签为3】准确度
#     avg_conf_matrix = np.zeros((n_classes, n_classes))
#     true_all = []
#     pred_all = []
#     loss_all = []
#
#     for train_idx, test_idx in kfold.split(all_data, all_labels_one_hot):
#         print(f"Training on fold {fold_no}...")
#
#         # 按索引划分训练集和测试集
#         X_train, X_test = all_data[train_idx], all_data[test_idx]
#         Y_train, Y_test = all_labels_one_hot[train_idx], all_labels_one_hot[test_idx]
#
#         # 配置 EEGNet 模型
#         model = EEGNet(nb_classes=4, Chans=chans, Samples=samples,
#                        dropoutRate=0.3, kernLength=32, F1=16, D=2, F2=32,
#                        dropoutType='Dropout', ratio=8)
#
#         model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#
#         # 设置模型回调，保存最佳模型权重
#         checkpointer = ModelCheckpoint(filepath=model_path, verbose=1, save_best_only=True)
#
#         # # 训练模型
#         fittedModel = model.fit(X_train, Y_train, batch_size=32, epochs=50,
#                                 verbose=2, validation_data=(X_test, Y_test),
#                                 callbacks=[checkpointer])
#
#         #保存训练 loss 和 val loss
#         loss_all.append(fittedModel.history['val_loss'][-1])
#
#         # 加载最佳模型权重进行测试
#         model.load_weights(model_path)
#
#         # 测试集评估
#         probs = model.predict(X_test)
#         preds = probs.argmax(axis=-1)
#         true = Y_test.argmax(axis=-1)
#         acc = np.mean(preds == Y_test.argmax(axis=-1))
#         accuracies.append(acc)
#
#         # 混淆矩阵
#         cm = confusion_matrix(true, preds,labels= np.arange(n_classes))
#         avg_conf_matrix += cm
#         test_confusion_matrix = cm.copy()
#         total_acc, wei_acc, Accuracy_real_imagine = compute_accuracy(test_confusion_matrix)
#
#
#         # 打开和关闭左拳【标签为0】，打开和关闭右拳【标签为1】
#         # 想象打开和关闭左拳【标签为2】，想象打开和关闭右拳【标签为3】
#         print('Fold {} | Test macro_Acc: {:.4f} | Opening and closing the left fist [labeled 0] Accuracy: {:.4f} | Opening and closing the right fist [labeled 1] Accuracy: {:.4f}'
#               'Imagine opening and closing the left fist [labeled 2] Accuracy: {:.4f} | Imagine opening and closing the right fist [labeled 3] Accuracy: {:.4f} | Accuracy_real_imagine: {:.4f}'.
#               format(fold_no, total_acc, wei_acc[0], wei_acc[1], wei_acc[2], wei_acc[3], Accuracy_real_imagine))
#         macro_Acc_list.append(total_acc)
#         Accuracy_real_imagine_list.append(Accuracy_real_imagine)
#         wei_acc_0.append(wei_acc[0])
#         wei_acc_1.append(wei_acc[1])
#         wei_acc_2.append(wei_acc[2])
#         wei_acc_3.append(wei_acc[3])
#
#         #汇总所有标签
#         true_all.extend(true)
#         pred_all.extend(preds)
#
#         fold_no += 1
#
#     # 输出最终交叉验证平均结果
#     print('Cross-validation accuracies | Test macro_Acc: {:.4f} | Opening and closing the left fist [labeled 0] Accuracy: {:.4f} | '
#           'Opening and closing the right fist [labeled 1] Accuracy: {:.4f} Imagine opening and closing the left fist [labeled 2] Accuracy: {:.4f} | Imagine opening and closing the right fist [labeled 3] Accuracy: {:.4f} | Accuracy_real_imagine: {:.4f}'.
#           format(math.fsum(macro_Acc_list)/len(macro_Acc_list), math.fsum(wei_acc_0)/len(wei_acc_0),math.fsum(wei_acc_1)/len(wei_acc_1),
#                  math.fsum(wei_acc_2)/len(wei_acc_2),math.fsum(wei_acc_3)/len(wei_acc_3), math.fsum(Accuracy_real_imagine_list)/len(Accuracy_real_imagine_list)))

#=======================================================================================================================
if __name__ == '__main__':

    '''
    process_type: 'train and test' or 'test',设置运行状态，'train and test'表示训练和测试，'test'表示只测试；
    test_dataset_path: 测试数据集路径；
    train_dataset_path: 训练数据集路径；
    file_path: 数据集路径；
    model_path: 模型保存路径；
    '''
    process_type = 'train and test'  # 'train and test' or 'test'
    test_dataset_path = r"D:\python\BCI_project\BCI_RC\label_test.txt"
    train_dataset_path = r"D:\python\BCI_project\BCI_RC\label_train.txt"
    file_path = r"D:\python\BCI_project\BCI_RC\process_data"
    model_path = r'./tmp/checkpoint.h5'

    our_data = EEGNet_Data(file_test_datapath=test_dataset_path, file_train_datapath=train_dataset_path,
                           file_path=file_path)
    X_train1, Y_train1, X_test0, Y_test0 = our_data.get_source_data()

    X_val1, X_test1, Y_val1, Y_test1 = train_test_split(
        X_test0, Y_test0, test_size=0.5, stratify=Y_test0
    )
    # convert labels to one-hot encodings.
    Y_train = to_categorical(Y_train1)
    Y_test = to_categorical(Y_test1)
    Y_val = to_categorical(Y_val1)

    # convert data to NHWC (trials, channels, samples, kernels) format. Data
    # contains 60 channels and 151 time-points. Set the number of kernels to 1.
    X_train = X_train1.reshape(X_train1.shape[0], chans, samples, kernels)
    X_test = X_test1.reshape(X_test1.shape[0], chans, samples, kernels)
    X_val = X_val1.reshape(X_val1.shape[0], chans, samples, kernels)

    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')
    print(X_val.shape[0], 'validation samples')

    # configure the EEGNet-8,2,16 model with kernel length of 32 samples (other
    # model configurations may do better, but this is a good starting point)
    model = EEGNet(nb_classes = 4, Chans = chans, Samples = samples,
                   dropoutRate = 0.3, kernLength = 32, F1 = 16, D = 2, F2 = 32,
                   dropoutType = 'Dropout',ratio=8)

    optimizer = Adam(learning_rate=1e-3)
    # compile the model and set the optimizers
    model.compile(loss='categorical_crossentropy', optimizer=optimizer,
                  metrics = ['accuracy'])

    # count number of parameters in the model
    numParams    = model.count_params()

    # set a valid path for your system to record model checkpoints
    checkpointer = ModelCheckpoint(filepath='/tmp/checkpoint.h5', verbose=1,
                                   save_best_only=True)

    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=10,
        verbose=1,
        restore_best_weights=True
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.1,
        patience=5,
        verbose=1,
        min_lr=1e-6
    )


    # the syntax is {class_1:weight_1, class_2:weight_2,...}. Here just setting
    # the weights all to be 1
    class_weights = {0:1, 1:1, 2:1, 3:1}

    if process_type == 'train and test':

        fittedModel = model.fit(X_train, Y_train, batch_size = 32, epochs = 100,
                                verbose = 2, validation_data=(X_val, Y_val),
                                callbacks=[checkpointer, early_stop, reduce_lr], class_weight = class_weights)

        model.save_weights(model_path)
        plot_training_history(fittedModel)
        history_df = pd.DataFrame(fittedModel.history)
        history_df.to_csv("training_metrics.csv",index = False)

    else:
        pass

    # load optimal weights
    if os.path.exists(model_path):
        model.load_weights(model_path)
        print('Model loaded from checkpoint.')


    probs = model.predict(X_test)
    preds = probs.argmax(axis = -1)
    acc = np.mean(preds == Y_test.argmax(axis=-1))
    print("Classification accuracy: %f " % (acc))
    test_confusion_matrix = confusion_matrix(Y_test.argmax(axis=-1), preds)
    total_acc, wei_acc, Accuracy_real_imagine = compute_accuracy(test_confusion_matrix)
    # 打开和关闭左拳【标签为0】，打开和关闭右拳【标签为1】
    # 想象打开和关闭左拳【标签为2】，想象打开和关闭右拳【标签为3】
    print('Test macro_Acc: {:.4f} | Opening and closing the left fist [labeled 0] Accuracy: {:.4f} | Opening and closing the right fist [labeled 1] Accuracy: {:.4f}'
          'Imagine opening and closing the left fist [labeled 2] Accuracy: {:.4f} | Imagine opening and closing the right fist [labeled 3] Accuracy: {:.4f} | Accuracy_real_imagine: {:.4f}'.
          format(total_acc,wei_acc[0],wei_acc[1],wei_acc[2],wei_acc[3],Accuracy_real_imagine))


    print("\nConfusion Matrix (raw values):")
    print("Labels: [0=left fist, 1=right fist, 2=imagine left fist, 3=imagine right fist] ")
    print(test_confusion_matrix)

    target_names = ['left fist [0]', 'right fist [1]', 'imagine left fist [2]', 'imagine right fist [3]']
    print("\nClassification Report:")
    print(classification_report(Y_test.argmax(axis=-1), preds, target_names=target_names, digits=4))



    # 打印所有参数数量
    trainable_count = np.sum([np.prod(v.get_shape()) for v in model.trainable_weights])
    non_trainable_count = np.sum([np.prod(v.get_shape()) for v in model.non_trainable_weights])

    print("Trainable parameters:", trainable_count)
    print("Non-trainable parameters:", non_trainable_count)
    model.summary()









