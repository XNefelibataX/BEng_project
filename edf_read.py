import mne
import os
import numpy as np
from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt
#=======================================================================================================================

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    """
    Removal of high-frequency noise (muscle artifacts, etc.)
    and low-frequency trends (heartbeat interference, etc.)
    """
    nyq = 0.5 * fs #  Nyquist frequency
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = lfilter(b, a, data)
    return y

''' 
Raw dataset: S100, S092, S089, S088, S104, and S106 were excluded due to inconsistent data lengths.
'''

def get_edf_data(edf_file_path):
    evevent = []
    save_file_path = r'.\process_data'
    # Define action labels for real and imaginary parts
    real_action = ['03','07','11']
    # Open and close the left or right fist, T1 corresponds to open and close the left fist [labeled 0],
    # T2 corresponds to open and close the right fist [labeled 1].
    imagine_action = ['04','08','12']
    # Imagine opening and closing a left or right fist,
    # T1 corresponds to imagining opening and closing a left fist [labeled 2],
    # and T2 corresponds to imagining opening and closing a right fist [labeled 3].

    # keep label address
    txt_file_path = r'.\label.txt'

    file_name_list = os.listdir(edf_file_path)
    k = 0
    for file_name in file_name_list:
        data_len = 0
        if file_name.endswith('.edf'):
            if file_name.split('.')[0][-2:] in real_action: # example: S001R01.edf

                # read .edf file
                raw = mne.io.read_raw_edf(os.path.join(edf_file_path, file_name), preload=True)

                # get data, a NumPy array as (num_channels, num_samples)
                K_data = raw.get_data()
                num_channels = K_data.shape[0]  # 通道数量
                # Apply bandpass filter，frequency range 8–30 Hz
                # num_channels = 64
                fs = 160
                lowcut = 8.0
                highcut = 30.0
                k_data = np.zeros_like(K_data)  # normalize the signal array with same size after being filtered

                # bandpass filter every channel
                for i in range(num_channels):
                    k_data[i] = bandpass_filter(K_data[i], lowcut, highcut, fs)

                # get event label
                events_from_annot, event_dict = mne.events_from_annotations(raw)

                # transform label to 0 or 1
                events_from_annot[:, 2] = events_from_annot[:, 2] - 2

                for i in range(events_from_annot.shape[0]):
                    if events_from_annot[i, 2] == 0:
                        if i == events_from_annot.shape[0] - 1:

                            k_data1 = k_data[:, events_from_annot[i, 0]:events_from_annot[i, 0] + data_len]

                            if k_data1.shape[1] != 656:
                                k += 1
                                print('error', file_name)
                                evevent.append(edf_file_path + file_name)
                            else:
                                np.save(os.path.join(save_file_path, file_name.split('.')[0] + '_' + str(i) + '.npy'), k_data1)
                                with open(txt_file_path, 'a') as f:
                                    f.write(file_name.split('.')[0] + '_' + str(i) + '.npy' + ' ' + str(0) + '\n')

                        else:
                            data_len = events_from_annot[i + 1, 0] - events_from_annot[i, 0]
                            k_data1 = k_data[:,events_from_annot[i, 0]:events_from_annot[i + 1, 0]]
                            if k_data1.shape[1] != 656:
                                k += 1
                                print('error', file_name)
                                evevent.append(edf_file_path+file_name)
                            else:
                                np.save(os.path.join(save_file_path, file_name.split('.')[0] + '_' + str(i) + '.npy'), k_data1)
                                with open(txt_file_path, 'a') as f:
                                    f.write(file_name.split('.')[0] + '_' + str(i) + '.npy' + ' ' + str(0) + '\n')

                    elif events_from_annot[i, 2] == 1:
                        if i == events_from_annot.shape[0] - 1:

                            k_data1 = k_data[:, events_from_annot[i, 0]:events_from_annot[i, 0]+data_len]
                            if k_data1.shape[1] != 656:
                                k += 1
                                print('error', file_name)
                                evevent.append(edf_file_path+file_name)
                            else:
                                np.save(os.path.join(save_file_path, file_name.split('.')[0] + '_' + str(i) + '.npy'), k_data1)
                                with open(txt_file_path, 'a') as f:
                                    f.write(file_name.split('.')[0] + '_' + str(i) + '.npy' + ' '+ str(1)+'\n')

                        else:
                            data_len = events_from_annot[i + 1, 0] - events_from_annot[i, 0]

                            k_data1 = k_data[:, events_from_annot[i, 0]:events_from_annot[i + 1, 0]]
                            if k_data1.shape[1] != 656:
                                k += 1
                                print('error', file_name)
                                evevent.append(edf_file_path+file_name)
                            else:
                                np.save(os.path.join(save_file_path, file_name.split('.')[0] + '_' + str(i) + '.npy'), k_data1)
                                with open(txt_file_path, 'a') as f:
                                    f.write(file_name.split('.')[0] + '_' + str(i) + '.npy' + ' ' + str(1) + '\n')

            elif file_name.split('.')[0][-2:] in imagine_action:

                raw = mne.io.read_raw_edf(os.path.join(edf_file_path, file_name), preload=True)


                K_data = raw.get_data()

                k_data = np.zeros_like(K_data)

                num_channels = K_data.shape[0]  # 通道数量
                # num_channels = 64
                fs = 160
                lowcut = 8.0
                highcut = 30.0


                for i in range(num_channels):
                    k_data[i] = bandpass_filter(K_data[i], lowcut, highcut, fs)


                events_from_annot, event_dict = mne.events_from_annotations(raw)
                # Labels of 2 or 3 are themselves 2 or 3 and do not need to be converted
                # events_from_annot[:, 2] = events_from_annot[:, 2]

                for i in range(events_from_annot.shape[0]):
                    if events_from_annot[i, 2] == 2:
                        if i == events_from_annot.shape[0] - 1:

                            k_data1 = k_data[:, events_from_annot[i, 0]:events_from_annot[i, 0] + data_len]
                            if k_data1.shape[1] != 656:
                                k += 1
                                print('error', file_name)
                                evevent.append(edf_file_path+file_name)
                            else:
                                np.save(os.path.join(save_file_path, file_name.split('.')[0] + '_' + str(i) + '.npy'), k_data1)
                                with open(txt_file_path, 'a') as f:
                                    f.write(file_name.split('.')[0] + '_' + str(i) + '.npy' + ' ' + str(2) + '\n')

                        else:
                            data_len = events_from_annot[i + 1, 0] - events_from_annot[i, 0]

                            k_data1 = k_data[:,events_from_annot[i, 0]:events_from_annot[i + 1, 0]]
                            if k_data1.shape[1] != 656:
                                k += 1
                                print('error', file_name)
                                evevent.append(edf_file_path+file_name)
                            else:
                                np.save(os.path.join(save_file_path, file_name.split('.')[0] + '_' + str(i) + '.npy'), k_data1)
                                with open(txt_file_path, 'a') as f:
                                    f.write(file_name.split('.')[0] + '_' + str(i) + '.npy' + ' ' + str(2) + '\n')

                    elif events_from_annot[i, 2] == 3:
                        if i == events_from_annot.shape[0] - 1:

                            k_data1 = k_data[:, events_from_annot[i, 0]:events_from_annot[i, 0]+data_len]
                            if k_data1.shape[1] != 656:
                                k += 1
                                print('error', file_name)
                                evevent.append(edf_file_path+file_name)
                            else:
                                np.save(os.path.join(save_file_path, file_name.split('.')[0] + '_' + str(i) + '.npy'), k_data1)
                                with open(txt_file_path, 'a') as f:
                                    f.write(file_name.split('.')[0] + '_' + str(i) + '.npy' + ' ' + str(3) + '\n')

                        else:
                            data_len = events_from_annot[i + 1, 0] - events_from_annot[i, 0]

                            k_data1 = k_data[:, events_from_annot[i, 0]:events_from_annot[i + 1, 0]]
                            if k_data1.shape[1] != 656:
                                k += 1
                                print('error', file_name)
                                evevent.append(edf_file_path+file_name)
                            else:
                                np.save(os.path.join(save_file_path, file_name.split('.')[0] + '_' + str(i) + '.npy'), k_data1)
                                with open(txt_file_path, 'a') as f:
                                    f.write(file_name.split('.')[0] + '_' + str(i) + '.npy' + ' ' + str(3) + '\n')
        else:
            continue
    return evevent, k

def plot_eeg_signals_C3(data, fs=160, lowcut=8.0, highcut=30.0, channel=8, n_samples=1000):
    """
    绘制指定通道的原始 EEG 信号和滤波后的 EEG 信号。

    参数：
        data: np.array，EEG 数据矩阵，形状为 (通道数, 采样点数)
        fs: int，采样率
        lowcut: float，滤波器低截止频率
        highcut: float，滤波器高截止频率
        channel: int，要绘制的通道索引（从 0 开始）
        n_samples: int，要绘制的采样点数量
    """
    # 获取指定通道的原始信号（只取前 n_samples 个采样点）
    original_signal = data[channel, :n_samples]

    # 对指定通道的信号进行滤波
    filtered_signal_830 = bandpass_filter(data[channel], lowcut, highcut, fs)[:n_samples]
    filtered_signal_040 = bandpass_filter(data[channel], 0.000001, 40, fs)[:n_samples]

    plt.figure(figsize=(10, 6))
    plt.plot(original_signal, label='raw signal', color='blue')
    plt.xlabel('sampling points')
    plt.ylabel('amplitude')
    plt.title(f'channel C3 raw signal')
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(filtered_signal_040, label='filtered signal(0-40)', color='red')
    plt.xlabel('sampling points')
    plt.ylabel('amplitude')
    plt.title(f'channel C3 filtered signal(0-40Hz)')
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(filtered_signal_830, label='filtered signal(8-30)', color='black')
    plt.xlabel('sampling points')
    plt.ylabel('amplitude')
    plt.title(f'channel C3 filtered signal(8-30Hz)')
    plt.legend()
    plt.show()



def plot_eeg_signals_C4(data, fs=160, lowcut=8.0, highcut=30.0, channel=12, n_samples=1000):
    """
    绘制指定通道的原始 EEG 信号和滤波后的 EEG 信号。

    参数：
        data: np.array，EEG 数据矩阵，形状为 (通道数, 采样点数)
        fs: int，采样率
        lowcut: float，滤波器低截止频率
        highcut: float，滤波器高截止频率
        channel: int，要绘制的通道索引（从 0 开始）
        n_samples: int，要绘制的采样点数量
    """
    # 获取指定通道的原始信号（只取前 n_samples 个采样点）
    original_signal = data[channel, :n_samples]

    # 对指定通道的信号进行滤波
    filtered_signal_830 = bandpass_filter(data[channel], lowcut, highcut, fs)[:n_samples]
    filtered_signal_040 = bandpass_filter(data[channel], 0.000001, 40, fs)[:n_samples]

    plt.figure(figsize=(10, 6))
    plt.plot(original_signal, label='raw signal', color='blue')
    plt.xlabel('sampling points')
    plt.ylabel('amplitude')
    plt.title(f'channel C4 raw signal')
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(filtered_signal_040, label='filtered signal(0-40)', color='red')
    plt.xlabel('sampling points')
    plt.ylabel('amplitude')
    plt.title(f'channel C4 filtered signal(0-40Hz)')
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(filtered_signal_830, label='filtered signal(8-30)', color='black')
    plt.xlabel('sampling points')
    plt.ylabel('amplitude')
    plt.title(f'channel C4 filtered signal(8-30Hz)')
    plt.legend()
    plt.show()

#=======================================================================================================================
if __name__ == '__main__':

    edf_file_path = r'.\new_data'
    edf_list = os.listdir(edf_file_path)
    jj = []
    for edf_file in edf_list:
        edf_file_path1 = os.path.join(edf_file_path, edf_file)
        evevent,k =get_edf_data(edf_file_path1)
        jj.append(k)
    print('error result:',jj)
    root_path = 'D:\\python\\BCI_project\\BCI_RC'
    edf_file_path = os.path.join(root_path, 'new_data', 'S001')
    file_name = 'S001R01.edf'
    full_path = os.path.join(edf_file_path, file_name)
    raw = mne.io.read_raw_edf(full_path, preload=True)
    K_data = raw.get_data()

    # 调用函数绘图，绘制第一个通道前 1000 个采样点的原始和滤波后的信号
    plot_eeg_signals_C3(K_data, fs=160, lowcut=8.0, highcut=30.0, channel=8, n_samples=1000)
    plot_eeg_signals_C4(K_data, fs=160, lowcut=8.0, highcut=30.0, channel=12, n_samples=1000)
    print(raw.ch_names)


