from matplotlib import pyplot as plt
import sys

all_data = {}

def readData():
    file_path = sys.argv[1]
    omit_lines = 4
    with open(file_path) as f:
        while omit_lines:
            f.readline()
            omit_lines -= 1
        
        for per_line in f.readlines():
            sta_idx = per_line.find(']')
            if sta_idx!=-1:
                per_line = per_line[sta_idx+2:]
            try:
                data_type, datas = per_line.split(':')
            except:
                print("read line error, pass")
                continue
            datas = datas.split()
            data_len = len(datas)
            if data_type in all_data:
                for i in range(data_len):
                    all_data[data_type][i].append(float(datas[i]))
            else:
                all_data[data_type] = []
                for i in range(data_len):
                    all_data[data_type].append([float(datas[i])])

def plotData(data_types):
    for da_ty in data_types:
        if da_ty in all_data:
            for i in range(1, len(all_data[da_ty])):
                plt.plot(all_data[da_ty][0], all_data[da_ty][i])
    
    plt.show()

def plot3DP():
    fig = plt.figure()
    ax = fig.add_subplot(211, projection='3d')
    if 'predict_P' in all_data:
        predict_pos = all_data['predict_P']
        ax.plot(predict_pos[1], predict_pos[2], predict_pos[3])
    else:
        print('no predict_P')

    if 'correct_P' in all_data:
        predict_pos = all_data['correct_P']
        ax.plot(predict_pos[1], predict_pos[2], predict_pos[3])
    else:
        print('no correct_P')
    ax1 = fig.add_subplot(221)
    if 'process_update' in all_data:
        process_update_data  = all_data['process_update']
        ax1.plot(process_update_data[0], process_update_data[1])
    plt.show()

def plotUpdate():
    fig = plt.figure()
    ax_x = fig.add_subplot(411)
    ax_y = fig.add_subplot(412)
    ax_z = fig.add_subplot(413)
    if 'predict_P' in all_data:
        predict_pos = all_data['predict_P']
        ax_x.plot(predict_pos[0], predict_pos[1], label='pre-p-x')
        ax_y.plot(predict_pos[0], predict_pos[2], label='pre-p-y')
        ax_z.plot(predict_pos[0], predict_pos[3], label='pre-p-z')
    else:
        print('no predict_P')

    if 'correct_P' in all_data:
        predict_pos = all_data['correct_P']
        ax_x.plot(predict_pos[0], predict_pos[1], label='cor-p-x')
        ax_y.plot(predict_pos[0], predict_pos[2], label='cor-p-y')
        ax_z.plot(predict_pos[0], predict_pos[3], label='cor-p-z')
    else:
        print('no correct_P')

    if 'delta_P' in all_data:
        predict_pos = all_data['delta_P']
        ax_x.plot(predict_pos[0], predict_pos[1], label='dlt-p-x')
        ax_y.plot(predict_pos[0], predict_pos[2], label='dlt-p-y')
        ax_z.plot(predict_pos[0], predict_pos[3], label='dlt-p-z')
    else:
        print('no delta_P')

    ax_up = fig.add_subplot(414)
    if 'process_update' in all_data:
        process_update_data  = all_data['process_update']
        ax_up.plot(process_update_data[0], process_update_data[1])
    ax_x.legend()
    ax_y.legend()
    ax_z.legend()
    plt.show()

def plotMag():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    if 'external_mag' in all_data:
        process_data  = all_data['external_mag']
        sum_x = 0.0
        sum_y = 0.0
        sum_z = 0.0
        data_size = len(process_data[0])
        raw_x = []
        raw_y = []
        raw_z = []
        for i in range(data_size):
            raw_x.append(process_data[1][i]/0.977 - 0.024)
            raw_y.append(process_data[2][i]/0.993 - 0.032)
            raw_z.append(process_data[3][i]/1.041 - 0.019)
        # sum_x /= data_size
        # sum_y /= data_size
        # sum_z /= data_size
        # ax.scatter(process_data[1], process_data[2], process_data[3], cmap='Blues')
        ax.scatter(raw_x, raw_y, raw_z, cmap='Blues')
        # ax.scatter(sum_x, sum_y, sum_z, marker='p')
    plt.show()
    

def main():
    readData()
    data_types = ['predict_P', 'correct_P']
    # plotData(data_types)
    # plot3DP()
    # plotUpdate()
    plotMag()


if __name__ == "__main__":
    main()