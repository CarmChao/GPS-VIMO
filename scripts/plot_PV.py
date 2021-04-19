from matplotlib import pyplot as plt
import sys
import math

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
    ax1 = fig.add_subplot(212)
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
    
def calculateExMag():
    if 'internal_mag' in all_data:
        process_data  = all_data['internal_mag'][:200]
        sum_x = 0.0
        sum_y = 0.0
        sum_z = 0.0
        data_size = len(process_data[0])
        raw_x = []
        raw_y = []
        raw_z = []
        for i in range(data_size):
            sum_x += process_data[1][i]
            sum_y += process_data[2][i]
            sum_z += process_data[3][i]
        ave_x = sum_x/data_size
        ave_y = sum_y/data_size
        theta = math.atan2(ave_y, ave_x)
        print(theta)
        # sum_x /= data_size

def plotYaw():
    fig = plt.figure()
    ax = fig.add_subplot(211)
    ax_pos = fig.add_subplot(212)

    if 'measure_yaw' in all_data:
        print("measure_yaw")

        process_data  = all_data['measure_yaw']
        data_size = len(process_data[0])
        ax.plot(process_data[0], process_data[1], label='measure')
        ax.plot(process_data[0], process_data[2], label='predict')
        # ax.plot(process_data[0], process_data[5], label='roll')
        # ax.plot(process_data[0], process_data[4], label='pitch')
        ax.plot(process_data[0], process_data[3], label='m_o')

    if 'measure_gav' in all_data:

        gav_data = all_data['measure_gav']
        ax.plot(gav_data[0], gav_data[1], label='gav_x')
        ax.plot(gav_data[0], gav_data[2], label='gav_y')
        ax.plot(gav_data[0], gav_data[3], label='gav_z') 

    if 'pos_cov' in all_data:
        fast_data = all_data['pos_cov']
        ax_pos.plot(fast_data[0], fast_data[1], label='x')
        ax_pos.plot(fast_data[0], fast_data[2], label='y')
        ax_pos.plot(fast_data[0], fast_data[3], label='z')
    ax.legend()
    ax_pos.legend()
    plt.show()

def plotGPS():
    fig = plt.figure()
    ax = fig.add_subplot(211)
    ax_pos = fig.add_subplot(212)

    if 'gps' in all_data:
        # print("measure_yaw")

        process_data  = all_data['gps']
        data_size = len(process_data[0])
        ax.plot(process_data[0], process_data[1], label='nums')
        ax.plot(process_data[0], process_data[3], label='eph')
        # ax.plot(process_data[0], process_data[5], label='roll')
        # ax.plot(process_data[0], process_data[4], label='pitch')
        ax.plot(process_data[0], process_data[4], label='cov_v')

    if 'gps_pos' in all_data:
        print("found gps_pos!")
        correct_data = all_data['correct_P']
        process_data  = all_data['gps_pos']
        data_size = len(process_data[0])
        for i in range(len(process_data[0])):
            process_data[0][i] = float(process_data[0][i])/1e6
        ax_pos.plot(process_data[0], process_data[1], label='x')
        ax_pos.plot(process_data[0], process_data[2], label='y')
        ax_pos.plot(correct_data[0], correct_data[1], label='ex')
        ax_pos.plot(correct_data[0], correct_data[2], label='ey')
        # ax.plot(process_data[0], process_data[5], label='roll')
        # ax.plot(process_data[0], process_data[4], label='pitch')
        # ax.plot(process_data[0], process_data[4], label='cov_v')
    if "gps_r" in all_data:
        gps_r = all_data['gps_r']
        for i in range(len(gps_r[0])):
            gps_r[0][i] = float(gps_r[0][i])/1e6
        
        ax_pos.plot(gps_r[0], gps_r[1], label='gps_rx')
        ax_pos.plot(gps_r[0], gps_r[2], label='gps_ry')
    
    if "gps_update" in all_data:
        gps_r = all_data['gps_update']
        for i in range(len(gps_r[0])):
            gps_r[0][i] = float(gps_r[0][i])/1e6
        
        ax_pos.plot(gps_r[0], gps_r[1], label='gps_update', marker='o')
        # ax_pos.plot(gps_r[1], gps_r[2], label='gps_ry')
    ax.legend()
    ax_pos.legend()
    plt.show()

def main():
    readData()
    data_types = ['predict_P', 'correct_P']
    # plotData(data_types)
    # plot3DP()
    # plotUpdate()
    # plotMag()
    # calculateExMag()
    # plotYaw()
    plotGPS()


if __name__ == "__main__":
    main()