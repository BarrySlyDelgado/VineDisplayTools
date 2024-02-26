import sys
import matplotlib.pyplot as plt
import hashlib
import math
import subprocess
from os import listdir
from os.path import isfile, join


totals = []
numss = []
ls = sys.argv[1:]
for dir_name in ls:
    config = ''
    tr_logs = []
    # getting transaction logs from directory
    files = [f for f in listdir(dir_name) if isfile(join(dir_name, f))]
    for f in files:
        if f.startswith('tr'):
            tr_logs.append(join(dir_name, f))
        if f.startswith('config'):
            config = join(dir_name, f)

    #print('config for this run:\n\n',open(config, 'r').read())


    global_task_info = {}
    count = 0
    for log in tr_logs:
        count += 1
        local_task_info = {}    
        print('Running vine_plot_txn_log on log {}'.format(count))
        #subprocess.run(['./vine_plot_txn_log', log, '{}/txn_plot_{}'.format(dir_name, count)])
        lines = open(log, 'r').read().splitlines()
        for line in lines:
            (time, m_pid, category, obj, status, info) = line.split(maxsplit=5)
            if category == 'TASK':
                g_obj = obj + '_' + str(count)
                if obj not in local_task_info:
                    local_task_info[obj] = {}
                    global_task_info[g_obj] = {}
                if status == 'READY': 
                    local_task_info[obj]['function'] = info.split()[0]
                    global_task_info[g_obj]['function'] = info.split()[0]
                if status == 'RUNNING':
                    local_task_info[obj]['start_time'] = float(time)/1000000
                    global_task_info[g_obj]['start_time'] = float(time)/1000000
                if status == 'WAITING_RETRIEVAL':
                    local_task_info[obj]['stop_time'] = float(time)/1000000
                    global_task_info[g_obj]['stop_time'] = float(time)/1000000


        total = []
        for x in local_task_info:
            try:
                time = local_task_info[x]['stop_time'] - local_task_info[x]['start_time']
            except:
                continue
            total.append(time)
        print('Analysis for run {}\n############################################################\n'.format(count))
        print('Total Execution Time', sum(total))
        print('Average Execution Time', sum(total)/len(total))
        print('############################################################')

    total = []
    for x in global_task_info:
        try:
            time = global_task_info[x]['stop_time'] - global_task_info[x]['start_time']
        except:
            continue
        total.append(time)
    print('Analysis for all runs:\n############################################################\n')
    print('Total Execution Time', sum(total))
    print('Average Execution Time', sum(total)/len(total))
    print('\n############################################################\n')


    total.sort()
    total = [x for x in total]
    nums = [x for x in range(len(total))]
    totals.append(total)
    numss.append(nums)


fig_source = "comparison runs"
source_hash = hashlib.md5(fig_source.encode())
for x in range(len(numss)):
    plt.plot(numss[x], totals[x], label=ls[x])
plt.title(fig_source)
plt.xlabel('sorted task numer by time')
plt.ylabel('time (s)')
plt.yscale('log')
plt.legend()
plt.savefig(source_hash.hexdigest())
print(source_hash.hexdigest())

#plt.show()








