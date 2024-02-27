import sys
import matplotlib.pyplot as plt
import ast
import matplotlib.patches as patch
import os
import argparse

MICROSECONDS = 1000000
GIGABYTES = 1000000000

def parse_info(log):

    task_info = {}
    worker_info = {}
    library_info = {}
    file_info = {}
    manager_info = {'log':log, 'start':0, 'stop':0, 'first_task':float('inf'), 'last_task_done':float('-inf')}

    # parse relevant info
    lines = open(log, 'r').read().splitlines()
    for line in lines:
        try:
            (time, m_pid, category, obj, status, info) = line.split(maxsplit=5)
            time = float(time)/MICROSECONDS
        except:
            continue
        if category == 'TASK':
            if obj not in task_info:
                task_info[obj] = {}
            if status == 'READY': 
                task_info[obj]['function'] = info.split()[0]
                task_info[obj]['ready'] = time
                task_info[obj]['id'] = obj
            if status == 'RUNNING':
                task_info[obj]['running'] = time
                task_info[obj]['worker'] = info.split()[0]
                if time < manager_info['first_task']:
                    manager_info['first_task'] = time
            if status == 'WAITING_RETRIEVAL':
                task_info[obj]['waiting_retrieval'] = time
            if status == 'RETRIEVED':
                result, exit_code, resource, info = info.split(maxsplit=4) 
                info = ast.literal_eval(info)
                task_info[obj]['retrieved'] = time
            if status == 'DONE':
                task_info[obj]['done'] = time
                if time > manager_info['last_task_done']:
                    manager_info['last_task_done'] = time

        if category == 'WORKER':
            if obj not in worker_info:
                worker_info[obj] = {'tasks':[], 'libraries':[], 'cache':[]}
            if status == 'CONNECTION':
                worker_info[obj]['start'] = time
            if status == 'DISCONNECTION':
                worker_info[obj]['stop'] = time
            if status == 'CACHE_UPDATE':
                (filename, size, wall_time, start_time) = info.split()
                size = float(size)
                if filename not in file_info:
                    file_info[filename] = {"workers":{}}
                if obj not in file_info[filename]["workers"]:
                    if size != 0:
                        file_info[filename]['workers'][obj] = [time, size/GIGABYTES]
                if filename.startswith('temp'):
                    worker_info[obj]['cache'].append([time, size/GIGABYTES])

        if category == 'LIBRARY':
            if obj not in library_info:
                library_info[obj] = {}
            if status == 'STARTED':
                library_info[obj]['start'] = time
                library_info[obj]['worker'] = info

        if category == 'MANAGER':
            if status == 'START':
                manager_info['start'] = time
            if status == 'END':
                manager_info['stop'] = time
    
    # match tasks and libraries to workers

    for task in task_info:
        if 'worker' in task_info[task] and task_info[task]['worker'] in worker_info and 'ready' in task_info[task]:
            worker_info[task_info[task]['worker']]['tasks'].append(task_info[task])

    for library in library_info:
            if 'worker' in library_info[library] and library_info[library]['worker'] in worker_info:
                worker_info[library_info[library]['worker']]['libraries'].append(library_info[library])

    log_info = {}
    log_info['worker_info'] = worker_info
    log_info['manager_info'] = manager_info
    log_info['task_info'] = task_info
    log_info['library_info'] = library_info
    log_info['file_info'] = file_info

    return log_info

def plot_cache(log_info, axs, args):

    worker_info = log_info['worker_info']
    origin = 0
    first_task =  log_info['manager_info']['first_task'] 
    manager_start = log_info['manager_info']['start']
    if args.origin == 'first-task':
        origin = first_task
    elif args.origin == 'manager-start':
        origin = manager_start
    for worker in worker_info:
        if 'tasks' not in worker_info[worker]:
            continue
        gb_count = 0
        xs = []
        ys = []
        for update in worker_info[worker]['cache']:
            gb_count += update[1]
            xs.append(update[0] - origin)
            ys.append(gb_count)
        axs.plot(xs, ys, label=worker)

    if args.sublabels:
        axs.set_xlabel('Manager Runtime (s)')
        axs.set_ylabel('Disk (GB)')
    if args.subtitles:
        axs.set_title(log_info['manager_info']['log'])
    if args.r_xlim:
        axs.set_xlim(right=args.r_xlim)
    if args.l_xlim:
        axs.set_xlim(left=args.l_xlim)
    if args.sublegend:
        axs.legend()

def plot_worker(log_info, axs, args):
    
    worker_info = log_info['worker_info']

    y_count = 0
    y_counts = []
    core_count = 0
    core_counts = []

    task_running_info = {"ys":[], "widths":[], "lefts":[]}
    task_waiting_info = {"ys":[], "widths":[], "lefts":[]}
    lib_plot_info = {"ys":[], "widths":[], "lefts":[]}
    worker_plot_info = {"ys":[], "widths":[], "heights":[], "lefts":[]}

    origin = 0
    first_task =  log_info['manager_info']['first_task'] 
    manager_start = log_info['manager_info']['start']
    if args.origin == 'first-task':
        origin = first_task
    elif args.origin == 'manager-start':
        origin = manager_start

    for worker in worker_info:
        # assign tasks to cores on the worker
        if 'tasks' not in worker_info[worker]:
            continue

        # info relevant to the worker
        y_count += 1
        start_y = y_count
        start_x = worker_info[worker]['start']

        slots = {}
        
        # sort tasks by time started running
        tasks = [[task['running'], task['waiting_retrieval'], task['retrieved']] for task in worker_info[worker]['tasks']]
        tasks = sorted(tasks, key=lambda x: x[0])
        
        # way to count the number of cores/slots on a worker
        for task in tasks:
            # intitate slots
            if not slots:
                slots[1] = [task]
            # go through slots to see next available slot
            else:
                fits = 0
                for slot in slots:
                    if task[0] > slots[slot][-1][2]:
                        # task fits in slot
                        slots[slot].append(task)
                        fits += 1
                        break
                # create new slot if task does not fit
                if not fits:
                    slots[len(slots) + 1] = [task]

        # accum results for tasks
        for slot in slots:
            core_count += 1
            y_count += 1
            for task in slots[slot]:
                task_running_info['ys'].append(y_count)
                task_waiting_info['ys'].append(y_count)
                task_running_info['widths'].append(task[1] - task[0])
                task_waiting_info['widths'].append(task[2] - task[1])
                task_running_info['lefts'].append(task[0] - origin)
                task_waiting_info['lefts'].append(task[1] - origin)

        if 'stop' not in worker_info[worker]:
            worker_info[worker]['stop'] = worker_info['manager_info']['last_task_done']

        # accum results for libraries
        for lib in worker_info[worker]['libraries']:
            y_count += 1
            lib_plot_info['ys'].append(y_count)
            lib_plot_info['widths'].append(worker_info[worker]['stop'] - lib['start'])
            lib_plot_info['lefts'].append(lib['start'] - origin)
        
        # For y scales
        core_counts.append(core_count)
        y_counts.append(y_count)
        
        # accum result for worker
        stop_y = y_count
        stop_x = worker_info[worker]['stop']

        worker_plot_info['ys'].append(start_y)
        worker_plot_info['widths'].append(stop_x - start_x)
        worker_plot_info['heights'].append(stop_y - start_y)
        worker_plot_info['lefts'].append(start_x - origin)
    
    if(worker_plot_info['ys']):
        axs.barh(worker_plot_info['ys'], worker_plot_info['widths'], height=worker_plot_info['heights'], left=worker_plot_info['lefts'], label='workers', color='grey', align='edge')
    if(task_running_info['ys']):
        axs.barh(task_running_info['ys'], task_running_info['widths'], left=task_running_info['lefts'], label='tasks running', height=-.8, align='edge')
    if(task_waiting_info['ys']):
        axs.barh(task_waiting_info['ys'], task_waiting_info['widths'], left=task_waiting_info['lefts'], label='tasks waiting retrieval', height=-.8, align='edge')
    if(lib_plot_info['ys']):
        axs.barh(lib_plot_info['ys'], lib_plot_info['widths'], left=lib_plot_info['lefts'], label='library tasks', color='green', height=-.8, align='edge')
    
    # trickery for y axis ticks
    tick_count = args.worker_ticks
    steps = int(len(y_counts)/tick_count) 
    y_labels = []
    y_ticks = [y_counts[x] for x in range(steps - 1, len(y_counts), steps) if y_counts[steps - 1]]
    y_axis = args.worker_y
    if y_axis == 'workers':
        y_labels = [x for x in range(steps, len(y_counts) + 1, steps)]
    elif y_axis == 'cores':
        y_labels = [core_counts[x] for x in range(steps - 1, len(core_counts), steps) if core_counts[steps - 1]]
    
    if args.sublabels:
        axs.set_xlabel('Time (s)')
        axs.set_ylabel(y_axis)
    axs.set_yticks(y_ticks)
    axs.set_yticklabels(y_labels)
    axs.set_xlim(0)
    axs.legend()
    
def plot_task(log_info, axs, args):

    worker_info = log_info['worker_info']

    task_running_info = {'ys':[], 'lefts':[], 'widths':[]}
    task_waiting_info = {'ys':[], 'lefts':[], 'widths':[]}
    task_retrieved_info = {'ys':[], 'lefts':[], 'widths':[]}

    origin = 0
    first_task =  log_info['manager_info']['first_task'] 
    manager_start = log_info['manager_info']['start']
    if args.origin == 'first-task':
        origin = first_task
    elif args.origin == 'manager-start':
        origin = manager_start

    times = []
    for worker in worker_info:
        if 'tasks' in worker_info[worker]:
            for task in worker_info[worker]['tasks']:
                times.append([task['running'], task['waiting_retrieval'], task['retrieved'], task['done']])
    
    times = sorted(times, key=lambda x: x[0])
    count = 0
    for time in times:
        count += 1
        task_running_info['ys'].append(count)
        task_waiting_info['ys'].append(count)
        task_retrieved_info['ys'].append(count)

        task_running_info['widths'].append(time[1] - time[0])
        task_waiting_info['widths'].append(time[2] - time[1])
        task_retrieved_info['widths'].append(time[3] - time[2])

        task_running_info['lefts'].append(time[0] - origin)
        task_waiting_info['lefts'].append(time[1] - origin)
        task_retrieved_info['lefts'].append(time[2] - origin)
            
    axs.barh(task_running_info['ys'], task_running_info['widths'], left=task_running_info['lefts'], label='tasks running')
    axs.barh(task_waiting_info['ys'], task_waiting_info['widths'], left=task_waiting_info['lefts'], label='tasks waiting retrieval')
    axs.barh(task_retrieved_info['ys'], task_retrieved_info['widths'], left=task_retrieved_info['lefts'], label='tasks retrieved')

    if args.sublabels:
        axs.set_xlabel('Time (s)')
        axs.set_ylabel('Tasks by Start Time')
    if args.r_xlim:
        axs.set_xlim(right=args.r_xlim)
    axs.legend()
    
def plot_state(log_info, axs, args):

    worker_info = log_info['worker_info']
    origin = 0
    first_task =  log_info['manager_info']['first_task'] 
    manager_start = log_info['manager_info']['start']
    if args.origin == 'first-task':
        origin = first_task
    elif args.origin == 'manager-start':
        origin = manager_start

    times = []
    for worker in worker_info:
        if 'tasks' in worker_info[worker]:
            for task in worker_info[worker]['tasks']:
                times.append([task['ready'], 'ready', task['id']])
                times.append([task['running'], 'running', task['id']])
                times.append([task['waiting_retrieval'], 'waiting_retrieval', task['id']])
                times.append([task['retrieved'], 'retrieved', task['id']])
                times.append([task['done'], 'done', task['id']])
    times = sorted(times, key=lambda x: x[0])

    states = ['ready', 'running', 'waiting_retrieval', 'retrieved', 'done']
    state_info = {}
    task_last_state = {}
    for state in states:
        state_info[state] = {'count':0, 'x':[], 'y':[]}

    for event in times:
        time = event[0]
        state = event[1]
        task_id = event[2]

        print(time, state, task_id)
        state_info[state]['count'] += 1
        state_info[state]['x'].append(time - origin)
        state_info[state]['y'].append(state_info[state]['count'])
        if task_id  in task_last_state:
            last_state = task_last_state[task_id]
            state_info[last_state]['count'] -= 1
            state_info[last_state]['x'].append(time - origin)
            state_info[last_state]['y'].append(state_info[last_state]['count'])
        task_last_state[task_id] = state

    for state in state_info:
        axs.plot(state_info[state]['x'], state_info[state]['y'], label=state)

    if args.sublabels:
        axs.set_xlabel('Runtime (s)')
        axs.set_ylabel('Number of Tasks in State')    
    axs.set_xlim(0)
    axs.legend()

def plot_runtime(log_info, axs, args):    
    
    worker_info = log_info['worker_info']

    times = []
    for worker in worker_info:
        if 'tasks' in worker_info[worker]:
            for task in worker_info[worker]['tasks']:
                times.append(task['waiting_retrieval'] - task['running'])
    
    p_complete = []
    runtime = []
    times = sorted(times, key=lambda x: x)
    total_tasks = len(times)
    for x in range(total_tasks):
        p_complete.append((x+1)/total_tasks)
        runtime.append(times[x])
    
    axs.plot(runtime, p_complete, label='task runtimes')
    axs.set_xscale('log')
    if args.sublabels:
        axs.set_xlabel('Task Runtime (s)')
        axs.set_ylabel('Percent of Tasks')
    if args.subtitles:
        axs.set_title(log_info['manager_info']['log'])
    if args.r_xlim:
        axs.set_xlim(right=args.r_xlim)
    if args.l_xlim:
        axs.set_xlim(left=args.l_xlim)

    axs.legend()
    
def plot_completion(log_info, axs, args):

    origin = 0
    first_task =  log_info['manager_info']['first_task'] 
    manager_start = log_info['manager_info']['start']
    if args.origin == 'first-task':
        origin = first_task
    elif args.origin == 'manager-start':
        origin = manager_start

    worker_info = log_info['worker_info']

    times = []
    for worker in worker_info:
        if 'tasks' in worker_info[worker]:
            for task in worker_info[worker]['tasks']:
                times.append([task['running'], task['waiting_retrieval']])
    
    p_complete = []
    time_completed = []
    times = sorted(times, key=lambda x: x[1])
    total_tasks = len(times)
    for x in range(total_tasks):
        p_complete.append((x+1)/total_tasks)
        time_completed.append(times[x][1] - origin)
    
    axs.plot(time_completed, p_complete, label='tasks completed')
    if args.sublabels:
        axs.set_xlabel('Time (s)')
        axs.set_ylabel('Percent of Tasks Completed')
    axs.legend()

def plot_file_accum(log_info, axs, args):

    origin = 0
    first_task =  log_info['manager_info']['first_task'] 
    manager_start = log_info['manager_info']['start']
    if args.origin == 'first-task':
        origin = first_task
    elif args.origin == 'manager-start':
        origin = manager_start

    file_info = log_info['file_info']
    for filename in file_info:
        file_accum = []
        for worker in file_info[filename]["workers"]:
            time = file_info[filename]["workers"][worker][0]
            file_accum.append([worker, time])

        file_accum = sorted(file_accum, key=lambda x:x[1])
        xs = []
        ys = []
        count = 0
        for accum in file_accum:
            count += 1
            xs.append(accum[1] - origin)
            ys.append(int(count))
        axs.plot(xs, ys, label=filename)

    if args.sublabels:
        axs.set_xlabel('Manager Lifetime (s)')
        axs.set_ylabel('Individual File Repilication')

def plot_file_hist(log_info, axs, args):

    origin = 0
    first_task =  log_info['manager_info']['first_task'] 
    manager_start = log_info['manager_info']['start']
    if args.origin == 'first-task':
        origin = first_task
    elif args.origin == 'manager-start':
        origin = manager_start
    
    hist = {}
    file_info = log_info['file_info']
    for filename in file_info:
        count = 0
        max_size = float('-inf')
        min_size = float('inf')
        for worker in file_info[filename]['workers']:
            time = file_info[filename]["workers"][worker][0]
            size = file_info[filename]["workers"][worker][1]
            count +=1
            if size > max_size:
                max_size = size
            if size < min_size:
                min_size = size
        if count not in hist:
            hist[count] = [1, max_size*count, [filename]]
        else:
            hist[count][0] += 1
            hist[count][1] += max_size*count
            hist[count][2].append(filename)
    xs = []
    ys = []
    ys2 = []
    axs2 = axs.twinx()
    for count in hist:
        xs.append(count)
        ys.append(hist[count][0])
        ys2.append(hist[count][1])

    axs.bar(xs, ys, label='file replication count', width=-.2, align='edge')
    axs2.bar(xs, ys2, label='distributed disk usage', width=.2, align='edge', color='orange')
    if args.sublabels:
        axs.set_xlabel('File Replication Count')
        axs.set_ylabel('Number of Files')
        axs2.set_ylabel('Distributed Disk Usage (GB)')
    if args.sublegend: 
        axs.legend(loc='upper left')
        axs2.legend(loc='upper right')


def plot_any(log_info, plot, axs, args):

    if plot == 'worker-view':
        plot_worker(log_info, axs, args)
    elif plot == 'worker-cache':
        plot_cache(log_info, axs, args)
    elif plot == 'task-view':
        plot_task(log_info, axs, args)
    elif plot == 'task-runtime':
        plot_runtime(log_info, axs, args)
    elif plot == 'task-completion':
        plot_completion(log_info, axs, args)
    elif plot == 'task-state':
        plot_state(log_info, axs, args)
    elif plot == 'file-accum':
        plot_file_accum(log_info, axs, args)
    elif plot == 'file-hist':
        plot_file_hist(log_info, axs, args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Consolidated plotting tool with a variety of options')
    parser.add_argument('logs', nargs='+')

    # Plotting views
    parser.add_argument('--worker-view', dest='plots', action='append_const', const='worker-view')
    parser.add_argument('--task-view', dest='plots', action='append_const', const='task-view')
    parser.add_argument('--task-runtime', dest='plots', action='append_const', const='task-runtime')
    parser.add_argument('--task-completion', dest='plots', action='append_const', const='task-completion')
    parser.add_argument('--task-state', dest='plots', action='append_const', const='task-state')
    parser.add_argument('--worker-cache', dest='plots', action='append_const', const='worker-cache')
    parser.add_argument('--file-accum', dest='plots', action='append_const', const='file-accum')
    parser.add_argument('--file-hist', dest='plots', action='append_const', const='file-hist')

    # Subplot options
    parser.add_argument('--sublabels', action='store_true', default=False)
    parser.add_argument('--subtitles', action='store_true', default=False)
    parser.add_argument('--sublegend', action='store_true', default=False)
    parser.add_argument('--worker-y', nargs='?', default='workers', choices=['workers', 'cores'], type=str)
    parser.add_argument('--origin', nargs='?', default='first-task', choices=['first-task', 'manager-start'], type=str)
    parser.add_argument('--r-xlim', nargs='?', default=None, type=float)
    parser.add_argument('--l-xlim', nargs='?', default=None, type=float)
    parser.add_argument('--worker-ticks', nargs='?', default=5, type=int)

    # Figure Options
    parser.add_argument('--title', nargs='?', default='TaskVine Plot Composition')
    parser.add_argument('--width', nargs='?', default=6.4, type=float)
    parser.add_argument('--height', nargs='?', default=4.8, type=float)
    parser.add_argument('--scale',  action='store_true', default=False)
    parser.add_argument('--show',  action='store_true', default=False)
    parser.add_argument('--out', nargs='?', default=None)

    # Other Options
    parser.add_argument('mode', nargs='?', default='tv', choices=['tv', 'wq'], type=str)

    args = parser.parse_args()
    
    ncols = len(args.plots)
    nrows = len(args.logs)

    width = args.width
    height = args.height
    if args.scale:
        width = 6.4 * ncols
        height = 4.8 * nrows
    
    fig, axs = plt.subplots(nrows, ncols, squeeze=False)

    row = 0
    for log in args.logs:
        print('Parsing log for {}'.format(log))
        log_info = parse_info(log)
        print('Ploting plot(s) for {}'.format(log))
        col = 0
        for plot in args.plots:
            plot_any(log_info, plot, axs[row][col], args)
            col += 1
        row += 1
    
    fig.suptitle(args.title)
    fig.set_size_inches(w=width, h=height)
    if args.out:
        fig.savefig(args.out)
    if args.show:
        plt.show()
