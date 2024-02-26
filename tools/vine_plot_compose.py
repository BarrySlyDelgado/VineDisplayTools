import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patch
import os
import argparse


def parse_info(log):
    task_info = {}
    worker_info = {}
    library_info = {}
    lines = open(log, 'r').read().splitlines()
    first_task = float('inf')
    last_task_done = float('-inf')
    manager_end = 0
    # parse relevant info
    for line in lines:
        try:
            (time, m_pid, category, obj, status, info) = line.split(maxsplit=5)
            time = float(time)/1000000
        except:
            continue
        if category == 'TASK':
            if obj not in task_info:
                task_info[obj] = {}
            if status == 'READY': 
                task_info[obj]['function'] = info.split()[0]
                task_info[obj]['ready'] = float(time)
                task_info[obj]['id'] = obj
            if status == 'RUNNING':
                task_info[obj]['start_time'] = float(time)
                task_info[obj]['worker'] = info.split()[0]
                if float(time) < first_task:
                    first_task = float(time)
            if status == 'WAITING_RETRIEVAL':
                task_info[obj]['stop_time'] = float(time)
            if status == 'RETRIEVED':
                task_info[obj]['retrieved'] = float(time)
            if status == 'DONE':
                task_info[obj]['done'] = float(time)
                if float(time) > last_task_done:
                    last_task_done = float(time)

        if category == 'WORKER':
            if obj not in worker_info:
                worker_info[obj] = {'tasks':[], 'libraries':[], 'cache':[]}
            if status == 'CONNECTION':
                worker_info[obj]['start_time'] = float(time)
            if status == 'DISCONNECTION':
                worker_info[obj]['stop_time'] = float(time)
            if status == 'CACHE_UPDATE':
                (filename, size, wall_time, start_time) = info.split()
                if filename.startswith('temp'):
                    worker_info[obj]['cache'].append([float(time), float(size)])

        if category == 'LIBRARY':
            if obj not in library_info:
                library_info[obj] = {}
            if status == 'STARTED':
                library_info[obj]['start_time'] = float(time)
                library_info[obj]['worker'] = info

        if category == 'MANAGER':
            if status == 'START':
                manager_start = float(time)
            if status == 'END':
                manager_end = float(time)
    
    # match transactions to workers
    for task in task_info:
        if 'worker' in task_info[task] and task_info[task]['worker'] in worker_info and 'stop_time' in task_info[task] and 'id' in task_info[task]:
            worker_info[task_info[task]['worker']]['tasks'].append(task_info[task])
    for library in library_info:
            if 'worker' in library_info[library] and library_info[library]['worker'] in worker_info:
                worker_info[library_info[library]['worker']]['libraries'].append(library_info[library])

    worker_info['manager_info'] = {'log':os.path.basename(log), 'start': manager_start, 'stop': manager_end, 'first_task':first_task, 'last_task_done':last_task_done}

    return worker_info

def plot_cache(log_info, axs, args):
    origin = log_info['manager_info']['start']
    for worker in log_info:
        if 'tasks' not in log_info[worker]:
            continue
        byte_count = 0
        xs = []
        ys = []
        for update in log_info[worker]['cache']:
            byte_count += update[1]
            xs.append(update[0] - origin)
            ys.append(byte_count/1000000000)
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
    y_count = 0
    y_counts = []
    core_count = 0
    core_counts = []
    task_plot_info = {"ys":[], "widths":[], "lefts":[]}
    lib_plot_info = {"ys":[], "widths":[], "lefts":[]}
    worker_plot_info = {"ys":[], "widths":[], "heights":[], "lefts":[]}
    first_task =  log_info['manager_info']['first_task'] 
    manager_start = log_info['manager_info']['start']
    origin = 0
    if args.worker_origin == 'first-task':
        origin = first_task
    elif args.worker_origin == 'manager-start':
        origin = manager_start
    for worker in log_info:
        # assign tasks to cores on the worker
        if 'tasks' not in log_info[worker]:
            continue
        y_count += 1
        start_y = y_count
        start_x = log_info[worker]['start_time']
        slots = {}
        tasks = [[task['start_time'], task['stop_time']] for task in log_info[worker]['tasks']]
        tasks = sorted(tasks, key=lambda x: x[0])
        for task in tasks:
            if not slots:
                slots[1] = [task]
            else:
                fits = 0
                for slot in slots:
                    if task[0] > slots[slot][-1][1]:
                        slots[slot].append(task)
                        fits += 1
                        break
                if not fits:
                    slots[len(slots) + 1] = [task]

        # add to task plot
        for slot in slots:
            core_count += 1
            y_count += 1
            for task in slots[slot]:
                task_plot_info['ys'].append(y_count)
                task_plot_info['widths'].append(task[1] - task[0])
                task_plot_info['lefts'].append(task[0] - origin)

        if 'stop_time' not in log_info[worker]:
            log_info[worker]['stop_time'] = log_info['manager_info']['last_task_done']
        # add to library plot
        for lib in log_info[worker]['libraries']:
            y_count += 1
            lib_plot_info['ys'].append(y_count)
            lib_plot_info['widths'].append(log_info[worker]['stop_time'] - lib['start_time'])
            #lib_plot_info['widths'].append(log_info['manager_info']['last_task_done'] - lib['start_time'])
            lib_plot_info['lefts'].append(lib['start_time'] - origin)
        
        # worker plot
        core_counts.append(core_count)
        y_counts.append(y_count)
        stop_y = y_count
        stop_x = log_info[worker]['stop_time']
        #stop_x = log_info['manager_info']['last_task_done']
        worker_plot_info['ys'].append(start_y)
        worker_plot_info['widths'].append(stop_x - start_x)
        worker_plot_info['heights'].append(stop_y - start_y)
        worker_plot_info['lefts'].append(start_x - origin)
    
    if(worker_plot_info['ys']):
        axs.barh(worker_plot_info['ys'], worker_plot_info['widths'], height=worker_plot_info['heights'], left=worker_plot_info['lefts'], label='workers', color='grey', align='edge')
    if(task_plot_info['ys']):
        axs.barh(task_plot_info['ys'], task_plot_info['widths'], left=task_plot_info['lefts'], label='tasks')
    if(lib_plot_info['ys']):
        axs.barh(lib_plot_info['ys'], lib_plot_info['widths'], left=lib_plot_info['lefts'], label='library tasks', color='green')

    y_axis = args.worker_y
    tick_count = args.worker_ticks
    steps = int(len(y_counts)/tick_count) 
    y_labels = []
    y_ticks = [y_counts[x] for x in range(steps - 1, len(y_counts), steps) if y_counts[steps - 1]]
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
    times = []
    task_plot_info = {'ys':[], 'lefts':[], 'widths':[]}
    retrieved_plot_info = {'ys':[], 'lefts':[], 'widths':[]}
    done_plot_info = {'ys':[], 'lefts':[], 'widths':[]}
    first_task = log_info['manager_info']['first_task']
    for worker in log_info:
        if 'tasks' in log_info[worker]:
            for task in log_info[worker]['tasks']:
                times.append([task['start_time'], task['stop_time'], task['retrieved'], task['done']])
    
    times = sorted(times, key=lambda x: x[0])
    count = 0
    for time in times:
        count += 1
        task_plot_info['ys'].append(count)
        retrieved_plot_info['ys'].append(count)
        done_plot_info['ys'].append(count)

        task_plot_info['widths'].append(time[1] - time[0])
        retrieved_plot_info['widths'].append(time[2] - time[1])
        done_plot_info['widths'].append(time[3] - time[2])

        task_plot_info['lefts'].append(time[0] - first_task)
        retrieved_plot_info['lefts'].append(time[1] - first_task)
        done_plot_info['lefts'].append(time[2] - first_task)
            
    axs.barh(task_plot_info['ys'], task_plot_info['widths'], left=task_plot_info['lefts'], label='tasks')
    axs.barh(retrieved_plot_info['ys'], retrieved_plot_info['widths'], left=retrieved_plot_info['lefts'], label='retrieved')
    axs.barh(done_plot_info['ys'], done_plot_info['widths'], left=done_plot_info['lefts'], label='done')

    if args.sublabels:
        axs.set_xlabel('Time (s)')
        axs.set_ylabel('Tasks by Start Time')
    if args.r_xlim:
        axs.set_xlim(right=args.r_xlim)
    axs.legend()
    
def plot_state(log_info, axs, args):
    times = []
    origin = log_info['manager_info']['start']
    for worker in log_info:
        if 'tasks' in log_info[worker]:
            for task in log_info[worker]['tasks']:
                times.append([task['ready'] - origin, 'ready'])
                times.append([task['start_time'] - origin, 'start'])
                times.append([task['stop_time'] - origin, 'stop'])
                times.append([task['retrieved'] - origin, 'retrieved'])
                times.append([task['done'] - origin, 'done'])
    times = sorted(times, key=lambda x: x[0])

    ready = {'count':0, 'x':[], 'y':[]}
    running = {'count':0, 'x':[], 'y':[]}
    waiting = {'count':0, 'x':[], 'y':[]}
    retrieved = {'count':0, 'x':[], 'y':[]}
    done = {'count':0, 'x':[], 'y':[]}
    for time in times:
        if time[1] == 'ready':
            ready['count'] += 1
            ready['x'].append(time[0])
            ready['y'].append(ready['count'])
        elif time[1] == 'start':
            ready['count'] -= 1
            ready['x'].append(time[0])
            ready['y'].append(ready['count'])

            running['count'] += 1
            running['x'].append(time[0])
            running['y'].append(running['count'])
        elif time[1] == 'stop':
            running['count'] -= 1
            running['x'].append(time[0])
            running['y'].append(running['count'])

            waiting['count'] += 1
            waiting['x'].append(time[0])
            waiting['y'].append(waiting['count'])
        elif time[1] == 'retrieved':
            waiting['count'] -= 1
            waiting['x'].append(time[0])
            waiting['y'].append(waiting['count'])

            retrieved['count'] += 1
            retrieved['x'].append(time[0])
            retrieved['y'].append(retrieved['count'])
        elif time[1] == 'done':
            retrieved['count'] -= 1
            retrieved['x'].append(time[0])
            retrieved['y'].append(retrieved['count'])

            done['count'] += 1
            done['x'].append(time[0])
            done['y'].append(done['count'])


    axs.plot(ready['x'], ready['y'], label='ready tasks')
    axs.plot(running['x'], running['y'], label='running tasks')
    axs.plot(waiting['x'], waiting['y'], label='waiting tasks')
    axs.plot(retrieved['x'], retrieved['y'], label='retrieved tasks')
    axs.plot(done['x'], done['y'], label='done tasks')

    if args.sublabels:
        axs.set_xlabel('Runtime (s)')
        axs.set_ylabel('Number of Tasks in Queue')
    axs.legend()

def plot_runtime(log_info, axs, args):
    times = []
    for worker in log_info:
        if 'tasks' in log_info[worker]:
            for task in log_info[worker]['tasks']:
                times.append(task['stop_time'] - task['start_time'])
    
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
    times = []
    for worker in log_info:
        if 'tasks' in log_info[worker]:
            for task in log_info[worker]['tasks']:
                times.append([task['start_time'], task['stop_time']])
    
    p_complete = []
    time_completed = []
    times = sorted(times, key=lambda x: x[1])
    total_tasks = len(times)
    for x in range(total_tasks):
        p_complete.append((x+1)/total_tasks)
        time_completed.append(times[x][1] - times[0][0])
    
    axs.plot(time_completed, p_complete, label='tasks completed')
    if args.sublabels:
        axs.set_xlabel('Time (s)')
        axs.set_ylabel('Percent of Tasks Completed')
    axs.legend()
    
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

    # Subplot options
    parser.add_argument('--sublabels', action='store_true', default=False)
    parser.add_argument('--subtitles', action='store_true', default=False)
    parser.add_argument('--sublegend', action='store_true', default=False)
    parser.add_argument('--worker-y', nargs='?', default='workers', choices=['workers', 'cores'], type=str)
    parser.add_argument('--worker-origin', nargs='?', default='first-task', choices=['first-task', 'manager-start'], type=str)
    parser.add_argument('--r-xlim', nargs='?', default=None, type=float)
    parser.add_argument('--l-xlim', nargs='?', default=None, type=float)
    parser.add_argument('--worker-ticks', nargs='?', default=5, type=int)

    # Figure Options
    parser.add_argument('--title', nargs='?', default='TaskVine Plot Composition')
    parser.add_argument('--width', nargs='?', default=6.4, type=float)
    parser.add_argument('--height', nargs='?', default=4.8, type=float)
    parser.add_argument('--scale',  action='store_true', default=False)
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
    plt.show()







        










  







        
