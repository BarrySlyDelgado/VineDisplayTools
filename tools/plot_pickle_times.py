import matplotlib.pyplot as plt
import sys

time = 0
times = []
filename = sys.argv[1]
lines = open(filename, 'r').read().splitlines()
for line in lines:
    time += float(line)
    times.append(float(line))

print(time)
xs = []
ys = []
times = sorted(times, key=lambda x: x)
count = 0
fig, axs = plt.subplots()
for time in times:
    count +=1
    xs.append(time)
    ys.append(count/len(times))
axs.plot(xs, ys, label='pickle times')
axs.set_title(filename)
axs.set_xlabel('Pickling Times (S)')
axs.set_ylabel('Percent of Times')
axs.set_xscale('log')
plt.show()
plt.savefig('{}-plot.pdf'.format(filename))
    
