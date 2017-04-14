import numpy as np
import matplotlib.pyplot as plt
import argparse, seaborn, pdb

parser = argparse.ArgumentParser(description='plot statisitc')
parser.add_argument('--log_path', type=str, required=True,
                    help='plot the log file')
args = parser.parse_args() 
f = open(args.log_path, 'r')
data = f.read()
f.close()
print "Load %s file" % args.log_path
data = data.split('\n')
data.pop(-1)
r_list = []
l_list = []
for d in data:
    r = float(d.split(' ')[6].replace(',', ''))
    l = float(d.split(' ')[9])
    r_list.append(r)
    l_list.append(l)

plt.plot(np.arange(len(data)), r_list, 'r')
plt.plot(np.arange(len(data)), l_list, 'b')
plt.legend(['episode reward', 'episode length'], loc='upper left')
plt.title('**Statistic**')
plt.savefig(args.log_path+'.png')
plt.show()


