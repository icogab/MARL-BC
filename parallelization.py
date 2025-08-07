from multiprocessing import Process
from subprocess import Popen
from time import sleep
import logging

WAIT = 0.5

# setup logging
FORMAT = '%(asctime)-15s %(message)s'

def do_job(cmd, outfile=None):
    if outfile is None:
        p = Popen(cmd).wait()
    else:
        with open(outfile, 'w') as f:
            p = Popen(cmd, stdout=f).communicate()
                        

def main(joblist, N_concurrent, t_log=True):
    if t_log:
        logging.basicConfig(format=FORMAT, filename='zombie_multi_serial.log', level=logging.DEBUG)
        logging.info('Running {} total jobs, {} at a time'.format(len(joblist), N_concurrent))
    thread_list = [None]*N_concurrent
    job_id_list = [0 for i in range(N_concurrent)]
  
    i = 0
    while True:
        # find a vacant slot
        for j in range(len(thread_list)):
            if thread_list[j] is None and i<len(joblist):
                if 'outfile' in joblist[i]:
                    thread_list[j] = Process(target=do_job, args=(joblist[i]['cmd'], joblist[i]['outfile']))
                else:
                    thread_list[j] = Process(target=do_job, args=(joblist[i]['cmd']))
                thread_list[j].start()
                job_id_list[j] = i
                if t_log:
        		        logging.info('Job {} started: {}'.format(i, ' '.join(joblist[i]['cmd'])))
                i+=1
        sleep(WAIT)
        if all(obj is None for obj in thread_list):
            # If all slots are vacant, then all tasks are complete
            break
        for j in range(len(thread_list)):
            if thread_list[j] is not None:
                if not thread_list[j].is_alive():
                    if t_log:
                        logging.info('Job {} ended'.format(job_id_list[j]))
                        thread_list[j] = None

def example():
    with open('do_work.py', 'w') as f:
        f.write('''
import sys
import random

data = random.sample(range(40000), 20000)

for i in range(int(sys.argv[1])):
    sys.stdout.write('iter {}'.format(i))
    sys.stdout.flush()
    for di in data:
        for dj in data:
            di*dj

sys.stdout.write('Done!')
''')

    jobs = \
        [
            {
                'cmd':['python', 'do_work.py', '22'],
                'outfile':'job_0.out'
            },
            {
                'cmd':['python', 'do_work.py', '25'],
                'outfile':'job_1.out'
            },
            {
                'cmd':['python', 'do_work.py', '21'],
                'outfile':'job_2.out'
            },
            {
                'cmd':['python', 'do_work.py', '20'],
                'outfile':'job_3.out'
            },
        ]

    main(jobs, 10)
