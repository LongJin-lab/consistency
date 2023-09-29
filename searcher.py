import os
import numpy as np
import argparse
import os
import sys
import time
from datetime import datetime
import random
import errno
from random import randint

from sympy import sec 

GPUS = [0]

def gpu_info(GpuNum):
    gpu_status = os.popen('nvidia-smi -i '+str(GpuNum)+' | grep %').read().split('|')
    # print('gpu_status', gpu_status)
    gpu_memory = int(gpu_status[2].split('/')[0].split('M')[0].strip())
    gpu_power = int(gpu_status[1].split(
        '   ')[-1].split('/')[0].split('W')[0].strip())
    return gpu_power, gpu_memory

def SearchAndExe(Gpus, cmd, interval):
    prefix = 'CUDA_VISIBLE_DEVICES='
    foundGPU = 0
    while foundGPU==0: # set waiting condition

        for u in Gpus: 
            gpu_power, gpu_memory = gpu_info(u)
            cnt = 0 
            first = 0
            second = 0 
            empty = 1
            print('gpu, gpu_power, gpu_memory, cnt', u, gpu_power, gpu_memory, cnt)
            for i in range(10):
                gpu_power, gpu_memory = gpu_info(u) 
                print('gpu, gpu_power, gpu_memory, cnt', u, gpu_power, gpu_memory, cnt)
                if gpu_memory > 2000 or gpu_power > 100: # running
                    empty = 0
                time.sleep(interval)
            if empty == 1:
                foundGPU = 1
                break
            
    if foundGPU == 1:
        prefix += str(u)
        cmd = prefix + ' '+ cmd
        print('\n' + cmd)
        os.system(cmd)


cmd = 'nohup ./r1.sh &'
    
SearchAndExe(GPUS, cmd, interval=3)