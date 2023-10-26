
import os
import sys
from MediaPipe import run,getData
from multiprocessing import Process
sys.path.append(os.environ.get('PYTHONPATH', ''))

mediapipe_process = Process(target=run)

def getOptions(opts, vars):

    pass


def getChannelNames(opts, vars):

    return {'mp': 'A body tracking signal'}


def initChannel(name, channel, types, opts, vars):

    if name == 'mp':
        channel.dim = 1
        channel.type = types.FLOAT
        channel.sr = 360
    else:
        print('unkown channel name')


def connect(opts, vars):
    print("Ciao")
    mediapipe_process.start()
    print("Ciaone")
    vars['mp'] = 0
    vars['pos'] = 0


def read(name, sout, reset, board, opts, vars):

    pos = vars['pos']
    mp = vars['mp']

    try:
        if name == 'mp':
            # read the last line of landmarks.stream
            for n in range(sout.num):
                #f = open("./landmarks.stream", "r")
                sout[n] = getData() #float(f.readlines()[-1])
                #print(sout[n])
                #f.close()

        else:
            print('unkown channel name')
    except Exception as error:
        print(error)

    vars['pos'] = pos


def disconnect(opts, vars):
    mediapipe_process.kill()
    pass
