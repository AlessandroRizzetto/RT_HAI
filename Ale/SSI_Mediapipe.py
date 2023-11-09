
import os
import sys
sys.path.append(os.environ.get('PYTHONPATH', ''))


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
    vars['mp'] = 0
    vars['pos'] = 0


def read(name, sout, reset, board, opts, vars):

    pos = vars['pos']
    mp = vars['mp']

    if name == 'mp':
        # read the last line of landmarks.stream
        for n in range(sout.num):
            f = open("landmarks.stream~", "r")
            sout[n] = float(f.readlines()[-1])
            f.close()

    else:
        print('unkown channel name')

    vars['pos'] = pos


def disconnect(opts, vars):
    pass
