import numpy as np
import torch
torch.set_num_threads(1)

def getOptions(opts, vars):
    opts['vad_tresh'] = 0.7

    vars['event_vad'] = False
    vars['confidence'] = 0

def getEventAddress(opts, vars):
    return 'vad@audio'

def getSampleDimensionOut(dim, opts, vars):
    vars['loaded'] = False

    try:
        load_model(opts, vars)
        vars['loaded'] = True
    except Exception as ex:
        print(ex)

    return 1

def getSampleTypeOut(type, types, opts, vars): 
    if type != types.FLOAT:  
        print('types other than float are not supported') 
        return types.UNDEF

    return type

def transform_enter(sin, sout, sxtra, board, opts, vars):
    pass

def transform(info, sin, sout, sxtras, board, opts, vars):
    if vars['loaded']:
        time_ms = round(1000 * info.time)
        audio_float32 = np.array(sin, dtype=np.float32).squeeze()

        new_confidence = vars['model'](torch.from_numpy(audio_float32), 16000).item()
        
        for n in range(sout.num):
            sout[n] = 0
        if type(new_confidence) == float:
            sout[0] = new_confidence
        
        if new_confidence > opts['vad_tresh'] and not vars['event_vad']:
            vars['event_vad'] = True
            board.update(time_ms, 0, 'vad@audio', state=board.CONTINUED)
        elif new_confidence < opts['vad_tresh'] and vars['event_vad']:
            vars['event_vad'] = False
            board.update(time_ms, 0, 'vad@audio', state=board.COMPLETED)
        

def load_model(opts, vars):
    model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
        model='silero_vad',
        force_reload=True)
    (get_speech_timestamps,
        save_audio,
        read_audio,
        VADIterator,
        collect_chunks) = utils

    vars['model'] = model
    vars['utils'] = utils
    vars['get_speech_timestamps'] = get_speech_timestamps
    vars['save_audio'] = save_audio
    vars['read_audio'] = read_audio
    vars['VADIterator'] = VADIterator
    vars['collect_chunks'] = collect_chunks

def int2float(sound):
    abs_max = np.abs(sound).max()
    sound = sound.astype('float32')
    if abs_max > 0:
        sound *= 1/32768
    sound = sound.squeeze()  # depends on the use case
    return sound