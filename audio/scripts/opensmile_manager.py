import numpy as np
import opensmile

def getOptions(opts, vars):
    opts['feature'] = ''

    vars['smile'] = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )

def transform(info, sin, sout, sxtras, board, opts, vars):   
    npin = np.asarray(sin)
    npin = npin.reshape(-1) if npin.shape[1] == 1 else npin

    features = vars['smile'].process_signal(
        npin,
        sin.sr
    ).fillna(0.0)
    
    sout[0] = features[opts['feature']].iloc[0] if opts['feature'] in features else 0.0