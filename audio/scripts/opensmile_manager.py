import numpy as np
import opensmile

def getOptions(opts, vars):
    '''
    SSI function called to get the options of the component and initialize the variables.

    Args:
        opts (dictionary<string,any>): options of the component
        vars (dictionary<string,any>): internal variables of the component
    '''
    
    opts['feature'] = '' # Feature to extract

    vars['smile'] = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    ) # openSMILE features extractor

def transform(info, sin, sout, sxtras, board, opts, vars):   
    '''
    SSI function called to transform the input samples into the output samples.
    Extract the specified openSMILE feature from the given input signal.

    Args:
        info (ssipyinfo): information about the input samples
        sin (ssipystream): input samples
        sout (ssipystream): output samples
        sextras (tuple<ssipystream>): extra streams of samples
        board (ssipyeventboard): event board to send events
        opts (dictionary<string,any>): options of the component
        vars (dictionary<string,any>): internal variables of the component
    '''

    npin = np.asarray(sin)
    npin = npin.reshape(-1) if npin.shape[1] == 1 else npin

    features = vars['smile'].process_signal(
        npin,
        sin.sr
    ).fillna(0.0)
    
    sout[0] = features[opts['feature']].iloc[0] if opts['feature'] in features else 0.0