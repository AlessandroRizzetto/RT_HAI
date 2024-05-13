def getOptions(opts, vars):
    '''
    SSI function called to get the options of the component and initialize the variables.

    Args:
        opts (dictionary<string,any>): options of the component
        vars (dictionary<string,any>): internal variables of the component
    '''

    vars['active'] = False

def transform(info, sin, sout, sxtras, board, opts, vars):
    '''
    SSI function called to transform the input samples into the output samples.
    Filter audio signal based on VAD, puting 0s when it is disabled (i.e., doesn't detect any voice).

    Args:
        info (ssipyinfo): information about the input samples
        sin (ssipystream): input samples
        sout (ssipystream): output samples
        sextras (tuple<ssipystream>): extra streams of samples
        board (ssipyeventboard): event board to send events
        opts (dictionary<string,any>): options of the component
        vars (dictionary<string,any>): internal variables of the component
    '''
    
    for n in range(sin.num):
        for d in range(sin.dim):
            sout[n,d] = sin[n,d] if vars['active'] else 0

def update(event, board, opts, vars):
    '''
    SSI function called when an event is received.
    If receive VAD starting event allow the signal go through the filter.

    Args:
        event (ssipyevent): event received
        board (ssipyeventboard): event board to send events
        opts (dictionary<string,any>): options of the component
        vars (dictionary<string,any>): internal variables of the component
    '''

    if event.glue == board.CONTINUED:
        vars['active'] = True
    elif event.glue == board.COMPLETED:
        vars['active'] = False