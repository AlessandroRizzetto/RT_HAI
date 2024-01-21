def getOptions(opts, vars):
    vars['active'] = False

def transform(info, sin, sout, sxtras, board, opts, vars):
    for n in range(sin.num):
        for d in range(sin.dim):
            sout[n,d] = sin[n,d] if vars['active'] else 0

def listen_enter(opts, vars):
    pass

def update(event, board, opts, vars):
    if event.glue == board.CONTINUED:
        vars['active'] = True
    elif event.glue == board.COMPLETED:
        vars['active'] = False

def listen_flush(opts, vars):
    pass