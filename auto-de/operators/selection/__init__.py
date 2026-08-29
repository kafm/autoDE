from core import SelectionContext, SelectionCallback

def elitist()->SelectionCallback:
    def callback(ctx: SelectionContext)->bool:
        return ctx.trial.fit < ctx.individual.fit
    return callback