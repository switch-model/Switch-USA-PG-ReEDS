"""
Let user interact with model in a standard REPL _before_ solving.
"""

import code
import sys


def pre_solve(m):
    old_hook = sys.excepthook
    sys.excepthook = sys.__excepthook__  # disable old hook to avoid premature exit
    banner = (
        "Entering interactive shell. Constructed model is in `m` variable. "
        "Type ctrl-d to exit shell and continue solving model or exit() to exit Python entirely."
    )
    code.interact(
        banner=banner, local=dict(list(globals().items()) + list(locals().items()))
    )
    sys.excepthook = old_hook
