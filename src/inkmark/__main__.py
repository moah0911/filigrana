"""`python -m inkmark` runs the writer CLI.

A console script already exists; this shim makes the module invocation work too, because
docs and scripts written against a checkout should not have to know whether the project was
installed.
"""

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
