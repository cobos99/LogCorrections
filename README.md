# LogCorrections

Code repository for the Logarithmic corrections project codebase

## Structure

1 folder per participant in the project. Commit your things there.

Please use scripts (.py) for files containing functions that other people may be interested in using and keep them clean and readable

Use Jupyter/Scripts for specific calculations (Using the more general functions in the .py scripts) that are relevant for the project.

Notice how I can import things from inside a file in cobos folder using
```Python
# -- Header (For changing the path to the parent folder so that there are no problems with relative imports)
import os
os.chdir(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))
# --
import sunny.shannon as ashannon
```