# LogCorrections

Code repository for the Logarithmic corrections project codebase

## Structure

1 folder per participant in the project. Commit your things there.

Please use scripts (.py) for files containing functions that other people may be interested in using and keep them clean and readable

Use Jupyter/Scripts for specific calculations (Using the more general functions in the .py scripts) that are relevant for the project.

Notice how I can import things made by Sunny from inside a file in cobos folder (Eg: cobos/XX analytical-numerical comparison) using
```Python
# -- Header (For changing the path to the parent folder so that 
# -- there are no problems with relative imports)
# -- Use this only for notebooks scripts should at most
# -- import from subfolders
import os
cwd = os.path.normpath(os.getcwd())
cwd = cwd.split(os.sep)
find = cwd.index("LogCorrections")
newd = f"{os.sep}".join(cwd[:find+1])
os.chdir(newd)
# --
import sunny.shannon as ashannon
```