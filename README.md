# faas_python3

Faas model for calmodulin translated from R into python3. Currently known to work with theta0.

Based on code written by David C. Sterratt and Judy Borowski.

# To Do:
1. ~~test against other values of theta~~ - DONE
2. ~~improve run time of code (unneccesary pandas stuff?)
-- ~~Use numba or cython to compile function?
-- think harder about closing multiprocessing pools~~ - DONE
3. try with SNL

# Requirements
Pathos
Numba
Numpy
Pandas
Scipy
