import numpy as np

def sp2wt(x,s):
    a = 100*x*s
    b = np.diag(x@s.T).reshape(-1,1)
    b = np.clip(b,1e-8,np.inf)
    return a/b
