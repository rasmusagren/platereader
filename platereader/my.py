import numpy as np
import os
import sys

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

def cat(A,B,dim=0):
    """ The purpose of this function is to expand on the functionality of the numpy
        concatenate function to allow for concatenation of one dimensional arrays and
        None values.
    """
    
    if A is None:
        return B
    
    if B is None:
        return A
        
    if len(A)==0:
        return B
    
    if len(B)==0:
        return A
        
    sA=A.shape
    sB=B.shape
    
    if len(sA)>1 and len(sB)>1:
        return np.concatenate((A,B),dim)
        
    #If both of them are one dimensional
    if len(sA)==1 and len(sB)==1:
        #Dim doesn't matter in this case
        return np.concatenate((A,B),dim)
        
    #If one of them is one dimensional
    if len(sA)==1:
        if dim==0:
            A=A[None,:]
        else:
            A=A[:,None]
    
    if len(sB)==1:
        if dim==0:
            B=B[None,:]
        else:
            B=B[:,None]
            
    return np.concatenate((A,B),dim)