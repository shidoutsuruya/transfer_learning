import fractions
import numpy as np
def fraction_print(x):
    return str(fractions.Fraction(x).limit_denominator())
    
np.set_printoptions(formatter={'all':fraction_print}) 
a=np.array([[-4,1,0],[0,-3,1],[0,0,-2]])
print(np.linalg.eig(a)[1])
