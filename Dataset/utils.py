import numpy as np

def caliColor(sample, whiteRef):
        #return div0(sample, self.imgWhiteRef)
        retV = np.zeros(sample.shape)
        for i in range(0, sample.shape[1]):
            retV[:,i,:] = div0(sample[:,i,:],whiteRef)
        return retV
    
def div0( a, b ):
    """ ignore / 0, div0( [-1, 0, 1], 0 ) -> [0, 0, 0] """
    with np.errstate(divide='ignore', invalid='ignore'):
        c = np.true_divide( a, b )
        c[ ~ np.isfinite( c )] = 0  # -inf inf NaN
    return c