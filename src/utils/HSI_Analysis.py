import numpy as np
import cv2

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

def getNDVIHeatmap(img, paraWv2Pst):
    pst640 = getPstFromWv(680, paraWv2Pst)
    pst800 = getPstFromWv(840, paraWv2Pst)
    img800 = img[:,:,pst800]
    img640 = img[:,:,pst640]
    ndvi = div0((img800-img640), (img800+img640))
    #ndvi = (np.float32(img[:, :, pst800] - img[:, :, pst640])) / (np.float32(img[:, :, pst800] + img[:, :, pst640]))
    return ndvi

def segmentation(img, ndvi, thresh):
    ImgWidth = img.shape[2]
    mask = ndvi
    mask[mask<=thresh] = 0
    mask[mask>thresh] = 1
    return img * (np.dstack([mask]*ImgWidth))

def getMask(img, ndvi, thresh):
    ImgWidth = img.shape[2]
    mask = ndvi
    mask[mask<=thresh] = False
    mask[mask>thresh] = True
    return np.array(mask, dtype=bool)

def removeRightRegion(hyperImg):
    _, w, _ = hyperImg.shape
    n = 0
    for i in range(w):
        if np.max(hyperImg[:,w-i-1,:]) > 0:
            n = i
            break            
    return hyperImg[:,0:w-n, :]

def getMeanSpecAlongLeaf(img):
    img[img == 0] = np.nan
    means = np.nanmean(img, axis=0)
    #print ('getMeanSpecLine', means.shape)
    return means

def getMeanSpecWholeLeaf(specs):
    #print ('specs.shape', specs.shape)
    specs[specs == 0] = np.nan
    return np.nanmean(specs, axis = 0)

def getPstFromWv(wv, paraWv2Pst):
    return np.uint16(wv*paraWv2Pst[0] + paraWv2Pst[1])

def getWvFromPst(pst, paraPst2Wv):
    return np.uint16(pst*paraPst2Wv[0] + paraPst2Wv[1])

def getWvs(range, paraPst2Wv):
    index = np.float32(range)
    return getWvFromPst(index, paraPst2Wv)