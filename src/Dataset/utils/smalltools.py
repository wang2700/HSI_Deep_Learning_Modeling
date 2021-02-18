# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 14:13:05 2018

some small but useful function

@author: wangliangju@gmail.com
"""


import numpy as np
import pandas as pd

def sortListStringHuman(text, reg=(0,0)):
    """Sort the string list with human order

    Args:
        text: The input string list

    Returns:
        data: sorted string list

    """
    import re

    def natural_keys(text):
        '''
        alist.sort(key=natural_keys) sorts in human order
        http://nedbatchelder.com/blog/200712/human_sorting.html
        (See Toothy's implementation in the comments)
        float regex comes from https://stackoverflow.com/a/12643073/190597
        '''
        def atof(text):
            try:
                retval = float(text)
                # print(retval)
            except ValueError:
                retval = text
            return retval
        
        return [atof(c) for c in re.split
                (r'[+-]?([0-9]+(?:[.][0-9]*)?|[.][0-9]+)', text)]
    if reg == (0,0):
        textReg = text
    elif reg[1] == 0:
        textReg = [t[reg[0]:] for t in text]
    else: 
        textReg = [t[reg[0]:reg[1]] for t in text]
        
    retIndex = sorted(range(len(textReg)), key=lambda k:natural_keys(textReg[k]))
    retV = [text[i] for i in retIndex]    
    return retV, retIndex


def dataCorrelationFig(pathname, x, y, name, colordict, markerdict={}, xlabel='x', ylabel='y',
                       tittle='tittle'):
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy import stats
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    datasize = len(x)
    slist = []
    clist = []
    if len(name[0]) == 1:
        slist = 'o'
        for n in name:
            clist.append(colordict[n])
    else:
        for n in name[0]:
            slist.append(markerdict[n])
        if len(name) == 2:
            for n in name[1]:
                clist.append(colordict[n])
        if len(name) > 2:
            nameback = name[1:]
            for i in range(datasize):
                c = ''
                for n in nameback:
                    c = c + '_'+n[i]
                clist.append(colordict[c[1:]])
    print(slist, clist)
    unique_markers = set(slist)
    fig, ax = plt.subplots()

    ax.cla()
    for n in unique_markers:
        inds = [i for i, ele in enumerate(slist) if ele == n]
        xsub = [x[i] for i in inds]
        ysub = [y[i] for i in inds]
        csub = [clist[i] for i in inds]
        ax.scatter(xsub, ysub, marker=n, c=csub)
        print(xsub, ysub, csub)

    #para = np.poly1d(np.polyfit(x,y,1))
    #xp = np.linspace(min(x), max(x), 100)
    #plt.plot(xp, para(xp), '-k')
    # regression part
    x = np.asarray(x)
    y = np.asarray(y)
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    line = slope*x+intercept
    ax.plot(x, line, 'k',
            label='y={:.2f}x+{:.2f}\t$r^2$ = {:.2f}'.format(slope, intercept, r_value))
    # end
    # ax.legend(fontsize=9)
    legend_elements = []
    if len(name[0]) > 1:
        for k, v in markerdict.items():
            legend_elements.append(Line2D([0], [0], marker=v, color='k', label=k,
                                          markerfacecolor='w', markersize=5))
    for k, v in colordict.items():
        legend_elements.append(Patch(facecolor=v, edgecolor='w',
                                     label=k))

    legend_elements.append(Line2D([0], [0], color='k', lw=1,
                                  label='y={:.2f}x+{:.2f}\n$r^2$ = {:.2f}'.
                                  format(slope, intercept, r_value)))
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    ax.legend(handles=legend_elements,
              loc='center left', bbox_to_anchor=(1, 0.5))
    plt.title(tittle)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(pathname, dpi=360)
    plt.show()


def div(a, b):
    """Divid, if the result is infinite, set it with zero

    Args:
        a: The dividend
        b: The divisor

    Return:
        c: The quotient

    """

    with np.errstate(divide='ignore', invalid='ignore'):
        c = np.true_divide(a, b)
        c[~np.isfinite(c)] = 0  # -inf inf NaN
    return c
def dictFilterOut(dct, keyfilter, vfilter, flag='all'):
    d = dct.copy()
    value = d[keyfilter]
    ind = List.findList(value, vfilter)
    if flag == 'all':
        d = dictpop(d, ind, keys='all')
    else:
        d = dictpop(d, ind, keys=[keyfilter])
    return d, ind

def dictFilterIn(dct, keyfilter, vfilter, flag='all'):
    d = dct.copy()
    value = d[keyfilter]
    ind = List.findList(value, vfilter)
    if flag == 'all':
        d = dictselect(d, ind, keys='all')
    else:
        d = dictselect(d, ind, keys=[keyfilter])
    return d, ind

def dictpop(dct, ind, keys='all'):
    d = dct.copy()
    if keys == 'all':
        keys = d.keys()
    for k in keys:
        d[k] = [v for i, v in enumerate(d[k]) if i not in ind]
    return d
    
def dictselect(dct, ind, keys='all'):
    d = dct.copy()
    if keys == 'all':
        keys = d.keys()
    for k in keys:
        d[k] = [v for i, v in enumerate(d[k]) if i in ind]
    return d

def crtTimeStr(flag='datetime'):
    """Get the current time string with the formation MM-DD-YYYY_hh-mm-ss.

    Args:
        None

    Returns:
        rtv: The time string

    """
    import datetime
    now = datetime.datetime.now()
    if flag == 'datetime':
        rtv = now.strftime("%Y%m%d-%H%M%S%f")[0:-3]
        rtv
    elif flag == 'date':
        rtv = now.strftime("%Y%m%d")
    elif flag == 'time':
        rtv = now.strftime("%H%M%S%f")[0:-3]
    return rtv

def setDateTimeZone(date):
    """Set the date time and time zone with a string whose formation is MMDDhhmmYYY.ss

    Args:
        date: The string with formation MMDDhhmmYYY.ss
    Returns:
        None

    """
    import subprocess
    datetimeStr = date[0:15]
    timezoneStr = date[15:]
    sudotimezone = subprocess.Popen(
        ["sudo", "timedatectl", "set-timezone", timezoneStr])
    sudotimezone.communicate()
    sudodate = subprocess.Popen(["sudo", "date", datetimeStr])
    sudodate.communicate()


def fig2data(fig):
    """Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it

    Args
        fig: a matplotlib figure
    Return:
        buf: a numpy 3D array of RGBA values

    """
    # draw the renderer
    fig.canvas.draw()
    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    # buf.shape = (w, h, 4)
    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel
    # to have it in RGBA mode
    buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return buf


def getElementFromList(wvlist, i):
    """Get wavelength from the pixel index of the spectrum image

    Args:
      i: The pixel index of the image in column direction
    Returns:
      The wavelength

    """
    return wvlist[i]


def getPathElement(pathstr):
    import os
    path = os.path.normpath(pathstr)
    pathparts = path.split(os.sep)
    return pathparts


def getIndexFromList(wvlist, wv):
    """Get the pixel index in colomn direction from the wavelength

    Args:
      wv: The wavelength
    Returns:
      The pixel index

    """
    bands = len(wvlist)
    wvdict = dict(zip(wvlist, range(0, bands)))
    _, wvPurp = min(enumerate(wvdict.keys()),
                    key=lambda x: abs(x[1]-wv))
    return wvdict[wvPurp]


def findLeftRightEdge(img):
    try:
        img = img[:, :, 0]
    except:
        pass
    maxAlongWidth = np.max(img, axis=0)
    maxAlongWidth = list(maxAlongWidth)
    left = next((i for i, x in enumerate(maxAlongWidth) if x), None)
    maxAlongWidth.reverse()
    right = next((i for i, x in enumerate(maxAlongWidth) if x), None)
    right = len(maxAlongWidth) - right
    return left, right


def isFile(fp):
    from os import path
    return path.isfile(fp)

def mkdirPath(pathstr):
    import os
    path = os.path.normpath(pathstr)
    pathparts = path.split(os.sep)

    for i in range(len(pathparts)):
        crtpath = '/'.join(pathparts[0:i+1])
        print(crtpath, end='  ')
        if os.path.isdir(crtpath):
            print('this is a folder')
        else:
            print('this is not a folder: ', end=' ')
            os.mkdir(crtpath)
            print('created')
            
def msc(input_data, reference=None):
    ''' Perform Multiplicative scatter correction'''
    import numpy as np
    # mean centre correction
    for i in range(input_data.shape[0]):
        input_data[i,:] -= input_data[i,:].mean()
 
    # Get the reference spectrum. If not given, estimate it from the mean    
    if reference is None:    
        # Calculate mean
        ref = np.mean(input_data, axis=0)
    else:
        ref = reference
 
    # Define a new array and populate it with the corrected data    
    data_msc = np.zeros_like(input_data)
    for i in range(input_data.shape[0]):
        # Run regression
        fit = np.polyfit(ref, input_data[i,:], 1, full=True)
        # Apply correction
        data_msc[i,:] = (input_data[i,:] - fit[0][1]) / fit[0][0] 
 
    return (data_msc, ref)

def PCA2D_2D(samples, row_top, col_top):
    '''samples are 2d matrices'''
    import numpy as np
    size = samples[0].shape
    # m*n matrix
    mean = np.zeros(size)

    for s in samples:
        mean = mean + s

    # get the mean of all samples
    mean /= float(len(samples))

    # n*n matrix
    cov_row = np.zeros((size[1],size[1]))
    for s in samples:
        diff = s - mean
        cov_row = cov_row + np.dot(diff.T, diff)
    cov_row /= float(len(samples))
    row_eval, row_evec = np.linalg.eig(cov_row)
    # select the top t evals
    sorted_index = np.argsort(row_eval)
    # using slice operation to reverse
    X = row_evec[:,sorted_index[:-row_top-1 : -1]]

    # m*m matrix
    cov_col = np.zeros((size[0], size[0]))
    for s in samples:
        diff = s - mean
        cov_col += np.dot(diff,diff.T)
    cov_col /= float(len(samples))
    col_eval, col_evec = np.linalg.eig(cov_col)
    sorted_index = np.argsort(col_eval)
    Z = col_evec[:,sorted_index[:-col_top-1 : -1]]

    return X, Z

class Table:    
    def __init__(self, path):
        self.path = path
    @staticmethod
    def readcsv2array(path):
        import pandas as pd
        df = pd.read_csv(path, sep=',', header=None)
        rtData = df.values
        return rtData
    @staticmethod
    def readcsv2dict(path):
        import csv
        reader = csv.reader(open(path, 'r'))
        result = {}
        keys = []
        for i, row in enumerate(reader):
            if i > 0:
                for j, v in enumerate(row):
                    result[keys[j]] += [v]
            else:
                for v in row:
                    result[v] = []
                keys = row
        return result
    def read(self):
        self.tabledf = pd.read_csv(self.path, sep=',', header=0)
        #return self.tabledf
    def asDict(self):
        self.tabledict = self.tabledf.to_dict('list')
        return self.tabledict
        
    def filtering(self, filterkey, filterfunc, axis=1):
        self.rawdf = self.tabledf
        df = self.tabledf
        for k, f in zip(filterkey, filterfunc):
            df = df[df[k].apply(f, axis=axis)]
        self.tabledf = df
        
    def write(self, path):
        self.tabledf.to_csv(path)  
        
    def join(self, keys, sep, name='newcolu'):        
        df = self.tabledf
        def f(df0):
            vlist = [df0[k] for k in keys]
            rtv = sep.join(vlist)
            return rtv        
        ser = df[keys].apply(f, axis=1)
        self.tabledf[name] = ser
        return ser
    def newCol(self, name='newcolu', keys=[], f=[]):
        self.tabledf[name] = ''
        if f != []:
            df = self.tabledf   
            if keys != []:
                df[name] = df[keys].apply(f, axis=1)
            else:
                df[name] = df[name].apply(f, axis=1)
        self.tabledf = df
    
class List:
    def __init__(self):
        pass
    @staticmethod
    def findList(lst, v):
        return list(filter(lambda x: lst[x] == v, range(len(lst))))
        
    @staticmethod
    def rmDuplicates(lst):
        return list(dict.fromkeys(lst))
    @staticmethod
    def replace(lst, dct):        
        for i, l in enumerate(lst):
            for k, v in dct.items():
                if l == k:
                    lst[i] = v
        return lst
class PLT:
    
    from matplotlib import cm
    from numpy import linspace
    import matplotlib as mpl
    
    def __init__(self):
        pass
    @staticmethod    
    def mscatter(x,y,ax=None, m=None, **kw):
        import matplotlib.markers as mmarkers
        import matplotlib.pyplot as plt
        if ax is None:
            ax=plt.gca()
        sc = ax.scatter(x,y,**kw)
        if (m is not None) and (len(m)==len(x)):
            paths = []
            for marker in m:
                if isinstance(marker, mmarkers.MarkerStyle):
                    marker_obj = marker
                else:
                    marker_obj = mmarkers.MarkerStyle(marker)
                path = marker_obj.get_path().transformed(
                            marker_obj.get_transform())
                paths.append(path)
            sc.set_paths(paths)
        return sc
    @staticmethod
    def plotComponents(x, y, colorlist=[], shapelist=[], levelCnt = 3,
                       colormap='jet'):
        import math
        from matplotlib import cm
        from numpy import linspace
        import matplotlib.pyplot as plt
        compCnt = x.shape[1]
        y = y.T.tolist()
        if colorlist != []:
            colors = colorlist
        else:
            start, stop = 0.0, 1.0
            cm_subsection = linspace(start, stop, levelCnt)
            colors = [cm.get_cmap(colormap)(x) for x in cm_subsection]

        classShapes=('o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X')
        classCnt = len(y)
        plt.figure(figsize=(8, 6))
        pltCols = math.ceil(classCnt/2)
        fig, axs = plt.subplots(2,pltCols)
        if x.shape[1] >= 2:
            for i, v in enumerate(y):                
                v = list(map(str, v))
                vlevel = List.rmDuplicates(v)
                dct = {vl: colors[j] for j, vl in enumerate(vlevel) }
                cs = List.replace(v, dct)
                ax = axs[int(i/2)][i - int(i/2)*2]
                ax.scatter(x[:, 0], x[:,1], c=cs, marker=classShapes[i])
            plt.show()

    @staticmethod
    def plotMultiCurves(x, y, legend, legendFontsize='large', path='',
                        linewidth=0.2, colormap='jet', usercolor=[],
                        xlabel='x', ylabel='y', labelFontsize='large'):
        import matplotlib.pyplot as plt
        from matplotlib import cm
        from numpy import linspace
        import matplotlib as mpl
        mpl.rc('font', family='Arial')
        start = 0.0
        stop = 1.0
        number_of_lines = y.shape[1]
        if colormap == 'user':
            colors = usercolor
        else:
            cm_subsection = linspace(start, stop, number_of_lines)
            colors = [cm.get_cmap(colormap)(x) for x in cm_subsection]
        for i, color in enumerate(colors):
            plt.plot(x, y[:, i], color=color, linewidth=linewidth)
        plt.legend(legend, fontsize=legendFontsize)
        plt.xlabel(xlabel, fontsize=labelFontsize)
        plt.ylabel(ylabel, fontsize=labelFontsize)
        if path == '':
            plt.show()
        else:
            plt.savefig(path, dpi=600)
        plt.clf()
'''
read json files in a folder.
input:
    path
    labeltype='points'
return
    allLabelDict = {'filename':label={'labelname0':label0, 'labelname1':label1, ...},
                    ...}
    fileNameOnlyList = [filename0, filename1, ...], which is sorted
'''

class Read:
    def __init__(self):
        pass
    @staticmethod
    def readJsonFiles(path, labeltype='points'):
        from wljbox.files import Files
        import json
        import numpy as np
        jsonFiles = Files(path, ['*.json'])
        jsonFiles.sortFilesName()
        fileNameList = jsonFiles.filesWithPath
        fileNameOnlyList = jsonFiles.filesNoExt
        allLabelDict = {}
        for i, f in enumerate(fileNameList):    
            filename = fileNameOnlyList[i]
            jsonStr = open(f).read()
            data = json.loads(jsonStr)
            labeldictlist = data['shapes']
            newLabel = {}
            for label in labeldictlist:
               # print(label['label'], label['points'])
               points = label[labeltype]
               newLabel[label['label']]= label[labeltype]
            allLabelDict[filename] = newLabel
        return allLabelDict, fileNameOnlyList
    @staticmethod
    def readMat2Dict(path):
        import scipy.io as sio
        f = sio.loadmat(path)
        return f

    @staticmethod
    def readNpNdArrfromFile(filename):
        import bloscpack as bp
        arr = bp.unpack_ndarray_from_file(filename)
        return arr

    @staticmethod
    def readPickle(name):
        import pickle
        with open(name, 'rb') as f:
            return pickle.load(f)
    @staticmethod
    def readcsv2array(path):
        import pandas as pd
        df = pd.read_csv(path, sep=',', header=None)
        rtData = df.values
        return rtData
    @staticmethod
    def readcsv2dict(path):
        import csv
        reader = csv.reader(open(path, 'r'))
        result = {}
        keys = []
        for i, row in enumerate(reader):
            if i > 0:
                for j, v in enumerate(row):
                    result[keys[j]] += [v]
            else:
                for v in row:
                    result[v] = []
                keys = row
        return result
    @staticmethod
    def readRaw(path):
        a = open(path, 'rb').read()
        return a

    @staticmethod
    def readRawPt(path, height=608, width=808):
        a = open(path, 'rb').read()

        dt = np.dtype(np.uint16)
        dt = dt.newbyteorder('>')
        cc = np.frombuffer(a, dtype=dt)
        img = np.reshape(cc, (608, 808))
        h = np.left_shift(img, 4)
        l = np.right_shift(img, 12)
        img16bit = (h+l)*16
        return img16bit

    @staticmethod
    def tiff2np(path, swapaxis=True):
        from tifffile import imread
        nparray = imread(path)
        if swapaxis:
            nparray = np.rollaxis(nparray, 0, 3)
        return nparray

class Write:
    def __init__(self):
        pass
    @staticmethod
    def save2DArray2csv(path, arr):
        import pandas as pd
        df = pd.DataFrame(arr)
        df.to_csv(path, index=False, header=False)
    @staticmethod
    def saveDict2csv(path, dsrc):
        import pandas as pd
        df = pd.DataFrame.from_dict(dsrc)
        df.to_csv(path, index=False)

    @staticmethod
    def saveDict2mat(path, d):
        import scipy.io as sio
        sio.savemat(path, d)

    @staticmethod
    def saveHeatmap(filename, data, rangee=[0, 1], bar=True, orient='horizontal', dpi=800):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(16, 9))
        im = ax.imshow(data, cmap='jet')
        im.set_clim(rangee[0], rangee[1])
        if bar == True:
            fig.colorbar(im, orientation=orient, pad=0.2)
        plt.savefig(filename, dpi=dpi)
        plt.clf()

    @staticmethod
    def saveNpNdArr2blp(filename, arr):
        import bloscpack as bp
        bp.pack_ndarray_to_file(arr, filename)

    @staticmethod
    def savePickle(name, obj):
        import pickle
        if name[-4:] != '.pkl':
            name = name + '.pkl'
        with open(name, 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
    @staticmethod        
    def csvWriteHeader(name, content):
        import csv
        from collections import OrderedDict
        ordered_fieldnames = OrderedDict([('field1', None), ('field2', None)])
        with open(name, 'w') as fou:
            dw = csv.DictWriter(fou, fieldnames=content, dialect='excel')
            dw.writeheader()

    @staticmethod
    def csvWriteRow(name, content):
        import csv
        with open(name, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(content)
    @staticmethod
    def tiff(name, nparray, swapaxis = True):
        from tifffile import imsave
        if swapaxis:
            nparray = np.rollaxis(nparray, 2, 0)
        imsave(name, nparray)

def segRedEdge(img, wvs, th, denoisSize=0, denoiseIter=1):
    redwv = 680
    nirwv = 750
    h, w, b = img.shape
    rindex = stools.getIndexFromList(wvs, redwv)
    nirindex = stools.getIndexFromList(wvs, nirwv)
    edgelength = nirindex - rindex
    weight = np.arange(-edgelength/2.0+0.5, edgelength/2.0+0.5)
    print(weight)
    img2d = img[:, :, rindex:nirindex].reshape((-1, edgelength))
    img1d = np.matmul(img2d, weight)/np.matmul(weight.T, weight)**0.5
    img2seg = img1d.reshape((h, w))
    mask = img2seg.copy()
    mask[mask <= th] = 0
    mask[mask > th] = 1
    if denoisSize > 0:
        kernel = np.ones((5, 5))
        mask = cv2.erode(mask, kernel, iterations=denoiseIter)
        mask = cv2.dilate(mask, kernel, iterations=denoiseIter)
    makedImg = img * (np.dstack([mask]*b))
    makedImg = makedImg
    return makedImg, mask, img2seg


def traverseFolders(path1, path2, filt1, operationf, filt2='*'):
    from wljbox.files import Files
    if filt2 == '*':
        files1 = Files(path1, [filt1])
        names1 = files1.filesNoExt
        count1 = files1.count
        print('count of files1', count1)
        for i in range(count1):
            name1 = names1[i]
            print(name1, end=' ')
            flag = operationf(path1, path2, name1)
            if flag == 0:
                continue
            elif flag == 1:
                print('got')
            else:
                print('failed', name1)
    else:
        files1 = Files(path1, [filt1])
        files2 = Files(path2, [filt2])
        names1 = files1.filesNoExt
        files2.sortFilesName()
        names2 = files2.filesNoExt
        namepath1 = files1.filesWithPath
        namepath2 = files2.filesWithPath
        count1 = files1.count
        count2 = files2.count
        print('count of files1', count1, 'count of files2', count2)
        for i in range(count1):
            name1 = names1[i]
            print(name1, end=' ')
            for j in range(count2):
                name2 = names2[j]
                flag = operationf(path1, path2, name1, name2)
                if flag == 0:
                    continue
                elif flag == 1:
                    print('got', name1, name2)
                    break
                else:
                    print('none', name1, name2)
                    break


def downSampleMean(arr, outSize, axis=0):
    import numpy as np
    arrLen = arr.shape[axis]
    perPartSize = int(arrLen/outSize)
    remSize = arrLen - perPartSize*outSize
    partSizeList = [perPartSize+1]*remSize + [perPartSize]*(outSize-remSize)

    def arrayApplyFunc(a):
        st = 0
        for i, s in enumerate(partSizeList):
            if i == 0:
                arrTemp = np.mean(a[st:st+s])
            else:
                arrTemp = np.append(arrTemp, np.mean(a[st:st+s]))
            st = st + s
        return arrTemp
    return np.apply_along_axis(arrayApplyFunc, axis, arr)

if __name__ == '__main__':

    fp = r'F:/data/leafspec/apple/LeafSpec_data/data1rnd/appleDataMatched/tableModified.csv'
    table = Table(fp)
    table.read()
    table.filtering([['geno','index']], [lambda x: (x['index']<200) & (len(x['geno'])>7)])
    #table.filtering([['geno']], [lambda x: len(x['geno'])<2], axis=1)
    #print(table.tabledf['geno'])
    ser = table.join(['geno','filename'], '/')
    print(table.tabledf.head(10))
    '''
    import numpy as np
    x = np.random.rand(10,2)
    y = np.random.rand(10,3)
    y[y>0.6]=3
    y[y<0.3]=1
    y[y<1]=2
    PLT.plotComponents(x, y)
    '''