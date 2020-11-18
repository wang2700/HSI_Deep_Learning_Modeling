import glob
import os
import numpy as np
from smalltools import sortListStringHuman
from fnmatch import fnmatch

class Files(object):
    """description of class
    get files list without extension, files list with path, files count in a
    folder with a specific extension.
    sortFilesName: sort files name list with human sorting method.
    splitFilesName: split files name list with a specific symbol whose default
    value is '_'."""
    def __init__(self, folder, ext, flag ='this'):
        folder = os.path.normpath(folder)
        self.path = folder
        self.ext = ext
        if flag is 'this':
            [self.filesNoExt, self.filesWithPath, self.count] = \
                Files.__getfiles(folder, ext)
        elif flag is 'all':
            [self.filesNoExt, self.filesWithPath, self.count] = \
                Files.__getfilesAll(folder, ext)
        self.filesPath = [folder] * self.count
        self.filesWithPathNoExt = [os.path.split(f)[0] for f in self.filesWithPath]

    @staticmethod
    def __getfiles(dir0, filter0):
        os.chdir(dir0)
        filenamesStr = []
        fileswithpath = []
        for t in filter0:
            for file in glob.glob(t):
                [fileNoExt, ext2] = os.path.splitext(file)
                filenamesStr.append(fileNoExt)
                fileswithpath.append(dir0+'/'+file)
        count = len(filenamesStr)
        return filenamesStr, fileswithpath, count
    
    @staticmethod
    def __getfilesAll(path, filter_):
        filenamesStr = []
        fileswithpath = []
        for path, subdirs, f in os.walk(path):
            for name in f:
                for t in filter_:
                    if fnmatch(name, t):
                        [fileNoExt, ext2] = os.path.splitext(name)
                        filenamesStr.append(fileNoExt)
                        fileswithpath.append(path+'/'+ name)
        count = len(filenamesStr)
        return filenamesStr, fileswithpath, count

    def sortFilesName(self, reg=(0,0)):
        self.filesNoExt, self.filesSortedIndex = sortListStringHuman(
                self.filesNoExt, reg=reg)
        self.filesWithPath = [self.filesWithPath[i] for i in  self.filesSortedIndex]

    def splitFilesName(self, sp='__'):
        b = self.filesNoExt[0].split(sp)
        for file in self.filesNoExt[1:]:
            name = file.split(sp)
            b = np.column_stack((b, name))
        self.namesElements = b
        return b

def getFileFullPath(path, filt):
    os.chdir(path)
    files = glob.glob(filt)
    return files

def splitPath(path):
    pathName = os.path.split(path)
    p = pathName[0]
    name = os.path.splitext(pathName[1])
    n, ext = name[0], name[1]
    return p, n, ext