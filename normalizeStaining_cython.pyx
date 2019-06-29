import os
import numpy as np
cimport numpy as np
import cv2

def normalizeStaining(imgPath, saveDir='normalized/', unmixStains=False, int Io=240, int alpha=1, double beta=0.15):
    ''' Normalize staining appearence of H&E stained images. 
    Produces a normalized copy of the input RGB image to the saveDir path.
    If unmixStains=True, separate H and E images are also saved.
    
    This is a modified version of the original https://github.com/schaugf/HEnorm_python
    optimized for multiprocessing - June '19 Joni Juvonen
    
    Example use:
        normalizeStaining('image.png', saveDir='normalized/')

    Example use with multiprocessing:
        with Pool(8) as p:
            p.map(normalizeStaining, ImagePathList)
        
    Input:
        imgPath (string): Path to an RGB input image
        saveDir (string): A directory path where the normalized image copies are saved. If this is None, the function returns the images (default='normalized/'))
        unmixStains (bool): save also H and E stain images 
        Io (int): transmitted light intensity (default=240)
        alpha (default=1)
        beta (default=0.15)
        
    Output (returns only if savePath=None):
        Inorm: normalized image
        H: hematoxylin image
        E: eosin image
    
    Reference: 
        A method for normalizing histology slides for quantitative analysis. M.
        Macenko et al., ISBI 2009
    '''

    #extract name for the savefile
    base=os.path.basename(imgPath)
    name_wo_ext = os.path.splitext(base)[0]
    fn = os.path.join(saveDir, name_wo_ext)

    # create output directory if it doesn't exist
    if not os.path.isdir(saveDir):
        os.mkdir(saveDir)
             
    # skip if this file already exists
    if (os.path.isfile(fn+'.png')):
        return

    # read image with OpenCV (faster than PIL)
    img = cv2.imread(imgPath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    cdef np.ndarray[np.float64_t, ndim=2] HERef
    HERef = np.array([[0.5626, 0.2159],
                      [0.7201, 0.8012],
                      [0.4062, 0.5581]])

    cdef np.ndarray[np.float64_t, ndim=1] maxCRef    
    maxCRef = np.array([1.9705, 1.0308])
    
    # define height and width of image
    h, w, c = img.shape
    
    # reshape image
    cdef np.ndarray[np.float64_t, ndim=2] rimg  
    rimg = np.reshape(img.astype(float), (-1,3))
    
    # calculate optical density
    cdef np.ndarray[np.float64_t, ndim=2] OD
    OD = -np.log((rimg+1)/Io)
    
    # remove transparent pixels
    cdef np.ndarray[np.float64_t, ndim=2] ODhat
    ODhat = np.array([i for i in OD if not any(i<beta)])
        
    # compute eigenvectors and handle some of the common errors that are caused by image colors with unattainable eigenvectors
    cdef np.ndarray[np.float64_t, ndim=2] cov
    cov = np.cov(ODhat.T)
    cdef np.ndarray[np.float64_t, ndim=1] eigvals
    cdef np.ndarray[np.float64_t, ndim=2] eigvecs
    #eigvals, eigvecs = None, None
    try:
        eigvals, eigvecs = np.linalg.eigh(cov)
    except AssertionError:
        print('Failed to normalize {0}, copying this to output file unaltered.'.format(imgPath))
        cv2.imwrite(fn+'.png', img)
        return
    except np.linalg.LinAlgError:
        print('Eigenvalues did not converge in {0}, copying this to output file unaltered.'.format(imgPath))
        cv2.imwrite(fn+'.png', img)
        return
    
    #project on the plane spanned by the eigenvectors corresponding to the two 
    # largest eigenvalues
    cdef np.ndarray[np.float64_t, ndim=2] That
    That = ODhat.dot(eigvecs[:,1:3])
    
    cdef np.ndarray[np.float64_t, ndim=1] phi
    phi = np.arctan2(That[:,1],That[:,0])
    
    cdef np.float64_t minPhi
    cdef np.float64_t maxPhi
    minPhi = np.percentile(phi, alpha)
    maxPhi = np.percentile(phi, 100-alpha)
    
    cdef np.ndarray[np.float64_t, ndim=2] vMin
    cdef np.ndarray[np.float64_t, ndim=2] vMax
    vMin = eigvecs[:,1:3].dot(np.array([(np.cos(minPhi), np.sin(minPhi))]).T)
    vMax = eigvecs[:,1:3].dot(np.array([(np.cos(maxPhi), np.sin(maxPhi))]).T)
    
    # a heuristic to make the vector corresponding to hematoxylin first and the 
    # one corresponding to eosin second
    cdef np.ndarray[np.float64_t, ndim=2] HE
    if vMin[0] > vMax[0]:
        HE = np.array((vMin[:,0], vMax[:,0])).T
    else:
        HE = np.array((vMax[:,0], vMin[:,0])).T
    
    # rows correspond to channels (RGB), columns to OD values
    cdef np.ndarray[np.float64_t, ndim=2] Y
    Y = np.reshape(OD, (-1, 3)).T
    
    # determine concentrations of the individual stains
    cdef np.ndarray[np.float64_t, ndim=2] C
    C = np.linalg.lstsq(HE,Y, rcond=None)[0]
    
    # normalize stain concentrations
    cdef np.ndarray[np.float64_t, ndim=1] maxC
    maxC = np.array([np.percentile(C[0,:], 99), np.percentile(C[1,:],99)])
    cdef np.ndarray[np.float64_t, ndim=2] C2
    C2 = np.array([C[:,i]/maxC*maxCRef for i in range(C.shape[1])]).T
    
    # recreate the image using reference mixing matrix
    Inorm = np.multiply(Io, np.exp(-HERef.dot(C2)))
    Inorm[Inorm>255] = 254
    Inorm = np.reshape(Inorm.T, (h, w, 3)).astype(np.uint8)  
    
    # unmix hematoxylin and eosin
    H = np.multiply(Io, np.exp(np.expand_dims(-HERef[:,0], axis=1).dot(np.expand_dims(C2[0,:], axis=0))))
    H[H>255] = 254
    H = np.reshape(H.T, (h, w, 3)).astype(np.uint8)
    
    E = np.multiply(Io, np.exp(np.expand_dims(-HERef[:,1], axis=1).dot(np.expand_dims(C2[1,:], axis=0))))
    E[E>255] = 254
    E = np.reshape(E.T, (h, w, 3)).astype(np.uint8)
    
    if saveDir is not None:
        Inorm = cv2.cvtColor(Inorm, cv2.COLOR_BGR2RGB)
        cv2.imwrite(fn+'.png', Inorm)
        if unmixStains:
            H = cv2.cvtColor(H, cv2.COLOR_BGR2RGB)
            E = cv2.cvtColor(E, cv2.COLOR_BGR2RGB)
            cv2.imwrite(fn+'_H.png', H)
            cv2.imwrite(fn+'_E.png', E)
        return
    else:
        # construct return tuple
        returnTuple = (Inorm,)
        if unmixStains:
            returnTuple = returnTuple +  (H, E)
        return returnTuple
