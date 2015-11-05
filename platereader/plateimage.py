#This should not be required since OpenCV will import, but it's a
#known issue with some numpy functions and PyInstaller
import numpy as np
import cv2
import matplotlib.pyplot as plt
import warnings
        
class PlateImage(object):
    """ Represents a plate image and all the methods for analysing it."""
    
    #The original image (but maybe cropped)
    _rawImage=None

    #Mask of pixels to be excluded based on plate geometry
    _geometryMask=None

    #Mask of pixels belonging to the background. Boolean array with True if a pixel
    #belongs to the background. Only used internally, see listOfBlobs
    _backgroundMask=None
    
    #Mask of cleared areas
    _clearMask=None

    #List of the contours for all potentially good blobs
    _listOfBlobs=None
    
    #To keep track of blobs which were deleted due to bad shape or size
    _listOfBadBlobs=None
    
    #Background color
    _bgColor=None

    #Colony color
    _colColor=None

    #Name of the loaded file
    _fileName=None

    #Dictionary of the parameters
    _parameters=None

    #The fitted circles. List of Nx3 array with x,y,r, one for each blob
    _listOfCircles=[]
    
    #All circles before filtering. Used internally for testing
    _listOfAllCircles=[]
    
    #Whether to mask (True) or unmask (False) when using manual drawing
    _maskWhenDrawing=True

    @property
    def statistics(self):
        """ Dictionary with statistics about the colony fittings."""
        
        #Default values
        stat={'numberOfColonies':0,
                      'meanColonyDiameter':np.nan,
                      'meanColonyArea':np.nan,
                      'medianColonyDiameter':np.nan,
                      'medianColonyArea':np.nan,
                      'numberOfBlobs':0,
                      'totalBlobArea':0,
                      'totalCircleArea':0,
                      'stdOfColonies':np.nan,
                      'blobFit':np.nan,
                      'numberOfBadBlobs':0,
                      'fractionBadBlobArea':np.nan,
                      'coloniesPerBlob':np.nan,
                      'adjNumberOfColonies':0,
                      'fileName':self._fileName,
                      'listOfSizes':[]
                      }  
        if not self._geometryMask is None and not self._listOfBlobs is None:
            if len(self._listOfBlobs)>0:
                #First concatenate all circle sizes
                sizes=[]
                
                blobMap=np.zeros(self._geometryMask.shape)
                circleMap=np.zeros(self._geometryMask.shape)
                badBlobAreas=[]
                
                numberBad=0 #Number of bad blobs
                for i in range(0,len(self._listOfBlobs)):
                    if len(self._listOfCircles[i])==0:
                        #For bad blobs
                        numberBad=numberBad+1
                        badBlobAreas.append(cv2.contourArea(self._listOfBlobs[i]))
                        cv2.drawContours(blobMap,self._listOfBlobs,i,2,-1)
                    else:
                        #Good blobs
                        cv2.drawContours(blobMap,self._listOfBlobs,i,1,-1)
                
                #Scale to diameter and mm
                factor=2*float(self._parameters['plateDiameter'])/float(self._rawImage.shape[0])
                for i in self._listOfCircles:
                    sizes=sizes+(i[:,2]*factor).tolist()
                    
                    #Draw the circles on the blobMap
                    for circle in i:
                        cv2.circle(circleMap,(int(np.rint(circle[0])),int(np.rint(circle[1]))),int(np.rint(circle[2])),1,-1)
                
                #Calculate difference between the blob areas and circle areas
                blobMapSum=float(blobMap.astype(bool).sum())
                mismatchMapSum=float(np.bitwise_xor(blobMap==1,circleMap.astype(bool)).sum())
                
                sizes=np.array(sizes)
                
                #Calculate some stuff here to avoid division by zero
                if (blobMap==1).sum()>0:
                    blobFit=1-mismatchMapSum/(blobMap==1).sum()
                else:
                    blobFit=0
                    
                if len(self._listOfCircles)>0:
                    colPerBlob=float(len(sizes))/len(self._listOfCircles)
                else:
                    colPerBlob=0
                    
                #Calculate the adjusted number of colonies
                meanColArea=np.mean(np.square(sizes/2)*np.pi)
                badBlobCols=np.rint(np.array(badBlobAreas)*np.square(factor/2)/meanColArea)
                badBlobCols[badBlobCols<1]=1
                                
                stat={'numberOfColonies':len(sizes),
                      'meanColonyDiameter':np.mean(sizes),
                      'meanColonyArea':meanColArea,
                      'medianColonyDiameter':np.median(sizes),
                      'medianColonyArea':np.median(np.square(sizes/2)*np.pi),
                      'numberOfBlobs':len(self._listOfCircles),
                      'totalBlobArea':np.square(self._parameters['plateDiameter'])*blobMapSum/(self._geometryMask.shape[0]*self._geometryMask.shape[1]),
                      'totalCircleArea':np.sum(np.square(sizes/2)*np.pi),
                      'stdOfColonies':np.std(sizes),
                      'blobFit':blobFit,
                      'numberOfBadBlobs':numberBad,
                      'fractionBadBlobArea':(blobMap==2).sum()/blobMapSum,
                      'coloniesPerBlob':colPerBlob,
                      'adjNumberOfColonies':len(sizes)+sum(badBlobCols),
                      'fileName':self._fileName,
                      'listOfSizes':sizes.tolist()
                      }         
        return stat 
        
    @property
    def bgColor(self):
        """ The background color in RGB."""
        if not self._bgColor is None:
            #Convert from BGR to RGB
            return self._bgColor[[2,1,0]]
        else:
            return None
    
    @bgColor.setter
    def bgColor(self,value):
        if not value is None:
            #Convert to BGR
            self._bgColor=value[[2,1,0]]
        else:
            self._bgColor=None
    
    @property
    def maskWhenDrawing(self):
        """ Whether to mask or unmask areas when using the manual drawing tools."""
        return self._maskWhenDrawing  
            
    @maskWhenDrawing.setter
    def maskWhenDrawing(self,value):
        self._maskWhenDrawing=value
            
    @property
    def colColor(self):
        """ The colony color in RGB."""
        if not self._colColor is None:
            #Convert from BGR to RGB
            return self._colColor[[2,1,0]]
        else:
            return None    
            
    @colColor.setter
    def colColor(self,value):
        if not value is None:
            #Convert to BGR
            self._colColor=value[[2,1,0]]
        else:
            self._colColor=None
            
    @property
    def histogram(self):
        """ A histogram image for the sizes of the fitted colonies."""
        if not self._listOfCircles is None:
            if len(self._listOfCircles)>0:
                #First concatenate all circle sizes
                sizes=[]
                #Scale to diameter and mm
                factor=2*float(self._parameters['plateDiameter'])/float(self._rawImage.shape[0])
                for i in self._listOfCircles:
                    sizes=sizes+(i[:,2]*factor).tolist()
                    
                if len(sizes)>0:
                    # Turn interactive plotting off
                    plt.ioff()
                    
                    # Create a new figure, plot into it, then close it so it never gets displayed
                    fig=plt.figure()
                    
                    plt.hist(sizes)
                    plt.xlabel('Colony diameter (mm)')
                    plt.ylabel('Count')
                    fig.patch.set_facecolor('white')
                    fig.canvas.draw()
        
                    # Get the RGBA buffer from the figure
                    w,h=fig.canvas.get_width_height()
                    buf=np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8 )
                    plt.close(fig)
                    
                    return np.fromstring(buf, dtype=np.uint8).reshape(h,w,3)
                else:
                    return None
            else:
                return None
        else:
            return None
            
    @property
    def rawImage(self):
        """ The raw image in RGB format."""
        #Convert from BGR to RGB
        if not self._rawImage is None:
            return np.dstack((self._rawImage[:,:,2],self._rawImage[:,:,1],self._rawImage[:,:,0]))
        else:
            return None
        
    @property
    def image(self):
        """ The working image in RGB format."""
        
        #Apply all masks
        if not self._rawImage is None:
            toShow=self._rawImage.copy()
            #This is because drawContours can't draw in 2D images
            if self._parameters['showMarkup']:
                cv2.drawContours(toShow,self._listOfBadBlobs,-1,[0,0,255],-1)
                if not self._listOfBlobs is None:
                    for i in range(0,len(self._listOfBlobs)):
                        if len(self._listOfCircles[i])==0:
                            cv2.drawContours(toShow,self._listOfBlobs,i,[0,255,255],-1)
                        else:
                            cv2.drawContours(toShow,self._listOfBlobs,i,[0,255,0],-1)
            
#            if not self._listOfAllCircles is None:
#                for circles in self._listOfAllCircles:
#                    for circle in circles:
#                        cv2.circle(toShow,(int(np.rint(circle[0])),int(np.rint(circle[1]))),int(np.rint(circle[2])),(0,0,0),2)

            if not self._listOfCircles is None:
                for circles in self._listOfCircles:
                    for circle in circles:
                        cv2.circle(toShow,(int(np.rint(circle[0])),int(np.rint(circle[1]))),int(np.rint(circle[2])),(0,0,0),3)
            
            #Make the clear mask a red shade
            if not self._clearMask is None:
                toShow[self._clearMask,2]=255

            #Make the geometry mask black
            if not self._geometryMask is None:
                toShow[self._geometryMask]=0
                
            #Convert from BGR to RGB
            return np.dstack((toShow[:,:,2],toShow[:,:,1],toShow[:,:,0]))
        else:
            return None
            
    @property
    def fileName(self):
        """ The current file name."""
        return self._fileName
            
    def __init__(self):
        """ Create the PlateImage object."""

        warnings.simplefilter("error")
        #minColonySize:         minimal colony diameter (mm)
        #maxColonySize:         maximal colony diameter (mm)
        #maxBlobSize:           maximal blob (one or several overlapping colonies) diameter (mm)
        #bgDetectFactor:        larger values mean less strict background
        #                       detection (larger objects returned)
        #maxDiameterRatio:      maximal ratio between long and short dimension of a blob
        #minHullDeviation:      minimal deviation from the convex hull to be considered
        #                       relevant for intersect detection (pixels)
        #minSegmentDist:        minimal distance of intersection points in a contour,
        #                       as measured in pixels on each side of a local maxima
        #minCircleCoverage:     minimal proportion of a circle being inside the blob
        #minCircleContrib:      minimal fraction of contribution to blob coverage
        #                       for each circle
        #maxCircleOverlap:      maximal fraction of circle overlapping with other circles
        #minBlobCoverage:       minimal fraction of blob covered for it to be considered
        #                       successful
        #showMarkup:            True if the blobs should be colored by type
        #minBlobCircularity:    minimal circularity (4*pi*A/P^2, A=area, P=perimiter).
        #                       this is complementary to maxDiameterRatio in that it also
        #                       finds "C-shaped" blobs
        #blobDetectionFactor:   larger values mean less strict blob
        #                       detection (fewer objects returned)

        self._parameters={'minColonySize':0.3,
                        'maxColonySize':7,
                        'maxBlobSize':10,
                        'bgDetectFactor':0.5,
                        'plateDiameter':90,
                        'maxDiameterRatio':4,
                        'minHullDeviation':2,
                        'minSegmentDist':5,
                        'minCircleCoverage':0.8,
                        'minCircleContrib':0.005,
                        'maxCircleOverlap':0.8,
                        'minBlobCoverage':0.8,
                        'showMarkup':True,
                        'minBlobCircularity':0.2,
                        'blobDetectFactor':0.9,
                        'clearBorder':0.07}
                        
    def setParameter(self,name,value):
        """ Set a parameter."""
        
        #Should check that it exists first
        self._parameters[name]=value
        
    def getParameter(self,name):
        """ Get a parameter value."""
        
        return self._parameters[name]
        
    def getAllParameterNames(self):
        """ Get all parameter names."""
        
        return self._parameters.keys()

    
    def loadImage(self,fileName=None):
        """ Loads an image as RGB."""

        fileName=fileName or self._fileName
        
        if not fileName is None:
            rawImage=cv2.imread(fileName)
            if rawImage is None:
                raise IOError
            else:
                self._rawImage=rawImage
                self._clearMask=np.zeros((rawImage.shape[0],rawImage.shape[1]),dtype=bool)
                self._geometryMask=None
                self._backgroundMask=None
                self._listOfBlobs=None
                self._listOfBadBlobs=None
                self._fileName=fileName
                self._listOfCircles=[]
                self._listOfAllCircles=[]
                
    def maskUnmaskArea(self,points,geometry):
        """ Mask or unmask an area. 
        
            geometry: 'line', 'rect' or 'circle'
        """
        
        minX=min(points[0][1],points[1][1])
        maxX=max(points[0][1],points[1][1])
        minY=min(points[0][0],points[1][0])
        maxY=max(points[0][0],points[1][0])
        if geometry=='rect':
            self._clearMask[int(np.rint(minX)):int(np.rint(maxX)),int(np.rint(minY)):int(np.rint(maxY))]=self._maskWhenDrawing
        
        if geometry=='circle':
            tempMask=self._clearMask.astype(int) #Typecast to int
            midX=minX+(maxX-minX)/2            
            midY=minY+(maxY-minY)/2
            
            cv2.circle(tempMask,(int(np.rint(midY)),int(np.rint(midX))),int(np.sqrt(np.square(midX-minX)+np.square(midY-minY))),self._maskWhenDrawing,-1)
            self._clearMask=tempMask.astype(bool)
            
        if geometry=='line':
            tempMask=self._clearMask.astype(int) #Typecast to int
            cv2.line(tempMask,(int(np.rint(points[0][0])),int(np.rint(points[0][1]))),(int(np.rint(points[1][0])),int(np.rint(points[1][1]))),self._maskWhenDrawing,4)
            self._clearMask=tempMask.astype(bool)
               
    def undoClearArea(self,points):
        """ Include a rectangular area from blob detection based on two points."""

        self._clearMask[min(points[0][1],points[1][1]):max(points[0][1],points[1][1]),min(points[0][0],points[1][0]):max(points[0][0],points[1][0])]=False
            
    def fitPlateGeometry(self,points):
        """ Fit an ellipse to the points, crop structures and calculate geometry mask.
            
            points: 5x2 array with x,y coordinates
        """
        
        #This typecasting should not be required, but is because of a bug
        #x0, y0, a, b, theta
        ellipse=cv2.fitEllipse(np.float32(points))
        
        self._geometryMask=np.ones((self._rawImage.shape[0],self._rawImage.shape[1]))
        
        #-1 means to create a filled ellipse
        cv2.ellipse(self._geometryMask,ellipse,(0),-1)
        
        #Crop images to fit ellipse. I do it like this because I didn't know how to interpret
        #the theta from the fitting
        imagePositions=np.nonzero(self._geometryMask==0)
        
        leftX=np.min(imagePositions[0])
        rightX=np.max(imagePositions[0])
        topY=np.min(imagePositions[1])
        bottomY=np.max(imagePositions[1])
        
        #Also exclude a border close to the edge of the colony. This is because
        #reflections make it difficult to identify colonies there (and there should
        #not be any anyways)
        #Must typecast as integer first
        if self._parameters['clearBorder']>0:
            temp=np.ones(self._clearMask.shape)
            w=np.int((rightX-leftX)*self._parameters['clearBorder'])
            ellipse=(ellipse[0],(ellipse[1][0]-w,ellipse[1][1]-w),ellipse[2])
            
            cv2.ellipse(temp,ellipse,(0),-1) #Clear inner ellipse
            self._clearMask=np.bitwise_or(self._clearMask,temp.astype(bool))

        #Crop all structures
        self._rawImage=self._rawImage[leftX:rightX][:,topY:bottomY]
        self._geometryMask=self._geometryMask[leftX:rightX][:,topY:bottomY]
        self._geometryMask=self._geometryMask.astype(bool)
        self._clearMask=self._clearMask[leftX:rightX][:,topY:bottomY]
        
#        gray=cv2.cvtColor(self._rawImage,cv2.COLOR_BGR2GRAY).astype(float)
#        masked=np.ma.masked_array(gray,self._geometryMask)
#        medianIntensity=np.median(masked,0)
#        medianIntensity=medianIntensity/np.mean(medianIntensity)
#        gray=[]
#        masked=[]
#        self._rawImage=self._rawImage/medianIntensity[:,np.newaxis]
#        self._rawImage=self._rawImage.astype('uint8')
#        self._rawImage[self._rawImage>255]=255
        
    def _calculateBackground(self):
        """ Calculate the background mask.
            
            The background if taken to be all pixels which are closer to the background
            color than to the colony color, weighted by bgDetectFactor.
        """
        
        if not self._bgColor is None and not self._colColor is None and not self._rawImage is None:
            #The main problem when it comes to detecting the background is that there can be an
            #overall intensity gradient due to lighting. It is therefore difficult to find one
            #colony/background color which is representative for all colonies. Here I normalize the
            #intensity in the horizontal direction. This is a special case which comes from that
            #the lamps in my photos are positioned in that direction. This is a temporary solution
            #and should be generalized
#            gray=cv2.cvtColor(self._rawImage,cv2.COLOR_BGR2GRAY).astype(float)
#            masked=np.ma.masked_array(gray,self._geometryMask)
#            medianIntensity=np.mean(masked,0)
#            medianIntensity=medianIntensity/np.mean(medianIntensity)
#            gray=[]
#            masked=[]
#            self._rawImage=self._rawImage/medianIntensity[:,np.newaxis]
#            self._rawImage=self._rawImage.astype('uint8')
#            #The main problem when it comes to detecting the background is that there can be an
#            #overall intensity gradient due to lighting. It is therefore difficult to find one
#            #colony/background color which is representative for all colonies. The first step
#            #is therefore to detect high-contrast regions in the image            
#            gray=cv2.cvtColor(self._rawImage,cv2.COLOR_BGR2GRAY).astype(float)
#            gray=cv2.blur(cv2.blur(gray,(5,5)),(5,5))            
#            #contrast=np.sqrt(cv2.blur(np.square(gray),(5,5))-np.square(cv2.blur(gray,(5,5))))
#            contrast=cv2.Canny(gray.astype('uint8'),10,20)            
#            cv2.imshow('dff',contrast)
#            
#            #It can be assumed that the high-contrast edges belong to the highest 20%
#            I=np.sort(contrast[~self._geometryMask],axis=None)
#            minIntensity=I[int(len(I)*(1-0.2))]
#            
#            contrast[np.bitwise_or(contrast<minIntensity,self._geometryMask)]=0
#
#            #Then get the intensity of the pixels and remove dark parts
#            I=np.sort(gray[~self._geometryMask],axis=None)
#            minIntensity=I[int(len(I)*(1-0.2))]
#            contrast[gray<minIntensity]=0           
#            
#            cv2.imshow('df',cv2.resize(contrast/10,(1000,1000)))
#            self._backgroundMask=contrast==0
#            
#            #The background is detected in tow steps. First the colony color is determined and
#            #then a mask is created based on the similarity to the colony color            
#            
#            #First exclude areas which belong to the darker part of the image
#            bgMask=np.bitwise_or(self._geometryMask>0,self._clearMask)
#            gray=cv2.cvtColor(self._rawImage,cv2.COLOR_BGR2GRAY)
#            
#            #Smooth the gray image to get rid of noise and decrease the number of
#            #connected components
#            gray=cv2.blur(cv2.blur(gray,(5,5)),(5,5))
#            
#            I=np.sort(gray[~bgMask],axis=None)
#            minIntensity=I[int(len(I)*(1-0.2))]
#            bgMask=np.bitwise_or(bgMask,gray<minIntensity)
#            
#            #Get the contours in the mask
#            contours, hierarchy = cv2.findContours((~bgMask).astype('uint8'), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
#
#            #Ignore contours out of size range
#            #Get limits on blob diameter
#            minDia=min([self._rawImage.shape[0],self._rawImage.shape[1]])*self._parameters['minColonySize']/self._parameters['plateDiameter']
#            maxDia=min([self._rawImage.shape[0],self._rawImage.shape[1]])*self._parameters['maxBlobSize']/self._parameters['plateDiameter']
#
#            goodContours=[]
#            for contour in contours:
#                rect=cv2.minAreaRect(contour)
#                if min(rect[1])>=minDia and max(rect[1])<=maxDia and max(rect[1])/float(min(rect[1]))<=self._parameters['maxDiameterRatio']:
#                    goodContours.append(contour)
#
#            #Get contour mask
#            cMask=np.zeros(gray.shape)
#            cv2.drawContours(cMask,goodContours,-1,1,-1)
#            
#            #For memory reasons
#            del contours
#            del bgMask
#            del gray
#            
#            #Get the mean color for the contours and the standard deviation
#            pixelColors=self._rawImage[cMask.astype(bool),:]
#            meanColor=np.mean(pixelColors,0)
#            stdColor=np.std(pixelColors,0)
#        
#            #Remove all pixels where any of the colors are more than two standard
#            #deviations away, or where the the total distance is more than one
#            #standard deviation away
#            cDist=np.abs(self._rawImage-meanColor)
#            
#            distMask=np.any(cDist>(stdColor*2),2)
#        
#            #self._rawImage[cMask.astype(bool),:]=150
#            
#            self._backgroundMask=distMask
            
#            #I=I[range(0,len(I),100)]
            
#            #First get the distances to the colony color
#            dist=np.sqrt(np.square(self._rawImage[:,:,0]-self._colColor[0])+np.square(self._rawImage[:,:,1]-self._colColor[1])+np.square(self._rawImage[:,:,2]-self._colColor[2]))
#            #I=np.sort(dist[self._geometryMask==0],axis=None)
#
#            #The approach now it to try to get the distribution of colony color and then exclude
#            #pixels significantly different from its mean. This is tricky since this should
#            #be done on a per-object basis to prevent large reflexes of having too large an
#            #effect. However, it cannot be done on the components in the background mask
#            #directly since there will be small objects due to noise. Therefore start by smoothing
#            #the image
#            sDist=cv2.blur(cv2.blur(dist,(3,3)),(3,3))            
#            
#            #Get the maximal distance to the colony color. It should be smaller
#            #than 70% of the distance to the background color. This is not super important
#            #since it's just for getting an initial guess
#            bgDist=0.5*np.sqrt(np.square(self._bgColor[0]-self._colColor[0])+np.square(self._bgColor[1]-self._colColor[1])+np.square(self._bgColor[2]-self._colColor[2]))
#          
#            #Get the preliminary background
#            knownBg=np.bitwise_or(np.bitwise_or(self._geometryMask>0,sDist>bgDist),self._clearMask)
#            
#            #Get the contours    
#            contours, hierarchy = cv2.findContours((~knownBg).astype('uint8'), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
#                        
#            #Ignore contours out of size range
#            #Get limits on blob diameter
#            minDia=min([self._rawImage.shape[0],self._rawImage.shape[1]])*self._parameters['minColonySize']/self._parameters['plateDiameter']
#            maxDia=min([self._rawImage.shape[0],self._rawImage.shape[1]])*self._parameters['maxBlobSize']/self._parameters['plateDiameter']
#            
#            goodContours=[]
#            for contour in contours:
#                rect=cv2.minAreaRect(contour)
#                if min(rect[1])>=minDia and max(rect[1])<=maxDia and max(rect[1])/float(min(rect[1]))<=self._parameters['maxDiameterRatio']:
#                    goodContours.append(contour)
#
#            #Get contour mask
#            cMask=np.zeros(dist.shape)
#            cv2.drawContours(cMask,goodContours,-1,1,-1)
#            
#            pixelDists=dist[cMask.astype(bool)]
#            meanDist=np.mean(pixelDists,0)
#            stdDist=np.std(pixelDists,0)
#
#            #Clear up some variables for memory reasons
##            pixelDists=None
##            cMask=None
##            goodContours=None
##            contour=None
##            knownBg=None
##            sDist=None
##            dist=None
#            
#            #The final background is all pixels being more than two standard deviations
#            #away from the mean distance
#            bgMask=np.abs(dist-meanDist)>stdDist*2
#            
#            #I=np.sort(dist[cMask.astype(bool)],axis=None)    
#            #plt.plot(I)
#            #plt.show()
#            
#            #Calculate the contrast
#            contrast=np.sqrt(cv2.blur(np.square(dist),(3,3))-np.square(cv2.blur(dist,(3,3))))
#            I=np.sort(contrast[~knownBg],axis=None)
#            
#            #Sort the contrast and keep high contrast areas
#            minContrast=I[int(len(I)*(1-0.5))]
#            
#            #cv2.imshow('df',cv2.resize(cMask,(1000,1000)))
#            
#            #Return the final background mask
#            #self._backgroundMask=np.bitwise_or(knownBg,contrast<minContrast)
#
#            self._backgroundMask=np.bitwise_or(np.bitwise_or(np.bitwise_or(self._geometryMask>0,bgMask),self._clearMask),contrast<minContrast)      
#            
#            #First blur the image to reduce noise
#            #Ignore for now since it makes it more difficult to detect intersections
#            #when using only the contours
#            #img=cv2.blur(self._rawImage,(3,3))    
            img=self._rawImage
#            
            #Calculate the Euclidian distance to the background and colony colors
            #bgDist=np.sqrt(np.square(img[:,:,0]-self._bgColor[0])+np.square(img[:,:,1]-self._bgColor[1])+np.square(img[:,:,2]-self._bgColor[2]))
            colDist=np.sqrt(np.square(img[:,:,0]-self._colColor[0])+np.square(img[:,:,1]-self._colColor[1])+np.square(img[:,:,2]-self._colColor[2]))
            bgColDist=np.sqrt(np.square(self._bgColor[0]-self._colColor[0])+np.square(self._bgColor[1]-self._colColor[1])+np.square(self._bgColor[2]-self._colColor[2]))
            
            #self._backgroundMask=colDist>bgColDist
            
#               
#            
#            #contrast=np.sqrt(cv2.blur(np.square(gray),(3,3))-np.square(cv2.blur(gray,(3,3))))
#            #contrast = cv2.adaptiveThreshold(colDist.astype('uint8'),127,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,100,2)
#            #cv2.imshow('gray',cv2.resize(gray,(1000,1000)))            
#            #cv2.imshow('df',cv2.resize(cv2.normalize(contrast,0,255),(1000,1000)))
#            #I=np.sort(colDist[self._geometryMask==0],axis=None)    
#            #I=I[range(0,len(I),100)]
#            
#            #plt.plot(I)
#            #plt.show()
#            #self._backgroundMask=np.bitwise_or(gray<1.3*np.mean(gray),np.bitwise_or(self._clearMask,np.bitwise_or(self._geometryMask>0,contrast<np.mean(contrast))))
            self._backgroundMask=np.bitwise_or(self._clearMask,np.bitwise_or(self._geometryMask,colDist>(bgColDist*self._parameters['bgDetectFactor'])))
            #self._badMask=np.bitwise_and(np.bitwise_and(~self._clearMask,colDist>1.5*bgColDist),~self._geometryMask)
        else:
            self._backgroundMask=None
            
    def detectBlobs(self):
        """ Calculate connected components (blobs) based on the background mask.
        
            This factors in the allowed size span and generates listOfBlobs
            and listOfBadBlobs
        """
        
        #Attempt to recalculate the backgroundMask. This will return None if the
        #background and colony colors aren't set
        self._calculateBackground()
            
        if not self._backgroundMask is None:  
            #This is partly to typecast and partly because temp is changed by findContours
            temp=np.uint8(~self._backgroundMask.copy())
            contours, hierarchy = cv2.findContours(temp, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
            
            #Loop through and sort the blobs depending and if they are too large or too small
            self._listOfBlobs=[]
            self._listOfBadBlobs=[]
            self._listOfCircles=[]
            self._listOfAllCircles=[]

            #Get limits on blob diameter
            minDia=min([self._rawImage.shape[0],self._rawImage.shape[1]])*self._parameters['minColonySize']/self._parameters['plateDiameter']
            maxDia=min([self._rawImage.shape[0],self._rawImage.shape[1]])*self._parameters['maxBlobSize']/self._parameters['plateDiameter']
            
            for contour in contours:
                rect=cv2.minAreaRect(contour)
                if min(rect[1])>=minDia and max(rect[1])<=maxDia and max(rect[1])/float(min(rect[1]))<=self._parameters['maxDiameterRatio']:
                    #Calculate this only for potentially good blobs since it's a bit slow
                    if 4*np.pi*cv2.contourArea(contour)/np.square(cv2.arcLength(contour,True))>self._parameters['minBlobCircularity']:
                        #Calculate the mean color distance between the blob and colony color
                        startX=min(contour[:,0,1])
                        endX=max(contour[:,0,1])
                        startY=min(contour[:,0,0])
                        endY=max(contour[:,0,0])
                        blob=np.zeros((endX-startX+1,endY-startY+1))
                        cv2.drawContours(blob,[contour],-1,1,-1,offset=(-startY,-startX))                        
                        imageSection=self._rawImage[startX:endX+1,startY:endY+1,:]
                        blobColor=imageSection[blob.astype(bool),:]
                        bgColDist=np.sqrt(np.square(self._bgColor[0]-self._colColor[0])+np.square(self._bgColor[1]-self._colColor[1])+np.square(self._bgColor[2]-self._colColor[2]))
                        blobDist=np.sqrt(np.square(blobColor[:,0]-self._colColor[0])+np.square(blobColor[:,1]-self._colColor[1])+np.square(blobColor[:,2]-self._colColor[2]))
                        
                        if sum(blobDist<=(bgColDist*self._parameters['bgDetectFactor']))/float(len(blobDist))>self._parameters['blobDetectFactor']:
                            self._listOfBlobs.append(contour)
                        else:
                            self._listOfBadBlobs.append(contour)
                    else:
                        self._listOfBadBlobs.append(contour)                            
                else:
                    self._listOfBadBlobs.append(contour)

    def detectColonies(self):
        """ Fits colonies to the detected blobs.
        
            This occurs in the following steps:
            1) Compare contour to convex set. If no deviation go to 1a, otherwise to 2
                1a) Circular object, fit single circle and go to 3
            2) Calculate the local deviation maxima, split the contour according
               to those lines and fit one circle to each segment
            3) Disregard circles based on the following criteria:
                -outside of size limit
                -circles which are not at least minCircleCoverage within the blob
            4) Sort the circles based on how much of their area is covered by the blob.
               For each circle check if their contribution to the total
               blob coverage is larger than minCircleContrib. If not remove them
               and iterate.
        """
        if not self._listOfBlobs is None:
            minR=0.5*min([self._rawImage.shape[0],self._rawImage.shape[1]])*self._parameters['minColonySize']/self._parameters['plateDiameter']
            maxR=0.5*min([self._rawImage.shape[0],self._rawImage.shape[1]])*self._parameters['maxColonySize']/self._parameters['plateDiameter']
            
            #Loop through the blobs and try to fit circles            
            isBadBlob=np.zeros(len(self._listOfBlobs))
            for i in range(0,len(self._listOfBlobs)):
                #Note the shift in x/y because OpenCV has another format
                contour=self._listOfBlobs[i][:,0]
                startX=min(contour[:,1])
                endX=max(contour[:,1])
                startY=min(contour[:,0])
                endY=max(contour[:,0])
                circles=np.zeros((0,3))
                
                #Identify junctions between colonies as points having a maximal
                #distance to the convex hull
                
                #Returns the start and end indexes of a convexity defect,
                #the point furthest from the convex hull in the defect zone,
                #and the distance for that point. However, I need the distance
                #for all points in defect zones. I therefore iterate and remove
                #the worst points in each iteration
                dist=np.zeros(contour.shape[0])
                indexMap=np.arange(len(dist))
                tempContour=contour.copy()
                while True:
                    defects=cv2.convexityDefects(tempContour,cv2.convexHull(tempContour,returnPoints=False))
                    
                    if not defects is None:
                        #Remove all points which were detected and add their
                        #distances
                        I=defects[:,0,2]
                        toRemove=defects[:,0,3]/256.>=self._parameters['minHullDeviation']
                        if any(toRemove):
                            dist[indexMap[I[toRemove]]]=defects[toRemove,0,3]/256.                            
                            indexMap=np.delete(indexMap,I[toRemove])
                            tempContour=np.delete(tempContour,I[toRemove],0)
                        else:
                            break
                    else:
                        break
                
                #At this point "dist" will contain the distances to the convex hull
                #for all points above the limit
                #If there is only one local maxima then ignore it and perform
                #a simple fitting since it's most likely an artifact
                localMaxima=np.zeros(len(dist))
                
                #This ignores the first and last points, effectively assuming that
                #they are not local maxima. This is incorrect, but will probably
                #rarely have any effect
                for j in range(1,len(localMaxima)-1):
                    if dist[j]>dist[j-1] and dist[j]>dist[j+1]:
                        localMaxima[j]=dist[j]
                
                #It could be that there are several local maxima too close to
                #each other. Loop through and remove such points
                sI=np.argsort(localMaxima)[::-1]
                sI=sI[localMaxima[sI]>0]
                for j in range(0,len(sI)):
                    #If it has already been cleared by a better maxima then continue
                    if localMaxima[sI[j]]>0:
                        if sI[j]>0:
                            localMaxima[max(sI[j]-self._parameters['minSegmentDist'],0):sI[j]]=0
                        if sI[j]<len(localMaxima)-1:
                            localMaxima[sI[j]+1:min(sI[j]+1+self._parameters['minSegmentDist'],len(localMaxima))]=0
                    #Also add that hull deviation should be a function of blob diameter!
                    
                if sum(localMaxima>0)>1:
                    #Split the contour in segments at the localMaxima points.
                    #The first segment will also include the last to get a closed
                    #figure
                    mI=np.where(localMaxima)[0]
                    for j in range(0,len(mI)):
                        #Special case for the first point
                        if j==0:
                            points=np.vstack([contour[mI[-1]:],contour[0:mI[j]]])
                        else:    
                            points=contour[mI[j-1]:mI[j],:]
                        
                        center, radius=PlateImage.fitCircle(points)
                        circles=np.vstack((circles,[center[0],center[1],radius]))
                else:
                    #This means that it's a single colony and should be fitted using
                    #least squares
                    center, radius=PlateImage.fitCircle(contour)
                    circles=np.vstack((circles,[center[0],center[1],radius]))
                
                #At this stage we have a number of circles fitted to the blob.
                #The correct circles are probably among them, but there can also
                #be bad ones
                badVote=np.zeros(circles.shape[0],dtype=bool)
                #Loop through and make each check
                for j in range(0,len(badVote)):
                    #First check for size
                    if circles[j,2]<minR or circles[j,2]>maxR:
                        badVote[j]=True
                        continue
                        
                    #Then calculate the proportion of the circle which is outside
                    #of the blob. Get 100 points in the circle
                    n=10
                    C=2*(np.array(np.where(np.ones((n,n))))/(n-1.)-0.5)
                    C=C.T
                    I=np.where(np.sqrt(np.square(C[:,0])+np.square(C[:,1]))>1)
                    C=np.delete(C,I,0)
                    C=C*circles[j,2]
                    C[:,0]=C[:,0]+circles[j,0]
                    C[:,1]=C[:,1]+circles[j,1]
                    
                    #Get maximal number of allowed points outside the contour
                    maxOut=len(C)*(1-self._parameters['minCircleCoverage'])
                    counter=0
                    for p in C:
                        if cv2.pointPolygonTest(contour,(p[0],p[1]),False)<0:
                            counter=counter+1
                        if counter>=maxOut:
                            badVote[j]=True
                            break
                        
                #Remove all circles which recieved bad votes
                circles=np.delete(circles,np.where(badVote>0),0)
                
                #Add the circles before filtering
                self._listOfAllCircles.append(circles)
                
                #Calculate the proportion of each circle which is inside the contour
                coverage=np.zeros(circles.shape[0])
                n=10
                for j in range(0,len(coverage)):
                    C=2*(np.array(np.where(np.ones((n,n))))/(n-1.)-0.5)
                    C=C.T
                    I=np.where(np.sqrt(np.square(C[:,0])+np.square(C[:,1]))>1)
                    C=np.delete(C,I,0)
                    C=C*circles[j,2]
                    C[:,0]=C[:,0]+circles[j,0]
                    C[:,1]=C[:,1]+circles[j,1]
                    
                    for p in C:
                        if cv2.pointPolygonTest(contour,(p[0],p[1]),False)>=0:
                            coverage[j]=coverage[j]+1
                
                #For each circle, remove it and see if the effect on total coverage
                #is < minCircleContrib
                blob=np.zeros((endX-startX+1,endY-startY+1))
                cv2.drawContours(blob,[contour],-1,1,-1,offset=(-startY,-startX))
                
                #This is done in an iterative manner since the coverage will change
                #as circles are, potentially, removed
                removedCircle=True
                while removedCircle==True:
                    I=np.argsort(coverage)
                    removedCircle=False
                    after=0
                    for j in range(0,len(I)):
                        circleCoverage=np.zeros((endX-startX+1,endY-startY+1))
                        allOthers=np.delete(circles,I[j],0)
                                
                        for circle in allOthers:
                            cv2.circle(circleCoverage,(int(np.rint(circle[0])-startY),int(np.rint(circle[1])-startX)),int(np.rint(circle[2])),1,-1)
                        #Calculate the contribution from this circle
                        before=(circleCoverage*blob).sum()
                        cv2.circle(circleCoverage,(int(np.rint(circles[I[j],0])-startY),int(np.rint(circles[I[j],1])-startX)),int(np.rint(circles[I[j],2])),1,-1)
                        after=(circleCoverage*blob).sum()
                        if before>0:
                            if (after/before-1)<self._parameters['minCircleContrib']:
                                circles=np.delete(circles,I[j],0)                            
                                coverage=np.delete(coverage,I[j],0)                            
                                removedCircle=True
                                break
                            
                #At this point all circles which do not contribute at least 
                #minCircleContrib or which do not overlap at least minCircleCoverage
                #have been removed. However, it could still be the case that there
                #are almost overlapping circles which still contribute to the
                #overall coverage. This presents a problem since it's difficult
                #to know the order in which the circles should be removed (or
                #merged?). I choose here to remove smaller circles first, but
                #this may not be the best solution
                removedCircle=True
                while removedCircle==True:
                    I=np.argsort(circles[:,2])
                    removedCircle=False
                    for j in range(0,len(I)):
                        circleCoverage=np.zeros((endX-startX+1,endY-startY+1))
                        cv2.circle(circleCoverage,(int(np.rint(circles[I[j],0])-startY),int(np.rint(circles[I[j],1])-startX)),int(np.rint(circles[I[j],2])),1,-1)
                        before=circleCoverage.sum()
                        allOthers=np.delete(circles,I[j],0)
                                
                        for circle in allOthers:
                            cv2.circle(circleCoverage,(int(np.rint(circle[0])-startY),int(np.rint(circle[1])-startX)),int(np.rint(circle[2])),0,-1)
                        
                        #Calculate the overlap with the other circles
                        after=circleCoverage.sum()
                        if ((before-after)/before)>self._parameters['maxCircleOverlap']:
                            circles=np.delete(circles,I[j],0)                            
                            removedCircle=True
                            break
                            
                #Exclude blobs which are not covered enough
                if len(circles)>0:
                    circleCoverage=np.zeros((endX-startX+1,endY-startY+1))
                    for circle in circles:
                        cv2.circle(circleCoverage,(int(np.rint(circle[0])-startY),int(np.rint(circle[1])-startX)),int(np.rint(circle[2])),1,-1)          
                            
                    if float(circleCoverage.sum())/blob.sum()<self._parameters['minBlobCoverage']:
                        circles=np.zeros((0,3))                
                    self._listOfCircles.append(circles)
                    
                    #Lastly, check if the mean color of the area covered by the circles is similar
                    #enough to the colony color. This is because if the blob is "O" or "C" shaped
                    #with a hollow interior it would pass all the previous tests (maybe not circularity
                    #if it's strict). This can be the case if there is text or marks on the plate. Such blobs
                    #should be moved to listOfBadBlobs
                    imageSection=self._rawImage[startX:endX+1,startY:endY+1,:]
                    blobColor=imageSection[circleCoverage.astype(bool),:].mean(0)
                        
                    bgColDist=np.sqrt(np.square(self._bgColor[0]-self._colColor[0])+np.square(self._bgColor[1]-self._colColor[1])+np.square(self._bgColor[2]-self._colColor[2]))
                    blobDist=np.sqrt(np.square(blobColor[0]-self._colColor[0])+np.square(blobColor[1]-self._colColor[1])+np.square(blobColor[2]-self._colColor[2]))
                    if blobDist>(bgColDist*self._parameters['bgDetectFactor']):
                        isBadBlob[i]=True    
                else:
                    isBadBlob[i]=True
                    self._listOfCircles.append(np.zeros((0,3)))
                    
                    #THIS IS NEVER USED! The blob should be reclassified as "bad" but
                    #now it will just be a yellow blob instead. Should be fixed.
                    
    #Static methods
    @staticmethod
    def fitCircle(points,weights=None):
        """ Fit a circle to a number of points using least squares.
    
            points:     Nx2 array with x,y values
            weights:    weight for each point (opt, default is all equal)
            
            returns
                array with the center
                radius
        """
        
        if weights is None:
            weights=np.ones((points.shape[0]))
        
        A=np.matrix(np.hstack((points,np.ones((points.shape[0],1)))))
        y=np.matrix(np.square(points).sum(1))
        
        eye=np.matrix(np.diag(weights))
        
        m1=A.T*eye*A
        m2=A.T*eye*y.T
        ppp=np.linalg.lstsq(m1,m2)[0]
        
        center=ppp[0:2]*0.5
        radius = np.sqrt(ppp[2]+center.T*center)
        
        return (np.array(center)[:,0],np.asscalar(radius))