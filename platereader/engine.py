from plateimage import PlateImage
from gui import GUI
#import Tkinter #This should not be required, but is needed for Py-Installer
#import FileDialog #This should not be required, but is needed for Py-Installer

class PlateReader(object):
    """ Main controller for the PlateReader software.
        
        This serves as the interface between PlateImage and GUI
    """
    #The PlateImage
    _plateImage=None
        
    #The GUI. This will also display the GUI
    _gui=None
    
    def __init__(self):
        #The PlateImage
        self._plateImage=PlateImage()
 
    #Properties       
    @property
    def fileName(self):
        """ The current file name.
            
            This is also retriavable via getObjectsForReporting, but that
            is rather computationally demanding
        """
        return self._plateImage.fileName
        
    def getImagesForReporting(self):
        """ Get images used for generating reports.
        
            This is used when exporting to RTF in the GUI
        """

        #This should return the processed image with markup. Therefore save the current setting
        #and recalculate if needed
        usingMarkup=self._plateImage.getParameter('showMarkup')

        self._plateImage.setParameter('showMarkup',True)
        
        image=self._plateImage.image
        self._plateImage.setParameter('showMarkup',usingMarkup)
        
        return (self._plateImage.rawImage,image,self._plateImage.histogram)
        
    def getStatistics(self):
        """ Get a dictionary with the statistics."""
        return self._plateImage.statistics        
        
    def showGUI(self):
        """ Load and show the GUI."""
        self._gui=GUI(self)
        
        #If anything has been loaded in the plateImage object then update the
        #GUI accordingly
        rImage=self._plateImage.rawImage
        if not rImage is None:
            self._gui.setOriginalImage(rImage)  
            #Start with controls enabled if there is an image loaded
            self._gui.changeEnabled(True,[])
            
        wImage=self._plateImage.image
        if not wImage is None:
            self._gui.setWorkingImage(wImage)
        
        if not self._plateImage.histogram is None:
            self._gui.setHistogram(self._plateImage.histogram)
            
        if not self._plateImage.statistics is None:
            self._gui.printStatistics(self._plateImage.statistics)    
        
        #Show GUI and give control over to the main GUI loop
        self._gui.show()
    
    def setColColor(self,points):
        """ Set the colony color and update the feature detection."""
        
        #Note that this doesn't round the values. Use np.rint if required
        #Note the shift in x/y
        color=self._plateImage.rawImage[[points.astype(int)[:,1],points.astype(int)[:,0]]]        
        self._plateImage.colColor=color.mean(0)
        self._plateImage.detectBlobs()
        self._plateImage.detectColonies()
        
        #Update the view
        if not self._gui is None:       
            self._gui.setWorkingImage(self._plateImage.image)
            self._gui.setHistogram(self._plateImage.histogram)
            self._gui.printStatistics(self._plateImage.statistics)    

    def setMaskUnmask(self,setToMask):
        """ Set whether to mask or unmask areas when using the manual drawing tools."""
        
        self._plateImage.maskWhenDrawing=setToMask
    
    def maskUnmaskArea(self,points,geometry):
        """ Mask or unmask an area. 
        
            geometry: 'line', 'rect' or 'circle'
        """
        
        self._plateImage.maskUnmaskArea(points,geometry)
        
        #This repeats all calculations. This is wasteful, but I keep it for now
        self._plateImage.detectBlobs()
        self._plateImage.detectColonies()
        
        #Display the new image
        if not self._gui is None:
            self._gui.setWorkingImage(self._plateImage.image)
            self._gui.setHistogram(self._plateImage.histogram)
            self._gui.printStatistics(self._plateImage.statistics)
        
    def setBgColor(self,points):
        """ Set the background color and update the feature detection."""
        
        #Note that this doesn't round the values. Use np.rint if required
        #Note the shift in w/y
        color=self._plateImage.rawImage[[points.astype(int)[:,1],points.astype(int)[:,0]]]        
        self._plateImage.bgColor=color.mean(0)
        self._plateImage.detectBlobs()
        self._plateImage.detectColonies()
        
        #Update the view
        if not self._gui is None:
            self._gui.setWorkingImage(self._plateImage.image)
            self._gui.setHistogram(self._plateImage.histogram)
            self._gui.printStatistics(self._plateImage.statistics)

    def setParameter(self,name,value):
        """ Set a parameter."""
        self._plateImage.setParameter(name,value)
        
        if not name is 'showMarkup':
            self._plateImage.detectBlobs()
            self._plateImage.detectColonies()
        
        #Display the new image
        if not self._gui is None:
            self._gui.setWorkingImage(self._plateImage.image)
            self._gui.setHistogram(self._plateImage.histogram)
            self._gui.printStatistics(self._plateImage.statistics)
            
    def getParameter(self,name):
        """ Get a parameter."""
        return self._plateImage.getParameter(name)
        
    def getAllParameterNames(self):
        """ Get all parameter names."""
        return self._plateImage.getAllParameterNames()
        
    def fitPlateGeometry(self,points):
        """ Fit plate geometry in the PlateImage object and update the GUI accordingly."""

        self._plateImage.fitPlateGeometry(points)
        
        #This is to prevent having to select new colors if loading a new image after
        #they have been selected
        self._plateImage.detectBlobs()
        self._plateImage.detectColonies()
        
        #Display the new image
        if not self._gui is None:
            self._gui.setWorkingImage(self._plateImage.image)
            self._gui.setOriginalImage(self._plateImage.rawImage)
            self._gui.setHistogram(self._plateImage.histogram)
            self._gui.printStatistics(self._plateImage.statistics)
        
    def loadImage(self,fileName=None):
        """ Load an image file.
        
            Sets both the working and original images
        """
        
        try:
            if not fileName is None:                
                self._plateImage.loadImage(fileName)
            else:   
                #This reloads the current image
                self._plateImage.loadImage()
        except:
            if not self._gui is None:
                if not fileName is None:                
                    self._gui.setStatusMessage('Could not load ' + fileName)
                else:   
                    self._gui.setStatusMessage('Could not reload image')
            return
        
        if not self._gui is None:
            self._gui.setWorkingImage(self._plateImage.rawImage)
            self._gui.setOriginalImage(self._plateImage.rawImage)
            self._gui.setHistogram(self._plateImage.histogram)
            self._gui.printStatistics(self._plateImage.statistics)
            if not fileName is None:                
                self._gui.setStatusMessage('Successfully loaded ' + fileName)
            else:   
                self._gui.setStatusMessage('Successfully reloaded image')
            
if __name__ == "__main__":
    """ Run the program."""
    PlateReader().showGUI()