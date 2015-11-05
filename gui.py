# -*- coding: utf-8 -*-
import gtk
import numpy as np
import os
import subprocess
import PyRTF
from datetime import datetime
import tempfile
import cv2
import my

class GUI(object):
    """ The GUI."""
    
    #The controlling object
    _controller=None
    
    #Widgets which will be referred to in the callbacks
    _w={}
    
    #This keeps track of x and y positions for clicks on images
    _clickCounter=None
    
    #Listeners for different types of click events
    _detectEdgeListener=False
    _detectBgListener=False
    _detectColListener=False
    _detectRectListener=False
    _detectCircleListener=False
    _detectLineListener=False
    _nPointsToGet=None
    
    #The dimensionality of the images displayed. This is used for scaling
    _displayDim=(800,600)
    
    #This assumes that original image and working image always have the same dimensions
    _originalDim=None
    
    #List of batch image names
    _batchImages=None
    
    #Counter for current batch item
    _batchIndex=None
    
    #Batch output dir
    _batchOutputDir=None
    
    #List of batch descriptions and list of list of sizes
    _listOfDescriptions=[]
    _listOfSizes=[]
    
    def __init__(self,controller):
        """Construct the GUI from the Glade file."""
        
        self._controller=controller #This contains the controlling object
        builder=gtk.Builder()
        builder.add_from_file(my.resource_path('gui.glade'))
        
        #Load all widgets which will be referred to in the callbacks       
        self._w['window']=builder.get_object('window')
        self._w['statusbar']=builder.get_object('statusbar')
        self._w['workingImage']=builder.get_object('workingImage')
        self._w['originalImage']=builder.get_object('originalImage')
        self._w['histImage']=builder.get_object('histImage')
        self._w['selectEdge']=builder.get_object('selectEdge')
        self._w['selectBg']=builder.get_object('selectBg')
        self._w['selectCol']=builder.get_object('selectCol')
        self._w['drawRect']=builder.get_object('drawRect')
        self._w['drawLine']=builder.get_object('drawLine')
        self._w['drawCircle']=builder.get_object('drawCircle')
        self._w['setMaskUnmask']=builder.get_object('setMaskUnmask')
        self._w['drawFrame']=builder.get_object('drawFrame')
        self._w['plateDescription']=builder.get_object('plateDescription')
        self._w['batchFrame']=builder.get_object('batchFrame')
        self._w['resultsText']=builder.get_object('resultsText')
        self._w['batchNext']=builder.get_object('batchNext')
        self._w['batchLabel']=builder.get_object('batchLabel')
        self._w['batch']=builder.get_object('batch')
        self._w['batchList']=builder.get_object('batchList')
        self._w['batchView']=builder.get_object('batchView')
        self._w['batchAdd']=builder.get_object('batchAdd')
        self._w['batchDone']=builder.get_object('batchDone')
        self._w['menuOpen']=builder.get_object('menuOpen')
        self._w['menuBatch']=builder.get_object('menuBatch')
        self._w['batchDir']=builder.get_object('batchDir')
        
        #Also load all parameters since they share names with widgets
        for p in self._controller.getAllParameterNames():
            self._w[p]=builder.get_object(p)
        
        #When this is called without argument an empty white image is generated
        self.setWorkingImage()
        self.setOriginalImage()
        builder.connect_signals(self)
        
        #Fill default parameter values
        for p in self._controller.getAllParameterNames():
            if hasattr(self._w[p],'set_text'):
                self._w[p].set_text(str(self._controller.getParameter(p)))
        
        #Start with controls in disabled mode
        self.changeEnabled(False,['window'])
        
    def show(self):
        """ Show the GUI and enter main loop."""
        
        self._w['window'].show()
        gtk.main()
        
    #These are all the callback functions
    def on_parameter_change(self, widget, data=None):
        """ Set parameters.
        
            This is called for all parameters with text boxes.
        """
        
        #This is a bug that it can't be retrieved in a normal way
        name=gtk.Buildable.get_name(widget)

        #First check that the text can be converted to a value        
        try:
            self._controller.setParameter(name,float(widget.get_text()))
        except ValueError:
            #Set the value to the default instead
            widget.set_text(str(self._controller.getParameter(name)))
        
    def on_showMarkup_toggled(self, widget, data=None):
        """ Toggle whether to show markings."""
        
        self._controller.setParameter('showMarkup',widget.get_active())

    def on_setMaskUnmask_toggled(self, widget, data=None):
        """ Set whether to mask or unmask areas when using the manual drawing tools."""
        
        self._controller.setMaskUnmask(widget.get_active())
        
    def on_selectCol_clicked(self, widget, data=None):
        """ Select five points from colonies."""
        
        #Inactivate the rest of the GUI while doing this
        self.changeEnabled(False)
        
        #Reset the click counter
        self._clickCounter=None
        
        #Start the listener
        self._detectColListener=True
        self._nPointsToGet=5
        
    def on_selectBg_clicked(self, widget, data=None):
        """ Select five points from the background."""
        
        #Inactivate the rest of the GUI while doing this
        self.changeEnabled(False)
        
        #Reset the click counter
        self._clickCounter=None
        
        #Start the listener
        self._detectBgListener=True
        self._nPointsToGet=5
        
    def on_drawCircle_clicked(self, widget, data=None):
        """ Select two points which define a circle to mask/unmask."""
        
        #Inactivate the rest of the GUI while doing this
        self.changeEnabled(False)
        
        #Reset the click counter
        self._clickCounter=None
        
        #Start the listener
        self._detectCircleListener=True
        self._nPointsToGet=2
        
    def on_drawLine_clicked(self, widget, data=None):
        """ Select two points which define a line to mask/unmask."""
        
        #Inactivate the rest of the GUI while doing this
        self.changeEnabled(False)
        
        #Reset the click counter
        self._clickCounter=None
        
        #Start the listener
        self._detectLineListener=True
        self._nPointsToGet=2
        
    def on_drawRect_clicked(self, widget, data=None):
        """ Select two points which define a rectangle to mask/unmask."""
        
        #Inactivate the rest of the GUI while doing this
        self.changeEnabled(False)
        
        #Reset the click counter
        self._clickCounter=None
        
        #Start the listener
        self._detectRectListener=True
        self._nPointsToGet=2
        
    def on_selectEdge_clicked(self, widget, data=None):
        """ Select five points on the edge of the plate in order to fit ellipse."""
        
        #Inactivate the rest of the GUI while doing this
        self.changeEnabled(False)
        
        #Reset the click counter
        self._clickCounter=None
        
        #Start the listener
        self._detectEdgeListener=True
        self._nPointsToGet=5
        
    def on_workingImageEventBox_button_press_event(self, widget, data=None):
        """ Keep track of position and number of clicks on the working image area."""
        if self._detectEdgeListener or self._detectBgListener or self._detectColListener or self._detectRectListener or self._detectCircleListener or self._detectLineListener:
            #Get the dimensions of the event box
            ebDim=widget.get_allocation()

            #Add the new point
            if self._clickCounter is None:
                self._clickCounter=np.array([[data.x,data.y]],)
            else:
                self._clickCounter=np.vstack([self._clickCounter,[[data.x,data.y],]])
            
            #Check if it has added the correct number of points
            if self._clickCounter.shape[0] is self._nPointsToGet:
                #Reenable the GUI
                self.changeEnabled(True)
                
                #The points are relative to the scaled image and have to be rescaled
                points=self._clickCounter

                #Note the shift in dimensions. At the moment the image is not rescaled if
                #EventBox is. Instead it will stay centered
                xPadding=(ebDim.width-self._displayDim[0])/2
                yPadding=(ebDim.height-self._displayDim[1])/2
                points[:,0]=(points[:,0]-xPadding)*float(self._originalDim[1])/self._displayDim[0] 
                points[:,1]=(points[:,1]-yPadding)*float(self._originalDim[0])/self._displayDim[1]
                
                #Give the detected points to the controller
                if self._detectEdgeListener:
                    self._controller.fitPlateGeometry(points)
                if self._detectBgListener:
                    self._controller.setBgColor(points)
                if self._detectColListener:
                    self._controller.setColColor(points)
                if self._detectRectListener:
                    self._controller.maskUnmaskArea(points,'rect')
                if self._detectCircleListener:
                    self._controller.maskUnmaskArea(points,'circle')
                if self._detectLineListener:
                    self._controller.maskUnmaskArea(points,'line')
                
                #Reset the listeners
                self._detectEdgeListener=False
                self._detectBgListener=False
                self._detectColListener=False
                self._detectRectListener=False
                self._detectCircleListener=False
                self._detectLineListener=False
        
    def on_menuOpen_activate(self, widget, data=None):
        """ Select and open an image file."""
        
        fileName=None
        
        chooser=gtk.FileChooserDialog("Open File...", self._w['window'],
                                      gtk.FILE_CHOOSER_ACTION_OPEN,
                                      (gtk.STOCK_CANCEL, gtk.RESPONSE_CANCEL, 
                                      gtk.STOCK_OPEN, gtk.RESPONSE_OK))
                                      
        filter=gtk.FileFilter()
        filter.set_name("JPG files")
        filter.add_pattern("*.jpg")
        chooser.add_filter(filter)
                
        response = chooser.run()
        if response == gtk.RESPONSE_OK:
            fileName = chooser.get_filename()
            chooser.destroy()
        
        if not fileName is None:
            self._controller.loadImage(fileName)
            self.changeEnabled(True,[])     
            
    def on_menuBatch_activate(self, widget, data=None):
        """ Open the batch dialog window."""
        
        self._w['batch'].show()
        
        renderer=gtk.CellRendererText()
        column=gtk.TreeViewColumn("Title", renderer, text=0)
        self._w['batchView'].append_column(column)

    def on_menuReload_activate(self, widget, data=None):
        """ Reload the current image file."""

        #Calling this without arguments uses the current image
        self._controller.loadImage()
        self.changeEnabled(True,[])
            
    def on_menuLoadSample_activate(self, widget, data=None):
        """ Load a sample image."""
     
        self._controller.loadImage(my.resource_path('sample.jpg'))
        self.changeEnabled(True,[])       
        
    def on_menuAbout_activate(self, widget, data=None):
        """ Display information about the software."""

        about = gtk.AboutDialog()
        about.set_program_name("PlateReader")
        about.set_version("0.14")
        about.set_copyright("(c) Novo Nordisk A/S")
        about.set_comments("This software is not to be distributed outside of Novo Nordisk. Please contact Rasmus Ågren (RAAG) for any questions or comments.")
        about.run()
        about.destroy()    
        
    def on_menuExportRTF_activate(self, widget, data=None):
        """ Export the results as an RTF file."""
        
        #Get the results structure
        statistics=self._controller.getStatistics()
        
        if not statistics is None:
            
            rawImage, image, histogram=self._controller.getImagesForReporting()
            
            #Open save as dialog
            chooser=gtk.FileChooserDialog("Save File...", self._w['window'],
                                      gtk.FILE_CHOOSER_ACTION_SAVE,
                                      (gtk.STOCK_CANCEL, gtk.RESPONSE_CANCEL, 
                                      gtk.STOCK_SAVE, gtk.RESPONSE_OK))
                                      
            filter=gtk.FileFilter()
            filter.set_name("RTF files")
            filter.add_pattern("*.rtf")
            chooser.add_filter(filter)
            
            #Use default name
            base=os.path.basename(statistics['fileName'])
            suggest=os.path.splitext(base)[0]
            
            chooser.set_current_name(suggest+'.rtf')
        
            chooser.run()
            fileName=chooser.get_filename()
            chooser.destroy()
            
            if fileName is None:
                return
                
            #Append .rtf if needed
            if not fileName[-4:].upper()=='.RTF':
                fileName=fileName+'.rtf'
                
            #Save the file
            self.saveRTF(fileName, rawImage, image, histogram, statistics)
            
    def on_menuManual_activate(self, widget, data=None):
        """ Open the manual in the default PDF reader."""
        subprocess.Popen(my.resource_path('manual.pdf'),shell=True)
        
    def on_window_destroy(self, widget, data=None):
        """ Quit the main GUI loop."""
        
        gtk.main_quit()
    
    def on_menuQuit_activate(self, widget, data=None):
        """ Close the window."""
        self._w['window'].destroy()
        
    #These are the methods involving the load batch window
    def on_batchAdd_clicked(self, widget, data=None):
        """ Add a batch of images."""
        fileNames=None
        chooser=gtk.FileChooserDialog("Open Files...", self._w['batch'],
                                      gtk.FILE_CHOOSER_ACTION_OPEN,
                                      (gtk.STOCK_CANCEL, gtk.RESPONSE_CANCEL, 
                                      gtk.STOCK_OPEN, gtk.RESPONSE_OK))
        chooser.set_select_multiple(True)
        filter=gtk.FileFilter()
        filter.set_name("JPG files")
        filter.add_pattern("*.jpg")
        chooser.add_filter(filter)
                
        response = chooser.run()
        if response == gtk.RESPONSE_OK:
            fileNames = chooser.get_filenames()
            chooser.destroy()
        
        if not fileNames is None:
            #Remove items which don't end in jpg
            noFolders=[x for x in fileNames if x[-4:].upper()=='.JPG']
            
            if len(noFolders)>0:
                self._w['batchList'].clear()
                
                if not self._batchImages is None:
                    self._batchImages=sorted(set(self._batchImages+noFolders))
                else:
                    self._batchImages=sorted(set(noFolders))
                
                for name in self._batchImages:
                    self._w['batchList'].append([name])
            
    def on_batchDone_clicked(self, widget, data=None):
        """ Update the view if there are loaded batch images."""
        
        if not self._batchImages is None and not self._batchOutputDir is None:
            self._batchIndex=0
            self._controller.loadImage(self._batchImages[0])
        
            #Set relevant GUI stuff
            self._w['menuOpen'].set_sensitive(False)
            self._w['menuBatch'].set_sensitive(False)
            self._w['batchFrame'].set_visible(True)
            self._w['batchLabel'].set_text("Image 1/"+str(len(self._batchImages)))
            self.changeEnabled(True,[])
        
            #Close the window
            self._w['batch'].destroy()
        else:
            self._w['batch'].hide()    
        
    def on_batchNext_clicked(self, widget, data=None):
        """ Load next image in batch."""
        
        #Save the description and sizes for the previous image
        desc=self._w["plateDescription"].get_text()
        if len(desc)==0:
            desc=self._controller.fileName
        
        stats=self._controller.getStatistics()
        if not stats is None:
            self._listOfDescriptions=self._listOfDescriptions+[desc]            
            self._listOfSizes=self._listOfSizes+[stats['listOfSizes']]
            
            #Save RTF file
            rawImage, image, histogram=self._controller.getImagesForReporting()
            fileName=os.path.splitext(os.path.basename(self._controller.fileName))[0]
            
            self.saveRTF(os.path.join(self._batchOutputDir,fileName + ".rtf"),rawImage,image,histogram,stats)
                
        if self._batchIndex<len(self._batchImages)-1:            
            self._batchIndex=self._batchIndex+1
            self._controller.loadImage(self._batchImages[self._batchIndex])
            self.changeEnabled(True,[])
            self._w['batchLabel'].set_text("Image " + str(self._batchIndex+1) + "/" +str(len(self._batchImages)))

    def on_batchSave_clicked(self, widget, data=None):
        """ Save CSV file with colony sizes."""

        desc=self._listOfDescriptions
        sizes=self._listOfSizes
        nDesc=len(self._listOfDescriptions)
        maxSize=len(max(self._listOfSizes,key=len))

        #Sort by size
        for i in range(0,len(sizes)):
            sizes[i]=np.sort(sizes[i])
        
        with open(os.path.join(self._batchOutputDir,"batch.txt"), "w") as csv:
            
            #Prints captions
            for i in range(0,nDesc):
                if i>0:
                    csv.write("\t")                    
                csv.write(desc[i])                
            csv.write("\n")            
            
            for i in range(0,maxSize):
                if i>0:
                    csv.write("\n")
                    
                for j in range(0,nDesc):
                    if j>0:
                        csv.write("\t")
                    
                    if len(sizes[j])>i:
                        csv.write(str(sizes[j][i]))
        
    def on_batchDirBrowse_clicked(self, widget, data=None):
        """ Set output directory for batch processing."""
        
        dirName=None
        chooser=gtk.FileChooserDialog("Select output directory...", self._w['batch'],
                                      gtk.FILE_CHOOSER_ACTION_SELECT_FOLDER,
                                      (gtk.STOCK_CANCEL, gtk.RESPONSE_CANCEL, 
                                      gtk.STOCK_OPEN, gtk.RESPONSE_OK))
                
        response = chooser.run()
        if response == gtk.RESPONSE_OK:
            dirName = chooser.get_filename()
            chooser.destroy()
        
        if not dirName is None:
            self._w['batchDir'].set_text(dirName)
            self._batchOutputDir=dirName
        
    #These are the public methods for affecting the GUI 
    def setStatusMessage(self,msg):
        """ Set a status bar message."""
        
        #There is a stack which should be dealt with better
        self._w['statusbar'].push(self._w['statusbar'].get_context_id('PlateReader'),msg)
        
    def setOriginalImage(self,image=None):
        """ Set the image in the "Original image" tab."""
        if image is None:
            img_pixbuf=gtk.gdk.Pixbuf(gtk.gdk.COLORSPACE_RGB, False, 8, self._displayDim[0], self._displayDim[1])
            img_pixbuf.fill(0xffffffff) #Fill to white
        else:
            img_pixbuf=gtk.gdk.pixbuf_new_from_array(image, gtk.gdk.COLORSPACE_RGB, 8)
            #Scale the image to be 800x600
            img_pixbuf=img_pixbuf.scale_simple(self._displayDim[0],self._displayDim[1],gtk.gdk.INTERP_BILINEAR)
            #Keep track of original dimensions            
            self._originalDim=image.shape
            
        self._w['originalImage'].set_from_pixbuf(img_pixbuf)
        self._w['originalImage'].show()
        
    def setWorkingImage(self,image=None):
        """ Set the image in the "Working image" tab."""
        if image is None:
            img_pixbuf=gtk.gdk.Pixbuf(gtk.gdk.COLORSPACE_RGB, False, 8, self._displayDim[0], self._displayDim[1])
            img_pixbuf.fill(0xffffffff) #Fill to white
        else:
            img_pixbuf=gtk.gdk.pixbuf_new_from_array(image, gtk.gdk.COLORSPACE_RGB, 8)
            #Scale the image to be 800x600
            img_pixbuf=img_pixbuf.scale_simple(self._displayDim[0],self._displayDim[1],gtk.gdk.INTERP_BILINEAR)
            #Keep track of original dimensions            
            self._originalDim=image.shape
            
        self._w['workingImage'].set_from_pixbuf(img_pixbuf)
        self._w['workingImage'].show()
        
    def setHistogram(self,image=None):
        """ Set the histogram image."""
        if image is None:
            img_pixbuf=gtk.gdk.Pixbuf(gtk.gdk.COLORSPACE_RGB, False, 8, 560, 420)
            img_pixbuf.fill(0xffffffff) #Fill to white
        else:
            img_pixbuf=gtk.gdk.pixbuf_new_from_array(image, gtk.gdk.COLORSPACE_RGB, 8)
            #Scale the image to be 560x420
            img_pixbuf=img_pixbuf.scale_simple(560,420,gtk.gdk.INTERP_BILINEAR)
            
        self._w['histImage'].set_from_pixbuf(img_pixbuf)
        self._w['histImage'].show()
        
    def printStatistics(self,statistics):
        """ Print statistics about the fittings."""
        
        if not statistics['fileName'] is None:
            toPrint='File name:\n' + statistics['fileName'] + '\n\n'
            toPrint=toPrint + 'Number of colonies: ' + "%0.0f" % statistics['numberOfColonies'] + '\n'
            toPrint=toPrint + 'Mean colony diameter (mm): ' + "%0.2f" % statistics['meanColonyDiameter'] + '\n'
            toPrint=toPrint + 'Mean colony area (mm^2): ' + "%0.2f" % statistics['meanColonyArea'] + '\n'
            toPrint=toPrint + 'Median colony diameter (mm): ' + "%0.2f" % statistics['medianColonyDiameter'] + '\n'
            toPrint=toPrint + 'Median colony area (mm^2): ' + "%0.2f" % statistics['medianColonyArea'] + '\n'
            toPrint=toPrint + 'Std of colony diameter (mm): ' + "%0.2f" % statistics['stdOfColonies'] + '\n\n'
            toPrint=toPrint + 'Number of blobs: ' + "%0.0f" % statistics['numberOfBlobs'] + '\n'
            toPrint=toPrint + 'Number of colonies per blob: ' + "%0.2f" % statistics['coloniesPerBlob'] + '\n'
            toPrint=toPrint + 'Total blob area (mm^2): ' + "%0.2f" % statistics['totalBlobArea'] + '\n'
            toPrint=toPrint + 'Total colony area (mm^2): ' + "%0.2f" % statistics['totalCircleArea'] + '\n'
            toPrint=toPrint + 'Total blob fit: ' + "%0.2f" % statistics['blobFit'] + '\n\n'
            toPrint=toPrint + 'Number of bad blobs: ' + "%0.0f" % statistics['numberOfBadBlobs'] + '\n'
            toPrint=toPrint + 'Area fraction of bad blobs: ' + "%0.2f" % statistics['fractionBadBlobArea'] + '\n'
            toPrint=toPrint + 'Adjusted number of colonies: ' + "%0.0f" % statistics['adjNumberOfColonies']
            
            self._w['resultsText'].get_buffer().set_text(toPrint)
        
    def changeEnabled(self,enable,exclude=['window','originalImage','workingImage','bgColor','colColor']):
        """ Enable or disable all GUI components.

            enable:     true if the components should be enabled
            exclude:    list with names to exclude
        """
        
        if exclude is None:
            keys=self._w.keys
        else:
            keys=set(self._w.keys()) - set(exclude)
            
        #Exclude components which cannot be enabled/disabled or which belong to other windows
        keys=set(keys)-set(['batch', 'batchList', 'batchView', 'batchDone', 'batchAdd', 'menuOpen', 'menuBatch', 'batchDir', 'batchDirBrowse'])
        for key in keys:
            self._w[key].set_sensitive(enable)
            
    def saveRTF(self, fileName, rawImage, image, histogram, statistics):
        """ Save an RTF file."""
        
        #Create the RTF object
        doc=PyRTF.Document()
        ss=doc.StyleSheet
        section=PyRTF.Section()
        doc.Sections.append(section)
        
        e=PyRTF.Paragraph(ss.ParagraphStyles.Normal) #Blank line
        e.append('')
        
        p=PyRTF.Paragraph(ss.ParagraphStyles.Heading1)
        p.append('PlateReader analysis')
        section.append(p)
        section.append(e)

        #Print the table
        #thin_edge=PyRTF.BorderPS(width=20, style=PyRTF.BorderPS.SINGLE)
        #thin_frame=PyRTF.FramePS(thin_edge, thin_edge, thin_edge, thin_edge)

        #Get date and time
        i=datetime.now()
        
        #This is a bug that it can't be retrieved in a normal way
        #descName=gtk.Buildable.get_name('plateDescription')
        
        table =PyRTF.Table(PyRTF.TabPS.DEFAULT_WIDTH*4,
				   PyRTF.TabPS.DEFAULT_WIDTH*10)
        names=('File name','Description','Number of colonies','Adj. number of colonies','Mean colony diameter','Date')
        values=(statistics['fileName'].replace('\\','/'),self._w['plateDescription'].get_text(),"%0.0f" % statistics['numberOfColonies'],"%0.0f" % (statistics['numberOfColonies']/(1-statistics['fractionBadBlobArea'])),"%0.2f" % statistics['meanColonyDiameter'],i.strftime('%Y-%m-%d %H:%M'))           
        for i in range(0,6):
            c1=PyRTF.Cell(PyRTF.Paragraph(PyRTF.TEXT(names[i],bold=True)))
            c2=PyRTF.Cell(PyRTF.Paragraph(values[i]))
            table.AddRow(c1,c2)

        section.append(table)
        section.append(e)
        
        #Save the raw image as a tempory jpg image and then imclude it
        tf=tempfile.NamedTemporaryFile()
        tempName=tf.name+'.png'
        tf.close
        
        table =PyRTF.Table(PyRTF.TabPS.DEFAULT_WIDTH*8,
				   PyRTF.TabPS.DEFAULT_WIDTH*6)
   
        #Convert from RGB to BGR since using OpenCV
        toShow=cv2.resize(rawImage,(270,270),interpolation=cv2.INTER_AREA)
        toShow=np.dstack((toShow[:,:,2],toShow[:,:,1],toShow[:,:,0]))
        
        cv2.imwrite(tempName,toShow)
        c1=PyRTF.Cell(PyRTF.Paragraph(PyRTF.Image(tempName)))
        c2=PyRTF.Cell(PyRTF.Paragraph(PyRTF.TEXT('Original image',bold=True)))
        table.AddRow(c1,c2)
        
        #Convert from RGB to BGR since using OpenCV
        toShow=cv2.resize(image,(270,270),interpolation=cv2.INTER_AREA)
        toShow=np.dstack((toShow[:,:,2],toShow[:,:,1],toShow[:,:,0]))
        cv2.imwrite(tempName,toShow)
        
        c1=PyRTF.Cell(PyRTF.Paragraph(PyRTF.Image(tempName)))
        c2=PyRTF.Cell(PyRTF.Paragraph(PyRTF.TEXT('Processed image',bold=True)))
        table.AddRow(c1,c2)

        section.append(table)
        
        #Also do this for the histogram which will be used later
        if not histogram is None:
            toShow=cv2.resize(histogram,(500,375),interpolation=cv2.INTER_AREA)
            toShow=np.dstack((toShow[:,:,2],toShow[:,:,1],toShow[:,:,0]))
            cv2.imwrite(tempName,toShow)
            histImage=PyRTF.Image(tempName)
        
            os.remove(tempName)
        
        #Print detailed results
        p=PyRTF.Paragraph(ss.ParagraphStyles.Heading1)
        p.append('Results')
        section.append(p)
        section.append(e)
        
        table =PyRTF.Table(PyRTF.TabPS.DEFAULT_WIDTH*5,
				   PyRTF.TabPS.DEFAULT_WIDTH*8)

        names=('Number of colonies','Mean colony diameter (mm)','Mean colony area (mm^2)','Median colony diameter (mm)',
               'Median colony area (mm^2)','Std of colony diameter (mm)','Number of blobs','Number of colonies per blob',
               'Total blob area (mm^2)','Total colony area (mm^2)','Total blob fit','Number of bad blobs',
               'Area fraction of bad blobs','Adjusted number of colonies')
               
        values=("%0.0f" % statistics['numberOfColonies'],"%0.2f" % statistics['meanColonyDiameter'],
                "%0.2f" % statistics['meanColonyArea'],"%0.2f" % statistics['medianColonyDiameter'],
                "%0.2f" % statistics['medianColonyArea'],"%0.2f" % statistics['stdOfColonies'],
                "%0.0f" % statistics['numberOfBlobs'],"%0.2f" % statistics['coloniesPerBlob'],
                "%0.2f" % statistics['totalBlobArea'],"%0.2f" % statistics['totalCircleArea'],
                "%0.2f" % statistics['blobFit'],"%0.0f" % statistics['numberOfBadBlobs'],
                "%0.2f" % statistics['fractionBadBlobArea'],"%0.0f" % statistics['adjNumberOfColonies'])     
        
        for i in range(0,len(names)):
            c1=PyRTF.Cell(PyRTF.Paragraph(PyRTF.TEXT(names[i],bold=True)))
            c2=PyRTF.Cell(PyRTF.Paragraph(values[i]))
            table.AddRow(c1,c2)

        section.append(table)
                
        #Append histogram
        if not histogram is None:
            section.append(e)
            p=PyRTF.Paragraph(histImage)
            section.append(p)
        
        DR=PyRTF.Renderer()
        DR.Write(doc,file(fileName,'w'))