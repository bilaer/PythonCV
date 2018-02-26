import PIL
from PIL import Image
from PIL import ImageDraw
import math
import numpy as np
import matplotlib.pyplot as plt


class pythonCV(object):
    pass

class image(object):
    def __init__(self, filePath):
        self.imagePath = filePath
        self.imageData, self.channels = self.ReadImageAsArray()
        self.grey = self.GetGrey()
        self.ReadImageAsArray()

    # Read the image into the numpy array
    def ReadImageAsArray(self):
        # Open the image
        try:
            im = np.array(Image.open(self.imagePath))
        except OSError:
            print("Cannot open the image, image does not exist")
        else:
            # Extract info from the image
            if len(im.shape) == 3:
                rchannel = im[:, :, 0]
                gchannel = im[:, :, 1]
                bchannel = im[:, :, 2]
                return im, (rchannel, gchannel, bchannel)
            else:
                return im, None

    # Get greyscale data of given image
    def GetGrey(self):
        try:
            im = Image.open(self.imagePath).convert("L")
        except OSError:
            print("Cannot open the image, image does not exist")
        else:
            return np.array(im)

    def ShowRGBImage(self):
        try:
            im = Image.open(self.imagePath).convert("L")
        except OSError:
            print("Cannot open the image, image does not exist")
        else:
            im = Image.open(self.imagePath)
            im.show()

    def ShowGReyImage(self):
        try:
            im = Image.open(self.imagePath).convert("L")
        except OSError:
            print("Cannot open the image, image does not exist")
        else:
            im.show()
    
    # Height X Weigth X Channels
    def GetWidth(self):
        return self.imageData.shape[1]

    def GetHeight(self):
        return self.imageData.shape[0]



# Filter class is used to processing the image
class Filter(object):
    def __init__(self):
        pass

    ###################################################
    #            Grey-Level Transformation            #
    ###################################################
    # Manipulate only on every single pixel
    # There are several common types of transformation:
    # linear, logarithmic and power-law
    def greyLevelTrans(self, image, method):
        output = np.full(image.shape, 0)
        for row in range(image.shape[0]):
            for col in range(image.shape[1]):
                newValue = method(image[row][col])
                output[row][col] = newValue

        return PIL.Image.fromarray(np.uint8(output))


    def negative(self, value):
        return 255 - value

    # s = clog(1 + r) r >= 0
    # expand the values of dark pixels in an image while
    # compressing the higher-level values
    def logTrans(self, value, c):
        return c*math.log(1 + value)

    # s = cr**gama
    def powerTrans(self, value, gama, c):
        return c*(value**gama)

    # Return the histogram of image
    def getHistGram(self, image):
        # Calculate the histogram data
        greyLevel = dict()
        for row in range(image.shape[0]):
            for col in range(image.shape[1]):
                if image[row][col] in greyLevel:
                    greyLevel[image[row][col]] += 1
                else:
                    greyLevel[image[row][col]] = 1

        # Draw the histogram using matplotlib
        x = []
        for level in sorted(greyLevel):
            x.append(greyLevel[level])
        plt.plot(x)
        plt.title("Image Histogram")
        plt.xlabel("Grey Level")
        plt.ylabel("Number of pixels")
        # Display
        plt.show()

        return x

    # Histogram Equalization algorithm
    def histogramEq(self, image):
        pass

    ###################################################
    #                   Kernels                       #
    ###################################################

    # Establish avg kernel
    def getAvgKernel(self, k):
        avgKernel = np.full((k, k), 1)
        return avgKernel
    
    # Establish gaussian kernel
    def getGaussianKernel(self, sigma):

        # 1-D gaussian formula
        def gaussian(self, x, sigma):
            first = 1/((2*math.pi*(sigma**2))**0.5)
            second = math.exp(-(x**2/(2*(sigma**2))))
            return first*second
        
        # Get the integral of a interval, i.e a pixiel
        def integral(self, x, increment, sigma):
            result = 0
            for i in range(int(1/increment)):
                cur = x + i*increment
                result = result + gaussian(self, cur, sigma) 
            return result

        # Approximate the discrete gaussian distribution by calculating the integral
        # of gaussian of each interval.
        # Automatically optimize the size of kernel
        def gaussianApproximate(self, sigma):
            x = -0.5
            increment = 0.01
            outside = 0.001
            result = list()

            # Get the optimized size of kernel and insert values into the list
            while integral(self, x, increment, sigma) >= 0.001 or len(result)%2 == 0:
                result.append(integral(self, x, increment, sigma))
                x = x + 1 + increment
            
            # Fulfil the kernel
            for i in range(len(result) - 1):
                result.insert(i, result[-1 - i])

            # Evenly distribute the weight that haven't been covered by the kernel
            # and distribute them into the existing kernel
            left = 100 - sum(result)
            dis = left/len(result)
            for i in range(len(result)):
                result[i] = result[i] + dis
            
            return (len(result), np.array([result])) 
            
        return gaussianApproximate(self, sigma)

    # laplacianKernel is used to detect the edges
    # second derivative of f is f(x+1, y) + f(x-1, y) + f(x, y+1) + f(x, y-1)
    # - 4(x, y)
    # Laplacian of Gaussian is sensitive to the noise
    def getLoGKernel(self, sigma):
        k, gauKernel = self.getGaussianKernel(sigma)
        # Get 2d gaussian kernel since the function can only
        # give 1d gaussian
        gauKernel = np.outer(gauKernel, gauKernel)
        # second derivative [1, -2, 1]*[-, -2, 1]T = laplacian
        der = np.array([[0, 1, 0],
                        [1, -4, 1],
                        [0, 1, 0]])
        r = int((der.shape[0] - 1)/2)
        # Add padding to the gaussian kernel before padding

        # Add padding
        padding = self.padding(gauKernel, gauKernel.shape[1], gauKernel.shape[0], r)
        # Convoluted gaussian kernel with second derivative
        # Array used for output image
        LoG = np.full(gauKernel.shape, 0)

        # Convolution
        for inputRow in range(r, padding.shape[0] - r):
            for inputCol in range(r, padding.shape[1] - r):

                # Get the subarray for operation
                convArray = padding[(inputRow - r):(inputRow + r + 1), (inputCol - r):(inputCol + r + 1)]

                # Get the sum of multiplcation of two arrays and then update the value
                newValue = 0
                for convRow in range(2 * r + 1):
                    for convCol in range(2 * r + 1):
                        # Remember that convolution matrix is start reversively from bottom right to
                        # up left
                        newValue = newValue + convArray[convRow][convCol] * der[2 * r - convRow][2 * r - convCol]
                        # Manipulate the convoluted values

                # Set the corresponding value in output array to newValue
                LoG[inputRow - r][inputCol - r] = newValue

        return LoG

    # Resemble to derivative of the gaussian
    # It is approximated by centered differences
    def getSobelKernel(self):
        gy = np.array([[-1, -2, -1],
                       [0, 0, 0],
                       [1, 2, 1]])
        gx = np.array([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]])
        return (gx, gy)

    def getGoberKernel(self):
        pass

    ##################################################
    #                   Convolution                  #
    ##################################################
    # General smoothing function
    def conv(self, image, width, height, factor, kernel, r, handle=None, correct=True, array=False, debug=False,
             Float=False):
        # Array used for output image
        output = np.zeros((height - 2*r, width - 2*r))

        # Flip the kernel
        flip = np.zeros(kernel.shape)
        for i in range(flip.shape[0]):
            for j in range(flip.shape[1]):
                flip[i][j] = kernel[kernel.shape[0] - 1 - i][kernel.shape[1] - 1 - j]

        # Convolution
        for inputRow in range(r, height - r):
            for inputCol in range(r, width - r):
                if debug:
                    print("current row: %d, total: %d" %(inputRow, height))

                # Get the sum of multiplcation of two arrays and then update the value
                convArray = image[(inputRow-r):(inputRow+r+1), (inputCol-r):(inputCol+r+1)]
                newValue = np.sum(convArray*flip)
                if handle != None:
                    newValue = handle(newValue)
                # Set the corresponding value in output array to newValue
                output[inputRow - r][inputCol - r] = newValue/factor

        # Correct the image in case there value smaller than 0 or larger than 255
        if correct == True:
            output = self.correctDifference(output)
        if not Float:
            output = np.uint8(output)
        if array == False:
            outputImg = PIL.Image.fromarray(output)
            return outputImg
        else:
            return output

    
    # A fast approaches that uses horizontal and vertical filters to do the convolution
    def sperablConv(self, image, width, height, hor, vertical, r, debug=False, Float=False):
        # Array used for output image
        output = np.zeros((height - 2*r, width - 2*r))
        temp = np.zeros((height, width - 2*r))

        # Horizontal convolution
        for inputRow in range(height):
            for inputCol in range(r, width - r):

                if debug:
                    print("current height: %d, total: %d" %(inputRow, temp.shape[0]))
                    print("current row: %d, total: %d" %(inputCol, temp.shape[1]))

                # Get the sum of multiplcation of two arrays and then update the value
                convArray = image[inputRow:(inputRow + 1), (inputCol - r):(inputCol + r + 1)]
                newValue = np.sum(convArray * hor)
                
                # Set the corresponding value in output array to newValue
                temp[inputRow][inputCol - r] = newValue/hor.sum()
        
        # Vertical convolution
        for inputRow in range(r, height - r):
            for inputCol in range(width - 2*r):

                if debug:
                    print("current height: %d, total: %d" %(inputRow, output.shape[0]))

                # Get the sum of multiplcation of two arrays and then update the value
                convArray = temp[(inputRow - r):(inputRow + r + 1), inputCol:(inputCol + 1)]
                newValue = np.sum(convArray * vertical)
                # Set the corresponding value in output array to newValue
                output[inputRow - r][inputCol] = newValue/vertical.sum()


        # Correct image
        #output = self.correctDifference(output)
        if not Float:
            return PIL.Image.fromarray(np.uint8(output))
        else:
            return PIL.Image.fromarray(output)

    #############################################################
    #                    Spatial filters                        #
    #############################################################
    # low and high band-pass filters are also called spatial
    # filter
    # Smoothing using box filter as average smoothing
    def AverageSmoothing(self, image, width, height, k):
        # Half width of the kernel
        r = int((k-1)/2)

        # Get the box kernel
        avgKernel = self.getAvgKernel(k)

        # Add the mirror padding onto the original image
        mirror = self.padding(image, width, height, r)

        # Get the smoothed image of mirror version
        # Remeber that the size of image has been changed due to the padding process
        return self.conv(mirror, width+2*r, height+2*r, k*k, avgKernel, r) 
                
    
    # Average smoothing algorithm that uses horizontial and vertical kernel
    # Much faster than original algorithm
    def SepAverageSmoothing(self, image, width, height, k):
        hor = np.full((1, k), 1)
        ver = np.full((k, 1), 1)
        r = int((k-1)/2)

        # Add the mirror padding onto the original image
        mirror = self.padding(image, width, height, r)

        return self.sperablConv(mirror, width+2*r, height+2*r, hor, ver, r)

    # Smoothing using gaussian kernel
    def GaussianSmoothing(self, image, width, height, sigma):
        k, hor = self.getGaussianKernel(sigma)
        ver = np.transpose(hor)
        r = int((k - 1)/2)
        mirror = self.padding(image, width, height, r)
        return self.sperablConv(mirror, width+2*r, height+2*r, hor, ver, r, Float=True)

    # Filtering with LoG kernel
    # LoG can be used to detect zero-crossing while remaining less
    # sensitive to the noise
    def LoGOpt(self, image, width, height, sigma):
        # Get the kernel and calculate the parameter
        kernel = self.getLoGKernel(sigma)
        r = int((kernel.shape[0] - 1) / 2)

        # Get the padding
        mirror = self.padding(image, width, height, r)
        # Convoluted image with LoG filter
        return self.conv(mirror, width+2*r, height+2*r, 1, kernel, r)

    # Helper function that correct the values of a image
    # in which negative values exists.
    # Firstly, find the minimum difference of the image
    # Secondly add the negative of minimum difference to all pixels
    # Thirdly, all the pixels in the image are scaled to the interval
    # [0, 255] by multiplying each pixel by the quantity 255/Max, where
    # Max is the maximum pixel value in the modified difference image
    # The format of image is numpy array
    def correctDifference(self, image):
        minValue = np.amin(image)
        valueRange = np.amax(image) - minValue
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                image[i][j] = image[i][j] - minValue

        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                image[i][j] = int(image[i][j]*255/valueRange)

        return image
    
    # Sharpen is done by extracting original image with it's blur version
    # Unfinished
    # g = f + k*(f - hblur*f)
    def sharpen(self, image, width, height, sigma):
        # Scaling constant
        k = 0.7
        
        print("creating sharpen mask...")
        # Firstly make a guassian smooth of the image
        smooth = np.array(self.GaussianSmoothing(image, width, height, sigma))
       
        # Generate unsharpen mask
        # Note the overflow problem of uint8 type
        unsharpenMask = np.full(image.shape, 0)

        # Substrate smoothed pixel from orignal image
        for i in range(smooth.shape[0]):
            for j in range(smooth.shape[1]):
                left = 2*image[i][j]
                right = smooth[i][j]
                unsharpenMask[i][j] = int(left) - int(right)

        print("calculating sharpen image...")
        # Get the sharpen version of original image bases on sharpen mask
        result = np.full(image.shape, 0)

        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                result[i][j] = image[i][j] + k*unsharpenMask[i][j]

        # Correct the sharpen image
        correctedOutPut = self.correctDifference(result)

        return PIL.Image.fromarray(np.uint8(correctedOutPut))


    #############################################################
    #                    Morphology                             #
    #############################################################
    def erosion(self, binaryImage):
        pass

    def dilation(self, binaryImage):
        pass

    def closing(self, binaryImage):
        pass

    def opening(self, binaryImage):
        pass


    #############################################################
    #                      Image Pyramid                        #
    #############################################################

    def getDifference(self, firstImg, secondImg):
        if firstImg.shape != secondImg.shape:
            raise Exception("image has different size")
        else:
            result = np.int16(np.full(firstImg.shape, 0))
            for i in range(firstImg.shape[0]):
                for j in range(secondImg.shape[1]):
                    result[i][j] = int(firstImg[i][j]) - int(secondImg[i][j])

        return result

    # Need to be implemented
    # Firstly blur the image and then remove even rows and columns
    # Save difference of original image and blur one alone the ways
    # Return the smaller image and laplacian pyramid for restoring the size of image
    def imagePrymaid(self, image, sigma, ratio):
        count = 1
        channel = image.channels
        laplacianPy = []
        while True:
            # Smooth the image
            print("take smooth of rgb channel")
            smoothrc = np.array(self.GaussianSmoothing(channel[0], channel[0].shape[1], channel[0].shape[0], sigma))
            smoothgc = np.array(self.GaussianSmoothing(channel[1], channel[1].shape[1], channel[1].shape[0], sigma))
            smoothbc = np.array(self.GaussianSmoothing(channel[2], channel[2].shape[1], channel[2].shape[0], sigma))

            print("get the difference")
            laprc = self.getDifference(channel[0], np.array(smoothrc))
            lapgc = self.getDifference(channel[1], np.array(smoothgc))
            lagbc = self.getDifference(channel[2], np.array(smoothbc))
            # Get the difference
            laplacianPy.append((laprc, lapgc, lagbc))
            # Remove even rows and columns
            newRowSize = channel[0].shape[0] - channel[0].shape[0]//2
            newColSize = channel[0].shape[1] - channel[0].shape[1]//2
            if newRowSize != 0 and newColSize != 0:
                print("update")
                nextPyrc, nextPygc, nextPybc = [], [], []
                for i in range(channel[0].shape[0]):
                    if i%2 == 0:
                            nextPyrc.append([smoothrc[i][j] for j in range(len(smoothrc[0])) if j%2 == 0])
                            nextPygc.append([smoothgc[i][j] for j in range(len(smoothgc[0])) if j%2 == 0])
                            nextPybc.append([smoothbc[i][j] for j in range(len(smoothbc[0])) if j%2 == 0])
                    else:
                        continue
                count = count + 1
                # Update the image to the current one
                channel = [np.array(nextPyrc), np.array(nextPygc), np.array(nextPybc)]
            else:
                # Reach to the top of the pyramid, save it into the list
                laplacianPy.append(channel)

            if 1/count == ratio:
                break

        # Switch arrays to rgb image
        result = np.full((len(channel[0]), len(channel[0][0]), 3), 0)
        result[:, :, 0] = np.array(channel[0])
        result[:, :, 1] = np.array(channel[1])
        result[:, :, 2] = np.array(channel[2])

        return PIL.Image.fromarray(np.uint8(result)), laplacianPy

    # Give the image and pyramid of this image, return the image
    # of its original size
    # Constructing...
    def imagePyramidRev(self, image, pyramid):
        pass

    #############################################################
    #        Image Segmentation/Feature Detection               #
    #############################################################

    # Using normalize cross-correlation method to enhance the
    # the result
    def findTemplate(self, path):
        pass


    def pointDetection(self, image):
        # Kernel for detection of point
        # the value of point is significantly different
        # than other points
        kernel = np.array([[-1, -1, -1],
                           [-1,  8, -1],
                           [-1, -1, -1]])
        pass

    # Detect the edges by firstly applying the sobel filters to the images first
    # and apply non maximum suppression and finally use double threshold to catch the
    # edges
    def cannyEdgeDetection(self, image, width, height, sigma, t1, t2):

        # Smooth the image
        print("smoothing the image...")
        if sigma != 0:
            image = np.array(self.GaussianSmoothing(image, width, height, sigma))

        # Apply gradient filter
        print("apply the sobel filter...")
        gx, gy = self.getSobelKernel()
        r = int((gx.shape[0] - 1) / 2)
        mirror = self.padding(image, width, height, r)

        Gx = self.conv(mirror, width + 2 * r, height + 2 * r, 8, gx, r, correct=False, array=True)
        Gy = self.conv(mirror, width + 2 * r, height + 2 * r, 8, gy, r, correct=False, array=True)

        # Calculate the magnitude and gradients from Gx and Gy
        print("Calculating the angle and magnitude...")
        magnitude, gradient = np.full(Gx.shape, 0), np.full(Gx.shape, 0)
        for i in range(Gx.shape[0]):
            for j in range(Gx.shape[1]):
                magnitude[i][j] = ((Gx[i][j])**2 + (Gy[i][j])**2)**(1/2)
                # atan cannot pass the value when gx = 0
                if Gx[i][j] != 0 and Gy[i][j] != 0:
                    gradient[i][j] = math.degrees(math.atan(Gy[i][j]/Gx[i][j]))
                elif Gx[i][j] == 0 and Gy[i][j] > 0:
                    gradient[i][j] = 90.0
                elif Gx[i][j] == 0 and Gy[i][j] < 0:
                    gradient[i][j] = -90.0
                else:
                    gradient[i][j] = -1.0

        # Maximum suppression
        print("suppressing")
        suppress = np.copy(magnitude)
        for i in range(Gx.shape[0]):
            for j in range(Gy.shape[1]):
                # Round the angle
                compare = []
                angle = gradient[i][j]
                if magnitude[i][j] != 0:
                    if angle >= 0 and angle < 22.5 or angle >= 157.5 and angle < 180:
                        # Check horizontal pixels
                        if j - 1 >= 0:
                            compare.append(magnitude[i][j - 1])
                        if j + 1 <= Gx.shape[1] - 1:
                            compare.append(magnitude[i][j + 1])

                        if len(compare) != 0 and not (magnitude[i][j] >= max(compare)):
                            suppress[i][j] = 0

                    elif angle >= 22.5 and angle < 67.5:
                        # Check pixels from top left to bottom right
                        if i - 1 >= 0:
                            if j - 1 >= 0:
                                compare.append(magnitude[i - 1][j - 1])
                        if i + 1 <= Gx.shape[0] - 1:
                            if j + 1 <= Gx.shape[1] - 1:
                                compare.append(magnitude[i + 1][j + 1])

                        if len(compare) != 0 and not (magnitude[i][j] >= max(compare)):
                            suppress[i][j] = 0

                    elif angle >= 67.5 and angle < 112.5:
                        # Check vertical pixels
                        if i - 1 >= 0:
                            compare.append(magnitude[i - 1][j])
                        if i + 1 <= Gx.shape[0] - 1:
                            compare.append(magnitude[i + 1][j])

                        if len(compare) != 0 and not (magnitude[i][j] >= max(compare)):
                            suppress[i][j] = 0
                    else:
                        # Check pixels from top right to bottom left
                        if i - 1 >= 0:
                            if j + 1 <= Gx.shape[1] - 1:
                                compare.append(magnitude[i - 1][j + 1])
                        if i + 1 <= Gx.shape[0] - 1:
                            if j - 1 >= 0:
                                compare.append(magnitude[i + 1][j - 1])
                        if len(compare) != 0 and not (magnitude[i][j] >= max(compare)):
                            suppress[i][j] = 0

        # Hysteresis Threshold
        # Use to two thresholds. T1 < T2. T2 contains less edge than T1
        # Travel through the edges of T2, fill the gap with T1 edge
        tlow, thigh = self.simpleThreshold(suppress, t1), self.simpleThreshold(suppress, t2)

        # Help function for finding the edges
        def searchEdge(thigh, tlow, gradient, i, j):
            search = []
            angle = gradient[i][j]
            # Check if has the same gradient
            if angle >= 0 and angle < 22.5 or angle >= 157.5 and angle < 180:
                # Check vertical pixels
                if i - 1 >= 0 and thigh[i][j] == 0:
                    if thigh[i - 1][j] == 0 and tlow[i - 1][j] != 0 and \
                       gradient[i - 1][j] >= 0 and gradient[i - 1][j] < 22.5 or \
                       gradient[i - 1][j] >= 157.5 and gradient[i - 1][j] < 180:
                        thigh[i - 1][j] = 255
                        search.append((i - 1, j))

                if i + 1 <= thigh.shape[0] - 1 and thigh[i][j] == 0:
                    if thigh[i + 1][j] == 0 and tlow[i + 1][j] != 0 and \
                       gradient[i + 1][j] >= 0 and gradient[i + 1][j] < 22.5 or \
                       gradient[i + 1][j] >= 157.5 and gradient[i + 1][j] < 180:
                        thigh[i + 1][j] = 255
                        search.append((i + 1, j))


            elif angle >= 22.5 and angle < 67.5:
                # Check pixels from top right to bottom left
                if j + 1 <= thigh.shape[1] - 1:
                    if thigh[i - 1][j + 1] == 0 and tlow[i - 1][j + 1] != 0 and \
                       gradient[i - 1][j + 1] >= 22.5 and gradient[i - 1][j + 1] < 67.5:
                        thigh[i - 1][j + 1] = 255
                        search.append((i - 1, j + 1))

                if i + 1 <= thigh.shape[0] - 1:
                    if j - 1 >= 0:
                        if thigh[i + 1][j - 1] == 0 and tlow[i + 1][j - 1] != 0 and \
                           gradient[i + 1][j - 1] >= 22.5 and gradient[i + 1][j - 1] < 67.5:
                            thigh[i + 1][j - 1] = 255
                            search.append((i + 1, j - 1))


            elif angle >= 67.5 and angle < 112.5:
                # Check horizontal pixels
                if j - 1 >= 0:
                    if thigh[i][j - 1] == 0 and tlow[i][j - 1] != 0 and \
                       gradient[i][j - 1] >= - 67.5 and gradient[i][j - 1] < 112.5:
                        thigh[i][j - 1] = 255
                        search.append((i, j - 1))
                if j + 1 <= thigh.shape[1] - 1:
                    if thigh[i][j + 1] == 0 and tlow[i][j + 1] != 0 and \
                       gradient[i][j + 1] >= 67.5 and gradient[i][j - 1] < 112.5:
                        thigh[i][j + 1] = 255
                        search.append((i, j + 1))

            else:
                # Check pixels from top left to bottom right
                if i - 1 >= 0:
                    if j - 1 >= 0:
                        if thigh[i - 1][j - 1] == 0 and tlow[i - 1][j - 1] != 0 and \
                           gradient[i - 1][j - 1] >= 112.5 and gradient[i - 1][j - 1] < 157.5:
                            thigh[i - 1][j - 1] = 255
                            search.append((i - 1, j + 1))
                if i + 1 <= thigh.shape[0] - 1:
                    if j + 1 <= thigh.shape[1] - 1:
                        if thigh[i + 1][j + 1] == 0 and tlow[i + 1][j + 1] != 0 and \
                           gradient[i + 1][j + 1] >= 112.5 and gradient[i + 1][j + 1] < 157.5:
                            thigh[i + 1][j + 1] = 255
                            search.append((i + 1, j + 1))

            for x, y in search:
                # Keep searching until hit a gradient with value lower than the tlow
                searchEdge(thigh, tlow, gradient, x, y)

        # Search and add the edges
        # Start from high threshold and if search alone with the gradient
        for i in range(thigh.shape[0]):
            for j in range(thigh.shape[1]):
                angle = gradient[i][j]
                if thigh[i][j] != 0:
                    searchEdge(thigh, tlow, gradient, i, j)

        return PIL.Image.fromarray(np.uint8(thigh))


    # Algorithm that is used to detect line
    # Hough transform is voting technique that uses
    # hough space and find the intersect
    # Firstly create a Hough Accumulator Array and vote and then find
    # the grid with the most vote
    # Use polar representation to avoid infinite slope
    # xcos + ysin = d
    def houghLine(self, edgeImage, drawImage, threshold):
        # Create accumulator
        print("Initialize the accumulator...")
        accumulator = dict()

        # Find the edge points and update the accumulator
        print("Update the accumulator...")
        for y in range(edgeImage.shape[0]):
            for x in range(edgeImage.shape[1]):
                if edgeImage[y][x] != 0:
                    for theta in range(0, 181):
                        rho = round(math.cos(math.radians(theta))*x + math.sin(math.radians(theta))*y)
                        if (rho, theta) in accumulator:
                            accumulator[(rho, theta)] = accumulator[(rho, theta)] + 1
                        else:
                            accumulator[(rho, theta)] = 1

        # Get the dict with votes as key and sort it into the list
        print("get the vote...")
        votes = dict()
        for info in accumulator:
            if accumulator[info] in votes:
                votes[accumulator[info]].append(info)
            else:
                votes[accumulator[info]] = [info]

        if len(votes) == 0:
            raise Exception("No lines were detected")
            exit(1)

        maxVotes = sorted(votes)

        # Get the lines we need
        print("drawing...")
        print("max vote is: %d" %max(maxVotes))
        for i in range(len(maxVotes)):
            if maxVotes[i] >= threshold:
                for j in range(len(votes[maxVotes[i]])):
                    rho, theta = votes[maxVotes[i]][j]
                    a = np.cos(np.radians(theta))
                    b = np.sin(np.radians(theta))
                    x0 = a * rho
                    y0 = b * rho
                    x1 = int(x0 + edgeImage.shape[1]*10 * (-b))
                    y1 = int(y0 + edgeImage.shape[1]*10 * (a))
                    x2 = int(x0 - edgeImage.shape[0]*10 * (-b))
                    y2 = int(y0 - edgeImage.shape[0]*10 * (a))
                    drawLine = ImageDraw.Draw(drawImage)
                    drawLine.line([(x1, y1), (x2, y2)], fill=(255, 0, 0), width=2)

        return drawImage

    # Hough transform that detect circle.
    # #the mechanism is similar to the previous one
    def houghCircle(self, image):
        pass

    # Morphology algorithm that is used for image segmentation
    def waterShed(self, image):
        pass

    # The standard simple threshold, set pixel that
    # are larger than t to 255 and pixel that smaller than
    # t to 0
    def simpleThreshold(self, image, t):
        result = np.copy(image)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if image[i][j] >= t:
                    result[i][j] = 255
                else:
                    result[i][j] = 0
        return result

    # Choose the mid value between min and max grey level as t0
    # Put all pixels of image into G1 > T and G1 < T
    # Compute average grey level a1 and a2 in G1 and G2
    # New T = 1/2*(a1 + a2)
    # Repeat steps 2 through 4 until the difference in T in successive iterations
    # is smaller than a predefined parameter T0, which is t as a
    # parameter t passed into the func
    def globalThreshold(self, image, histo, t):
        # Get the min and max grey level
        grey = set()
        for i in range(len(histo)):
            if histo[i] != 0:
                grey.add(i)

        maxValue, minValue = max(grey), min(grey)

        # get initial threshold value
        initiT = (maxValue - minValue)/2 + minValue

        # prevT is used to record the threshold value of last
        # iteration
        prevT, newT = initiT, initiT

        # Separate all the pixiels into two groups
        # newT would be used as the final threshold
        while True:
            G1, G2 = dict(), dict()
            for i in range(image.shape[0]):
                for j in range(image.shape[1]):
                    if image[i][j] < prevT:
                        G1[(i, j)] = image[i][j]
                    else:
                        G2[(i, j)] = image[i][j]

            # Compute the average value of G1 and G2
            a1, a2 = sum(G1.values())/len(G1), sum(G2.values())/len(G2)

            # Obtain new T
            newT = 1/2*(a1 + a2)
            if abs(newT - prevT) <= t:
                break
            else:
                prevT = newT

        # Set the value of pixels according to the calculated threshold
        result = np.full(image.shape, 0)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if image[i][j] <= newT:
                    result[i][j] = 0
                else:
                    result[i][j] = 255
        return PIL.Image.fromarray(np.uint8(result))

    # Threshold using otsu's method
    # def: chooses the threshold to minimize the interclass variance of the
    # thresholded black and white pixels
    # May not work well when image has noise
    # It also perform poorly if image has no binomial histogram
    # Here use between-class variance to accelerate the algorithm
    def otsuThreshold(self, image, histo):

        def mean(greyDict):
            up = 0
            for key in greyDict:
                up = key*greyDict[key] + up
            return up/sum(greyDict)

        def variance(greyDict, mean):
            up = 0
            for key in greyDict:
                up = up + ((greyDict[key] - mean)*key)
            return up/sum(greyDict)

        # Get all the grey values exist in the image
        grey = []
        for i in range(len(histo)):
            if histo[i] != 0:
                grey.append(i)

        # Calculate the within-class variance
        # of all the possible threshold value
        bcVariance = dict()
        p = [0] * len(grey)
        for i in range(len(grey)):
            p[i] = grey[i] / sum(grey)

        for threshold in grey:

            # Normalize the histogram
            P1 = sum(p[:threshold+1])
            P2 = 1 - P1

            # Calculate the mean
            M1, M2 = 0, 0
            for i in range(0, threshold + 1):
                M1 += p[i]*i
            if P1 != 0:
                M1 = M1*(1/P1)
            else:
                M1 = 0
            for j in range(threshold + 1,len(grey)):
                M2 += p[j]*j
            M2 = M2*(1/P2)

            bcVariance[threshold] = P1*P2*((M1 - M2)**2)

        maxBcThreshold = max(bcVariance, key=bcVariance.get)
        print("the max is: %f" %maxBcThreshold)

        # Get the threshold image
        result = np.full(image.shape, 0)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if image[i][j] <= maxBcThreshold:
                    result[i][j] = 0
                else:
                    result[i][j] = 255
        return PIL.Image.fromarray(np.uint8(result))


    #############################################################
    #                     Corner Detection                      #
    #############################################################

    # Detection that invariant to the rotation but not scale
    def harrisCorner(self, image, width, height, r, k):
        # Compute x and y derivatives of image
        gx, gy = self.getSobelKernel()
        r = int((gx.shape[0] - 1) / 2)
        mirror = self.padding(image, width, height, r)

        Ix = np.array(self.conv(mirror, width + 2 * r, height + 2 * r, 8, gx, r, correct=False, array=True))
        Iy = np.array(self.conv(mirror, width + 2 * r, height + 2 * r, 8, gy, r, correct=False, array=True))

        # Compute products of derivatives at every pixel
        Ix2, Ixy, Iy2 = np.zeros(Ix.shape), np.zeros(Ix.shape), np.zeros(Ix.shape)
        for i in range(Ix.shape[0]):
            for j in range(Ix.shape[1]):
                Ix2[i][j] = float(Ix[i][j])**2
                Ixy[i][j] = float(Ix[i][j]) * float(Iy[i][j])
                Iy2[i][j] = float(Iy[i][j])**2

        # Compute the sums of the products of derivatives at each pixels
        sumKernel = np.full((3, 3), 1)
        Ix2Mirror = self.padding(image, width, height, r)
        IxyMirror = self.padding(image, width, height, r)
        Iy2Mirror = self.padding(image, width, height, r)
        Sx2 = np.array(self.conv(Ix2Mirror, width + 2 * r, height + 2 * r, 1, sumKernel, r, correct=False, array=True))
        Sxy = np.array(self.conv(IxyMirror, width + 2 * r, height + 2 * r, 1, sumKernel, r, correct=False, array=True))
        Sy2 = np.array(self.conv(Iy2Mirror, width + 2 * r, height + 2 * r, 1, sumKernel, r, correct=False, array=True))

        # Define the matrix at each pixel and take threshold on the responses
        result = np.copy(image)
        for i in range(Sx2.shape[0]):
            for j in range(Sx2.shape[1]):
                M = np.matrix([Sx2[i][j], Sxy[i][j], Sxy[i][j], Sy2[i][j]])
                R = np.linalg.det(M) + k*(np.trace(M))**2
                if result[i][j] > R:
                    result[i][j] = 255
                else:
                    result[i][j] = 0

        # Non-maximum suppression

    # Detector that is invariant to the scale, unlike the harris corner detector
    def SIFT(self, image):
        pass

    #############################################################
    #                      Image Padding                        #
    #############################################################

    # Add different paddings onto the original image
    # Mirror method
    # K is the half width of the kernel
    def padding(self, image, width, height, k):
        output = np.full((height + 2*k, width + 2*k), 0.0)
        # Add Left and right padding
        for i in range(height):
            for j in range(k):
                output[k + i][j] = image[i][k - 1 - j]
                output[k + i][output.shape[1] - k + j] = image[i][width - 1 - j]
        
        # Add original content of image onto the middle of the output image
        for i in range(height):
            for j in range(width):
                output[i + k][j + k] = image[i][j]
        
        # Add up and down padding 
        for i in range(k):
            for j in range(output.shape[1]):
                output[i][j] = output[2*k - 1 - i][j]
                output[output.shape[0] - k + i][j] = output[output.shape[0] - k - 1 - i][j]

        return output



