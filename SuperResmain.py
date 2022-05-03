import tifffile
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sys

# set matplotlib backend so figure isn't rendered to screen
matplotlib.use('Agg')

class TiffStack:
    """
    Tiffstack holds information about the images to process and accesses them in a memory-efficient manner
    improvements: implement a TiffFile class for each image to more readily access parameters such as width/height
    :param pathname to the tif image
    """
    def __init__(self, pathname):
        self.ims = tifffile.TiffFile(pathname)
        self.nfiles = len(self.ims.pages)
        page = self.ims.pages[0]
        self.width = page.shape[0]
        self.height = page.shape[1]

    def getimage(self, index):
        return self.ims.pages[index].asarray()


def rolling_ball(image, radius=3):
    """
    Rolling Ball/Sliding Window background subtraction - Takes an m x n image and loops over each pixel, subtracting
    the mean value around that pixel within a specified radius.
    :param image: an m x n image-style array, e.g. numpy array or nested list
    :param radius: radius of ball/window, should be equal to half the expected object size, currently set by default
    to 3 as the general radius of a single molecule in SMLM
    :return: a background subtracted m x n image
    """
    bgsubtracted = []

    # pad array to deal with borders
    image = np.pad(image, radius, mode='reflect')

    # looping only occurs over non-padded region
    for rowindex, row in enumerate(image[radius:-radius], start=radius):
        bgrow = []
        for colindex, _ in enumerate(row[radius:-radius], start=radius):
            # crop out section of image to find mean
            left = rowindex-radius
            right = rowindex+radius+1
            top = colindex-radius
            bottom = colindex+radius+1
            crop = image[left:right, top:bottom]
            bg = np.mean(crop)

            if image[rowindex][colindex]-bg > 0:
                bgrow.append(image[rowindex][colindex]-bg)
            else:
                bgrow.append(0)
        bgsubtracted.append(bgrow)
    return bgsubtracted


def find_peaks(image):
    """
    Finds peaks in a clean image, ignores points with in a 1 pixel boundary of image edge
    Checks if the pixels in a radius of 1 (i.e. 3x3) are all lower in intensity than the center
    and that the mean of the 3x3 region is greater the mean of the image + the standard deviation
    :param image: an m x n background-less image with clear peaks of approx. radius 3
    :return: a P x 2 list of x,y pairs where P is the number of peaks found and x,y are the location in the image
    """
    peaklist = []
    stdimage = np.std(image)
    meanimage = np.mean(image)
    radius = 1
    for rowindex, row in enumerate(image[radius:-radius], start=radius):
        for colindex, element in enumerate(row[radius:-radius], start=radius):
            # Crop out region to test
            left = rowindex - radius
            right = rowindex + radius + 1
            top = colindex - radius
            bottom = colindex + radius + 1
            crop = image[left:right, top:bottom].copy()  # copy so don't overwrite original array
            cropmean = np.mean(crop)
            crop[radius+1, radius+1] = 0  # set center pixel to zero so doesn't interfere
            # if all pixels less than central pixel and the crop mean is greater than the image mean + std
            truth = crop <= element
            if np.all(truth) and cropmean > meanimage+stdimage:
                peaklist.append([rowindex, colindex])
    return peaklist


def filter_peaks(peaks, image, radius=2):
    """
    Filters a list of peaks for to remove those that are deemed too close, if a peak (p) has other peaks (op)
    that are close and brighter than it, then p is not added to the new list. if p is the brightest peak in p and op
    then p is added to the list.
    :param peaks: an nx2 list of peak positions
    :param image: an mxn image
    :param radius: region around pixel in which to filter
    :return: filtered list, subset of peaks
    """
    filteredlist = []
    peaksarray = np.array(peaks)
    # euclidean distance in which peaks that are closer than get checked *left squared
    radiusdistance =  radius**2 + 1
    for index, pointtocheck in enumerate(peaksarray):
        x = pointtocheck[0]
        y = pointtocheck[1]
        # Calculate euclidean distance between peak and all other peaks *left squared for efficiency
        dist = (peaksarray - pointtocheck) ** 2
        dist = np.sum(dist, axis=1)

        # find other peaks within radius
        indexes = np.where(dist < radiusdistance)
        closepeaks = peaksarray[indexes[0]]
        vals = [image[x, y] for x, y in closepeaks]

        # if current peak is the brightest, add to filteredlist
        if image[x, y] == max(vals):
            filteredlist.append([x, y])
    return filteredlist


def plot_and_save(image, peaks, outputpath, index):
    # Unpack peak locations
    xpositions = [pair[0] for pair in peaks]
    ypositions = [pair[1] for pair in peaks]

    # Generate plot
    # Image is upsampled by a factor of 2 so the red circles don't occlude as much of the data
    pixels = 1 / plt.rcParams['figure.dpi']
    fig = plt.figure(figsize=(image.shape[1] * 2 * pixels, image.shape[0] * 2 * pixels), frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    # Plot image and add scatter plot on top
    plt.imshow(image, aspect='equal', interpolation='none')
    plt.set_cmap('gray')
    plt.scatter(ypositions, xpositions, s=(plt.rcParams['lines.markersize']*2)**2, facecolors='none', edgecolors='r')

    # Save file, if tif extension isn't provided, add it
    if outputpath[-4:] == ".tif":
        pathwithindex = outputpath[:-4] + '_' + str(index) + ".tif"
        plt.savefig(pathwithindex, bbox_inches='tight', pad_inches=0)
    else:
        plt.savefig(outputpath + '_' + str(index) + '.tif', bbox_inches='tight', pad_inches=0)
    plt.clf()
    print(f'processed image {index+1} and found {len(peaks)} peaks')


def main(inputpath, outputpath):
    """
    Takes a path to an image file, then performs the folling operations:
    1. Load in data to a tiff container class
    2. Background subtract image
    3. Find peaks in the image
    4. Filter peaks that are too close to be individual
    5. Plot the located peaks on the image and save it.
    :param inputpath: Location of the image file
    :param outputpath: Location to save the output
    :return:
    """

    imagecontainer = TiffStack(inputpath)

    for tifindex in range(imagecontainer.nfiles):
        # Load image
        image = imagecontainer.getimage(tifindex)
        # Background subtract image
        background_subtracted = np.asarray(rolling_ball(image, radius=3))
        # Find peaks in the image
        pl = find_peaks(background_subtracted)
        # Filter peaks that are part of same smudge
        fp = filter_peaks(pl, image, radius=2)
        # Plot and save the output
        plot_and_save(image, fp, outputpath, tifindex)

    print(f'Script finished - processed {imagecontainer.nfiles} image(s)')


def run_from_cli():
    """
    This program finds smudges in an image or stack of images. Takes as input a path to the input file, and a path
    to the output file location
    """

    if len(sys.argv) != 3:
        pass
        # print('Invalid number of arguments, should take as input: \n'
        #       '\t- a path to the file to process \n'
        #       '\t- an output path to save the output to'
        #       'e.g. smudge-finder input output')
        # sys.exit(0)

        # inputpath = sys.argv[1]
        # outputpath = sys.argv[2]
        inputpath = 'file1.tif'
        outputpath = 'outfile'
        main(inputpath, outputpath)


if __name__ == "__main__":
    run_from_cli()