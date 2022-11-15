import numpy as np
from astropy.io import fits


def read_fits(path_name):
    """This function takes in the path of a fits file and returns the header and data of that fits file.

    ARGUMENTS:
    - path_name (string) -- the absolute path to the fits file to be opened, including file name and .fits extension

    RETURNS:
    - stokes_header (dictionary) -- the fits header which contains various attributes about the fits data, such as the x and y shape
    - stokes_data (2D ndarray)   -- the fits data, which contains the actual pixel values from the fits image
    """
    with fits.open(path_name) as stokes_file:
        stokes_header = stokes_file[0].header
        stokes_data = stokes_file[0].data[0, 0, :, :]
    return stokes_header, stokes_data


def make_xy_arrays(fits_header):
    size_x = fits_header['NAXIS1']
    size_y = fits_header['NAXIS2']

    x_array = np.zeros((size_x, size_y))
    y_array = np.zeros((size_x, size_y))
    x_long = np.zeros((size_x, size_y))
    y_lat = np.zeros((size_x, size_y))

    long_value = fits_header['CRVAL1']
    long_pix = fits_header['CRPIX1']
    long_delta = fits_header['CDELT1']

    lat_value = fits_header['CRVAL2']
    lat_pix = fits_header['CRPIX2']
    lat_delta = fits_header['CDELT2']

    long_start = (long_pix - 1) * (-1.0 * long_delta) + long_value
    lat_start = (lat_pix - 1) * (-1.0 * lat_delta) + lat_value

    for i in range(size_x):
        x_array[:, i] = i
        x_long[:, i] = long_start + (i * long_delta)

    for j in range(size_y):
        y_array[j, :] = j
        y_lat[j, :] = lat_start + (j * lat_delta)

    return x_array, y_array, x_long, y_lat


def cut_out_stamp(array_to_slice, x_min, x_max, y_min, y_max):
    """This function takes in a 2D ndarray and cuts out a 2D rectangular slice of it.

    ARGUMENTS:
    - array_to_cut (2D ndarray) -- the array to be sliced
    - x_min (int)               -- the beginning of the slice in the x dimension
    - x_max (int)               -- the end of the slice in the x dimension
    - y_min (int)               -- the beginning of the slice in the y dimension
    - y_max (int)               -- the end of the slice in the y dimension

    RETURNS:
    - array_stamp (2D ndarray) -- the sliced array stamp
    """
    array_stamp = array_to_slice[y_min:y_max + 1, x_min:x_max + 1]  # The + 1s are because IDL indexes arrays differently than python
    return array_stamp


def is_stamp_invalid(stamp, source_threshold, min_non_source):
    """This function takes in a 41 pixel x 41 pixel Stokes I stamp and counts the number of non-source pixels.
    If there are too many source pixels in the stamp (i.e. there seems to be some extended object like a supernova remnant), or if there are
    too few non-source pixels in the stamp for another reason (i.e. there are too many NaNs in the stamp),
    the stamp can't be used.

    ARGUMENTS:
    - stamp (2D ndarray)       -- a stokes I stamp which supposedly has a source in the centre
    - source_threshold (float) -- the minimum brightness (in Jy) a pixel must possess in order to be considered a source pixel, usually 1.2 mJy
    - max_num_source (float)   -- the minimum number of non-source pixels a stamp must have in order for it to be usable

    RETURNS:
    - True if there are fewer than min_non_source non-source pixels
    - False if there are more than min_non_source non-source pixels
    """

    # if a pixel is not a NaN, and is also below the threshold, add it to the count

    # We're turning each NaN into a value that is larger than the threshold so that they don't get added to the count of pixels that are less than the threshold
    # THIS DOES NOT MEAN THAT THE NaN PIXELS ARE SOURCE PIXELS!!
    # All this function does is count the number of non-source pixels and return a boolean representing whether there are enough of them in the stamp
    # Since NaNs are neither source pixels nor non-source pixels (they're not data at all), all we're doing here is making sure they don't show up in the count
    stamp_no_nans = np.where(np.isnan(stamp), source_threshold + 1, stamp)
    num_non_source = (stamp_no_nans < source_threshold).sum()

    if num_non_source < min_non_source:  # If there are too few non-source pixels,
        return True  # the stamp is too bright to only contain a single, non-extended source
    else:  # If there are a sufficient number of non-source pixels,
        return False  # the stamp probably contains a single non-extended source
