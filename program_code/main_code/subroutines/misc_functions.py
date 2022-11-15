import math
import numpy as np
from astropy.coordinates import SkyCoord
from astropy import units


def nround(number):
    """As I've recently learned, the built in Python round() and int() functions
    both round .5s down instead of up, so round(2.5) = 2, when it should really be 3.

    This makes no sense, and so I need to implement my own rounding function.

    ARGUMENTS:
    - number (float or int) -- the number to be rounded up or down to an int

    RETURNS:
    - the rounded number in integer form
    """
    if number - math.floor(number) < 0.5:
        return math.floor(number)
    else:
        return math.ceil(number)


def truncate(number):
    """I also had to write a function that truncated a float to an integer without rounding.
    For example, both 2.1 and 2.9 truncate to 2.

    ARGUMENTS:
    - number (float or int) -- the number to be truncated to an int

    RETURNS:
    - output (int) -- the truncated number in integer form
    """
    string_number = str(number)
    output = ''
    for digit in string_number:
        if digit == '.':
            return int(output)
        output += digit
    return int(output)


def string_normalise(string_bad, expected_len, negatives=False, front_load=False):
    """This function normalises the length of a string for ease of printing to a file, meaning if the length of the string
    is less than the expected length, it will add whitespace after the string until the length of the string equals the
    expected length.
    For example:
    '-131.1' has a length of 6, but should have a length of 8. This function then outputs '-131.1  '

    ARGUMENTS:
    - string_bad (string) -- the un-normalised string
    - expected_len (int)  -- the expected length of the string, must be at least equal to the length of string_bad
    - negatives (bool)    -- an optional condition which allows for further normalising of numbers that may have a negative sign
    - front_load (bool)   -- an optional condition which, if true, will add the whitespace to the beginning of the string instead of the end

    RETURNS:
    - string_good (string) -- the normalised string
    """
    if string_bad[0] != '-' and negatives:
        string_bad = ' ' + string_bad
    string_len = len(string_bad)
    len_diff = expected_len - string_len  # expected_len must be at least as large as string_len
    whitespace = ' ' * len_diff  # Interestingly, Python lets you multiply strings by ints and it just repeats the string
    if front_load:
        string_good = whitespace + string_bad
    else:
        string_good = string_bad + whitespace
    return string_good


def idl_where(array_condition):
    flat_bool_array = array_condition.flatten()
    return flat_bool_array.nonzero()[0]


def calc_background3(stamp, x_ann, y_ann):
    """This function calculates the median of the pre-calculated annulus of an array stamp.

    ARGUMENTS:
    - stamp (2D ndarray) -- the 41x41 array stamp to calculate the media of
    - x_ann (1D ndarray) -- the x coordinates of the annulus mask, to determine which pixels from stamp
                            should be used to calculate the median
    - y_ann (1D ndarray) -- the y coordinates of the annulus mask, to determine which pixels from stamp
                            should be used to calculate the median

    RETURNS:
    - stamp (2D ndarray) -- the same 414x41 array but with the median subtracted from every pixel"""
    annulus_median = np.median(stamp[y_ann.astype(int), x_ann.astype(int)])
    stamp -= annulus_median
    return stamp


def get_position_angle(ra, dec):
    # Using the SkyCoord proper motion attributes to convert the position angle to galactic coordinates
    # Nothing is actually moving here, but proper motion is just a 2D vector in the sky and so we can use it to convert our position angle
    # The magnitude of this 'proper motion' position angle vector isn't important, and neither are the units, since they cancel out later on.
    # What's important is that it begins at each source and points directly towards the celestial north pole.
    # noinspection PyUnresolvedReferences
    celestial_north = SkyCoord(ra=ra, dec=dec, pm_ra_cosdec=0 * units.degree / units.s, pm_dec=90 * units.degree / units.s, frame='fk5', unit='degree')
    galactic_celestial_north = celestial_north.galactic
    gl_src = galactic_celestial_north.l.degree
    gb_src = galactic_celestial_north.b.degree
    # noinspection PyUnresolvedReferences
    pm_long = float(galactic_celestial_north.pm_l_cosb * units.s / units.mas)
    # noinspection PyUnresolvedReferences
    pm_lat = float(galactic_celestial_north.pm_b * units.s / units.mas)

    pa = np.arctan(pm_long / pm_lat)

    return pa, gl_src, gb_src
