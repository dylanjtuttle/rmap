import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from main_code.subroutines import fits_handling as fh
import scipy.optimize as opt


def calculate_annulus(s_ra, s_dec, s_x_array, s_y_array, s_centre_x, s_centre_y, s_stokes_i, threshold, outer_boundary=4):
    """This function takes in various attributes of an array stamp and returns an annulus around the source(s) in order for the local noise to be estimated
    around the source.

    ARGUMENTS:
    - s_ra (float)            -- the right ascension of the source, in degrees
    - s_dec (float)           -- the declination of the source, in degrees
    - s_x_array (2D ndarray)  -- a 2D array containing the x values in pixel number for every pixel in the array stamp
    - s_y_array (2D ndarray)  -- a 2D array containing the y values in pixel number for every pixel in the array stamp
    - s_centre_x (int)        -- the x coordinate of the center of the source, in pixels (for calculating the beam ellipse shape)
    - s_centre_y (int)        -- the y coordinate of the center of the source, in pixels (for calculating the beam ellipse shape)
    - s_stokes_i (2D ndarray) -- the array stamp in question, containing the actual pixel values, in Jy
    - threshold (float)       -- the minimum pixel value needed to be considered a source pixel, in Jy (usually 1.2 mJy)
    - outer_boundary (int)    -- outer boundary of the annulus region, in multiples of beam FWHM (optional, defaults to 4 FWHM)

    RETURNS:
    - foreground_pixels (2D ndarray) -- a 2D array containing 1s and 0s, where 1 represents a pixel outside the annulus boundary and a 0 represents a pixel
                                        inside the annulus boundary
    """
    min_beams = 5  # minimum foreground area needed, in multiples of beam solid angle (NOT FWHM ellipse areas!)

    # Theoretical beam parameters:
    # Each beam is defined as an ellipse, with a major axis, a minor axis, and a position angle (the orientation of the major axis).
    # For the CGPS (in equatorial coordinates), the shape of this beam ellipse is purely dependent on declination.
    # That is, the length of the major axis is inversely proportional to the declination of the source, the length of the minor axis stays unchanged,
    # and the position angle is always zero (meaning every beam ellipse is oriented 'pointing' towards the celestial north pole, 0 deg RA, 90 deg Dec)
    # In galactic coordinates, that position angle is no longer zero, and changes with latitude and longitude,
    # so it must be converted just like the regular RA and Dec.
    major = 49 / 3600 / 0.005 / np.sin(s_dec / (180 / np.pi)) / 2.3548  # 49"/sin(dec) FWHM, in pixels, in Gaussian sigma instead of FWHM
    minor = 49 / 3600 / 0.005 / 2.3548  # 49" FWHM, in pixels, in Gaussian sigma instead of FWHM

    # Using the SkyCoord proper motion attributes to convert the position angle to galactic coordinates
    # Nothing is actually moving here, but proper motion is just a 2D vector in the sky and so we can use it to convert our position angle
    # The magnitude of this 'proper motion' position angle vector isn't important, and neither are the units, since they cancel out later on.
    # What's important is that it begins at each source and points directly towards the celestial north pole.
    # noinspection PyUnresolvedReferences
    celestial_north = SkyCoord(ra=s_ra, dec=s_dec, pm_ra_cosdec=0 * u.degree / u.s, pm_dec=90 * u.degree / u.s, frame='fk5', unit='degree')
    galactic_celestial_north = celestial_north.galactic
    gl_src = galactic_celestial_north.l.degree
    gb_src = galactic_celestial_north.b.degree
    # noinspection PyUnresolvedReferences
    pm_long = float(galactic_celestial_north.pm_l_cosb * u.s / u.mas)
    # noinspection PyUnresolvedReferences
    pm_lat = float(galactic_celestial_north.pm_b * u.s / u.mas)

    pa = -1 * np.arctan(pm_long / pm_lat)

    # Define elliptical Gaussian parameters:
    a = (np.cos(pa) ** 2 / (2 * minor ** 2)) + (np.sin(pa) ** 2 / (2 * major ** 2))
    b = (-1 * np.sin(2 * pa) / (4 * minor ** 2)) + (np.sin(2 * pa) / (4 * major ** 2))
    c = (np.sin(pa) ** 2 / (2 * minor ** 2)) + (np.cos(pa) ** 2 / (2 * major ** 2))

    # inv_gauss is an 'inverse' elliptical gaussian, meaning its pixel values get larger as you move further away from the centre
    inv_gauss = a * (s_x_array - s_centre_x) ** 2 + 2 * b * (s_x_array - s_centre_x) * (s_y_array - s_centre_y) + c * (s_y_array - s_centre_y) ** 2
    beam_area = 2 * np.pi * major * minor  # Beam area, units of pixels^2

    # This test was originally intended to prevent possible cases of box-filling Stokes I,
    # which might cause infinite recursion. Mostly it catches sources off the edge of the mosaic,
    # where the data is all NaNs.
    if fh.is_stamp_invalid(s_stokes_i, threshold, min_beams * beam_area) or outer_boundary > 30:
        print('\nNo usable pixels for foreground within box! Probably at edge of mosaic?')
        print(f'gl, gb = {gl_src}, {gb_src}')
        source_pixels = np.zeros(s_stokes_i.shape)
        foreground_pixels = source_pixels
        return [foreground_pixels, source_pixels]  # This is almost always when a source is at the edge of a mosaic
        # so it's usually all NaNs anyway

    # Array to hold which pixels are inside the 'foreground' region (the region where the source isn't)
    foreground_pixels = np.zeros(np.shape(s_stokes_i)) + 1

    # Each element of this array is a 1 if it's inside a certain number of standard deviations away from the
    # centre of the elliptical gaussian (remember inv_gauss is an inverse elliptical gaussian and 2.35482 is the number to
    # convert FWHM to standard deviations) and a 0 if it's outside (i.e. a non-source pixel)
    foreground_pixels = np.where(inv_gauss > outer_boundary**2 * 2.35482**2 / 8, 0, foreground_pixels)

    # elements of source_pixels are 1 if they are inside the same standard deviation boundary as foreground pixels
    # AND where the value of the pixel is above the Jy threshold, and 0 otherwise
    source_pixels = np.zeros(np.shape(s_stokes_i))

    # Trying to compare a NaN with a number (like threshold) will result in a warning, so this is a small workaround that
    # results in the same boolean array
    stokes_not_nan_bool = ~np.isnan(s_stokes_i)  # This boolean array is true if the pixel is not a NaN
    # This boolean array is true if the pixel is not a NaN and also greater than the threshold
    stokes_not_nan_bool[stokes_not_nan_bool] = s_stokes_i[stokes_not_nan_bool] > threshold

    source_pixels = np.where(np.logical_and(foreground_pixels == 1, stokes_not_nan_bool), 1, source_pixels)  # calculating source_pixels

    # Setting the NaNs equal to zero so they don't count as source pixels
    # THIS DOES NOT MEAN THE NaN PIXELS ARE NON-SOURCE PIXELS
    # Unlike stokes_not_nan_bool, this array keeps the pixel values, but sets the NaN values equal to 0
    stokes_i_no_nans = np.where(np.isnan(s_stokes_i), 0, s_stokes_i)
    num_source_pixels = np.sum(stokes_i_no_nans > threshold)

    # foreground_pixels may have included some source pixels because it only accounted for the gaussian,
    # not the actual data. This is just to make sure none of the source pixels appear in foreground_pixels
    if num_source_pixels > 0:
        only_source_pixels = np.where(stokes_i_no_nans > threshold, stokes_i_no_nans, 0)
        foreground_pixels = np.where(only_source_pixels > 0, 0, foreground_pixels)

    if np.sum(foreground_pixels) < min_beams * beam_area:
        # print(f'Not enough foreground pixels identified inside annulus! Increasing boundary to {outer_boundary + 1} FWHMs')
        annulus_pixels = calculate_annulus(s_ra, s_dec, s_x_array, s_y_array, s_centre_x, s_centre_y, s_stokes_i, threshold, outer_boundary + 1)
        foreground_pixels = annulus_pixels[0]
        source_pixels = annulus_pixels[1]

    return [foreground_pixels, source_pixels]


def estimate_local_noise(foreground_pixels, qa, qb, qc, qd, ua, ub, uc, ud):
    """This function takes in the foreground pixel array stamp that we just calculated using calculate_annulus(), as well as the stokes Q and U array stamps
    and returns the local noise in that array stamp.

    ARGUMENTS:
    - foreground_pixels (2D ndarray) -- the foreground_pixels array just calculated in calculate_annulus
    - qa (2D ndarray)                -- the Stokes Q array stamp in band A
    - qb (2D ndarray)                -- the Stokes Q array stamp in band B
    - qc (2D ndarray)                -- the Stokes Q array stamp in band C
    - qd (2D ndarray)                -- the Stokes Q array stamp in band D
    - ua (2D ndarray)                -- the Stokes U array stamp in band A
    - ub (2D ndarray)                -- the Stokes U array stamp in band B
    - uc (2D ndarray)                -- the Stokes U array stamp in band C
    - ud (2D ndarray)                -- the Stokes U array stamp in band D

    RETURNS:
    - mean_vector (1D ndarray) -- an array containing the mean local noise in stokes Q and U in bands A, B, C, and D
    - sigma_qu (float)         -- the noise, i.e. the square root of the average of each squared element of mean_vector
    """
    # Calculates the local noise and foreground in polarization, for all channels
    # using the pixel mask to select foreground pixels
    # Uses outlier-resistant mean code, which should exclude pixels that belong to
    # potential neighboring sources

    default_noise = 0.0008  # This is used for sigma_QU when there are no foreground pixels below the threshold
    num_foreground = np.sum(foreground_pixels)  # since the foreground pixels are 1s and the background pixels are 0s, we can just sum the array
    band_list = [qa, qb, qc, qd, ua, ub, uc, ud]

    if num_foreground > 0:
        # resistant_mean_cve reports the sigma in the mean, so it needs to be converted
        # back to the sigma of the distribution for error analysis purposes
        # This may break if there's only one pixel that isn't flagged! Hopefully that never happens
        sigma_vector = []
        mean_vector = []
        for band in band_list:
            mean, sigma, num_reject = resistant_mean_cve(band[np.logical_and(foreground_pixels == 1, ~np.isnan(band))], 2.0)
            sigma = sigma * np.sqrt(num_foreground - num_reject - 1)
            sigma_vector.append(sigma)
            mean_vector.append(mean)

        mean_vector, sigma_vector = np.array(mean_vector), np.array(sigma_vector)
        sigma_qu = np.sqrt(np.sum(sigma_vector**2) / sigma_vector.size)

    else:  # In the event that there are no useful foreground pixels, set noise to zero
        mean_vector = np.zeros(len(band_list))
        sigma_qu = default_noise

    return mean_vector, sigma_qu


def resistant_mean_cve(array, sigma_cut):
    """This function calculates the mean and standard deviation of an input array, and trims away outliers using the median and
    the median absolute deviation. An approximation formula is used to correct for the truncation caused by trimming away outliers

    ARGUMENTS:
    - array (ndarray) -- the array we wish to calculate the mean and standard deviation of
    - sigma_cut (float)  -- the number of standard deviations from the median in order for an outlier to be ignored (recommended 2.0 and up)

    RETURNS:
    - mean_good (float)  -- the mean of the input array
    - sigma_good (float) -- the approximate standard deviation of the mean. This is the sigma of the distribution divided by sqrt(N-1) where N
                            is the number of unrejected points. The larger the sigma_cut, the more accurate. It will tend to underestimate the
                            true uncertainty of the mean, and this may become significant for cuts of 2.0 or less.
    - num_reject (float) -- the number of rejected points
    """
    num_points = array.size
    array = array[~np.isnan(array)]
    median = np.median(array)

    # The absolute deviation of a set of data is the difference of each data point and the median of the array
    absolute_deviation = np.abs(array - median)
    med_abs_dev = np.median(absolute_deviation) / 0.6745  # Sorry about the magic number, it was in the IDL code without any explanation

    if med_abs_dev < 1 * 10 ** -24:
        med_abs_dev = np.mean(absolute_deviation) / 0.8  # Another magic number

    cutoff = sigma_cut * med_abs_dev

    good_points = array[absolute_deviation < cutoff]  # 1D array containing all elements less than cutoff
    num_good = good_points.size

    if num_good == 0:  # This code will gently break if there are no good points, so we just skip the error messages and return NaNs, which the code would have done anyway
        mean_good = np.nan
        sigma_good = np.nan
        num_reject = num_points
    elif num_good == 1:  # It will also gently break if there is only one good point
        mean_good = good_points[0]  # Since good points is an array of only 1, the mean is just that one good value
        sigma_good = np.sqrt(np.sum((good_points - mean_good) ** 2) / num_good)  # This will equal zero, but it's better than just writing zero (no magic numbers!)
        num_reject = num_points - 1
    else:
        mean_good = np.mean(good_points)
        sigma_good = np.sqrt(np.sum((good_points - mean_good) ** 2) / num_good)

        # Compensate sigma for truncation (formula by HF) ((whoever or whatever that is))
        if sigma_cut < 1.0:
            sigma_cut = 1.0
        if sigma_cut < 4.5:
            sigma_good = sigma_good / (-0.15405 + (0.90723 * sigma_cut) - (0.23584 * sigma_cut**2) + (0.020142 * sigma_cut**3))

        cutoff = sigma_cut * sigma_good
        good_points = array[absolute_deviation < cutoff]
        num_good = good_points.size
        mean_good = np.mean(good_points)
        sigma_good = np.sqrt(np.sum((good_points - mean_good) ** 2) / num_good)
        num_reject = num_points - num_good

        if sigma_cut < 1.0:
            sigma_cut = 1.0
        if sigma_cut < 4.5:
            sigma_good = sigma_good / (-0.15405 + (0.90723 * sigma_cut) - (0.23584 * sigma_cut**2) + (0.020142 * sigma_cut**3))

        # Now the standard deviation of the mean:
        sigma_good = sigma_good / np.sqrt(num_good - 1)

    return mean_good, sigma_good, num_reject


def construct_new_ston_cutout(stokes_i, qa, qb, qc, qd, ua, ub, uc, ud, foreground_vector, sigma_qu):
    """This function calculates the position dependent noise (due to leakage), calculate PI and apply
    debias calculation, and work out the S-N array, around a single source. This can then be used to
    determine if there is a statistically significant polarization signal.
    Be aware that NaNs can crop up in the array stamps.

    ARGUMENTS:
    - stokes_i (2D ndarray) -- the Stokes I array stamp
    - qa (2D ndarray)       -- the Stokes Q array stamp in band A
    - qb (2D ndarray)       -- the Stokes Q array stamp in band B
    - qc (2D ndarray)       -- the Stokes Q array stamp in band C
    - qd (2D ndarray)       -- the Stokes Q array stamp in band D
    - ua (2D ndarray)       -- the Stokes U array stamp in band A
    - ub (2D ndarray)       -- the Stokes U array stamp in band B
    - uc (2D ndarray)       -- the Stokes U array stamp in band C
    - ud (2D ndarray)       -- the Stokes U array stamp in band D
    - foreground_vector (1D ndarray) -- an array containing the mean local noise in stokes Q and U in bands A, B, C, and D,
                                        as calculated by estimate_local_noise()
    - sigma_qu (float) -- the noise, i.e. the square root of the average of each squared element of mean_vector,
                                        as calculated by estimate_local_noise()

    RETURNS:
    - pi_debiased (float)      --
    - noise_array (2D ndarray) --
    - ston_array (2D ndarray)  --
    """
    leakage_noise_fraction = 0.003  # alpha_c, Thesis

    # Position dependent noise:
    noise_array = np.sqrt((stokes_i * leakage_noise_fraction)**2 + sigma_qu**2)

    # Subtract foreground mean from Q/U and calculate PI
    # Now apply WK74 bias correction
    # First, calculate PI^2 with bias term, then do the necessary sign correction
    pi_a2 = (ua - foreground_vector[4])**2 + (qa - foreground_vector[0])**2 - noise_array**2
    pi_b2 = (ub - foreground_vector[5])**2 + (qb - foreground_vector[1])**2 - noise_array**2
    pi_c2 = (uc - foreground_vector[6])**2 + (qc - foreground_vector[2])**2 - noise_array**2
    pi_d2 = (ud - foreground_vector[7])**2 + (qd - foreground_vector[3])**2 - noise_array**2

    pi_a = np.sqrt(np.abs(pi_a2))
    pi_b = np.sqrt(np.abs(pi_b2))
    pi_c = np.sqrt(np.abs(pi_c2))
    pi_d = np.sqrt(np.abs(pi_d2))

    pi_debiased = (pi_a + pi_b + pi_c + pi_d) / 4

    # Calculating the S:N is a bit probelematic. In the zero signal case (Rayleigh distribution),
    # the variance is (4 - pi) sigma_qu^2 / 2 (for a single channel)
    # and the corresponding sigma_pi_i would be ~0.655 sigma_qu.
    # The de-biased PI won't follow Rayleigh statistics, though,
    # and so it's not clear what the noise in the band average should be.
    # In the high S:N case, (Ricean, behaves like Gaussian), sigma_pi_i ~= sigma_qu
    # and then the 4-band averaged PI should then have sigma_pi_avg ~= sigma_QU / 2/
    # In the intermediate S:N, all bets are off.
    # Since we're mostly interested in the higher S:N cases, that should give a useful approximation.
    ston_arr = pi_debiased / (noise_array / 2)

    return pi_debiased, noise_array, ston_arr


def gaussian_2d(xy_tuple, amplitude, x_0, y_0, sigma_x, sigma_y, theta, offset):
    x, y = xy_tuple
    a = (np.cos(theta)**2) / (2 * sigma_x**2) + (np.sin(theta)**2) / (2 * sigma_y**2)
    b = -(np.sin(2 * theta)) / (4 * sigma_x**2) + (np.sin(2 * theta)) / (4 * sigma_y**2)
    c = (np.sin(theta)**2) / (2 * sigma_x**2) + (np.cos(theta)**2) / (2 * sigma_y**2)

    gauss = offset + amplitude * np.exp(-(a * ((x - x_0)**2) + 2 * b * (x - x_0) * (y - y_0) + c * ((y - y_0)**2)))

    return gauss.flatten()


def gauss_fit_2d(z_arr, x_arr, y_arr, par_estimate):
    x_arr2d = np.zeros((len(x_arr), len(x_arr)))
    y_arr2d = np.zeros((len(y_arr), len(y_arr)))
    for i in range(len(x_arr)):
        x_arr2d[:, i] = x_arr[i]
    for i in range(len(y_arr)):
        y_arr2d[i, :] = y_arr[i]

    # par_estimate must have the form [amplitude, x_0, y_0, sigma_x, sigma_y, theta, offset]
    popt, pcov = opt.curve_fit(gaussian_2d, (x_arr2d, y_arr2d), z_arr.flatten(), p0=par_estimate)

    amp = popt[0]
    xmean = popt[1]
    ymean = popt[2]
    x_std = popt[3]
    y_std = popt[4]
    theta = popt[5]
    yfit = gaussian_2d((x_arr2d, y_arr2d), *popt).reshape(len(x_arr), len(y_arr))

    return amp, xmean, ymean, x_std, y_std, theta, yfit
