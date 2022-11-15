import numpy as np
from astropy.coordinates import SkyCoord
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import gridspec
import scipy.optimize as opt
import scipy.special as sp
from pathlib import Path
import os
import copy
from main_code.subroutines import fits_handling as fh
from main_code.subroutines import misc_functions as mf
from main_code.subroutines import array_calculations as ac


def read_pi_source_list_cd(path):
    """This function reads a previously generated candidate sourcelist to obtain a list of sources to examine
    and various basic parameters about them.
    """
    with open(path) as pi_source_list:
        pi_source_list = pi_source_list.readlines()[6:]  # Skipping the header of the file, we only want the data

    pi_source_list = np.array(pi_source_list)
    sdata = pi_source_list.size

    lmax = []
    bmax = []
    xpixmax = []
    ypixmax = []
    pimax = []
    simax = []
    stonmax = []
    for line in pi_source_list:
        split = line.split()
        lmax.append(float(split[0]))
        bmax.append(float(split[1]))
        xpixmax.append(int(split[2]))
        ypixmax.append(int(split[3]))
        pimax.append(float(split[4]))
        simax.append(float(split[5]))
        stonmax.append(float(split[6]))

    # Store each parameter as a list, where each element is a source
    pi_data = {'len': sdata,
               'lmax': np.array(lmax),
               'bmax': np.array(bmax),
               'xpixmax': np.array(xpixmax),
               'ypixmax': np.array(ypixmax),
               'pimax': np.array(pimax),
               'simax': np.array(simax),
               'stonmax': np.array(stonmax)}

    return pi_data


def read_qu_data_cve(directory_path, mosaic_name):
    """This function reads data and header information from fits images from a particular mosaic.
    """
    stokes = {}
    header = {}

    for band in ['I', 'Q_A', 'Q_B', 'Q_C', 'Q_D', 'U_A', 'U_B', 'U_C', 'U_D']:
        stokes_header, stokes_data = fh.read_fits(f'{directory_path}{mosaic_name}/m{mosaic_name}_1420_MHz_{band}_image.fits')

        stokes[band] = stokes_data
        header[band] = stokes_header

    return stokes, header


def read_chitable(chitable_directory):
    """This function reads the values of the Chi table from a given file.
    """
    chitable_name = 'Chi_Table_05.dat'
    chitable_path = f'{chitable_directory}{chitable_name}'

    print(f'\n\nCHI-TABLE being used: {chitable_name}')

    with open(chitable_path) as chitable:
        chitable = chitable.readlines()

    degrees = []
    chi2 = []
    redchi2 = []
    for line in chitable:
        split = line.split()
        degrees.append(int(split[0]))
        chi2.append(float(split[1]))
        redchi2.append(float(split[2]))

    return degrees, chi2, np.array(redchi2)


def check_freq2(freq_a, freq_b, freq_c, freq_d, header_qa, header_qb, header_qc, header_qd):
    """This function checks to make sure the frequencies of the 4 Q bands equal the frequencies of the 4 U bands.
    """
    freq_qa = float(header_qa['OBSFREQ'])
    freq_qb = float(header_qb['OBSFREQ'])
    freq_qc = float(header_qc['OBSFREQ'])
    freq_qd = float(header_qd['OBSFREQ'])

    freq_ok = True

    if freq_a != freq_qa:
        print(f'\nError in frequency for Band A')
        print(f'Q: {freq_qa}')
        print(f'U: {freq_a}')
        freq_ok = False

    if freq_b != freq_qb:
        print(f'\nError in frequency for Band B')
        print(f'Q: {freq_qb}')
        print(f'U: {freq_b}')
        freq_ok = False

    if freq_c != freq_qc:
        print(f'\nError in frequency for Band C')
        print(f'Q: {freq_qc}')
        print(f'U: {freq_c}')
        freq_ok = False

    if freq_a != freq_qa:
        print(f'\nError in frequency for Band D')
        print(f'Q: {freq_qd}')
        print(f'U: {freq_d}')
        freq_ok = False

    if 27400000 > (freq_qd - freq_qa) > 27800000:  # This condition will never be met, something can't be less than 27.4M and greater than 27.8M
        print(f'Error with frequency separation')
        print(f'freq_QD: {freq_qd}')
        print(f'freq_DA: {freq_qa}')
        print(f'diff: {freq_qd - freq_qa}')

    return freq_ok


# *************************************
# POL INT MAP
# *************************************


def plot_pol_int_map(x_l_2, y_b_2, data_fit, levels, mosaic_name, num, x_gauss_rot, y_gauss_rot, x_center_gauss, y_center_gauss,
                     x_long, y_lat, x_loc, y_loc, x_fwxm_ae, y_fwxm_ae, x_fwxm_se, y_fwxm_se, rm_text, rm_err_text, chi_string, chitable_string, m_string, n_pixels, passfail, ax):
    """This function plots the Polarised Intensity Map.
    """
    ax.contour(x_l_2, y_b_2, data_fit, levels=levels, colors='darkgrey', linewidths=1)
    ax.set_title('Pol. Int. Map')
    ax.set_xlim(np.max(x_l_2), np.min(x_l_2))
    ax.set_xlabel('Longitude')
    ax.set_ylim(np.min(y_b_2), np.max(y_b_2))
    ax.set_ylabel('Latitude')

    ax.text(-0.15, -0.1, mosaic_name, transform=ax.transAxes, color='darkorange')  # ax.transAxes allows us to give the location of the text in a normalised range from
    ax.text(-0.15, -0.15, num, transform=ax.transAxes, color='white')  # 0 to 1 instead of using the units of the data

    ax.scatter(x_gauss_rot + x_center_gauss, y_gauss_rot + y_center_gauss, s=2, color='lime')  # psym=1, symsize=0.2
    ax.text(x_long[y_loc, x_loc], y_lat[y_loc, x_loc], '*', color='lime')  # The location of this asterix is given in data space, not in the normalised 0 to 1 range

    # PLOTTING ANNULUS EDGE:
    ax.scatter(x_fwxm_ae, y_fwxm_ae, s=2, color='red')

    # PLOTTING SOURCE EDGE:
    ax.scatter(x_fwxm_se, y_fwxm_se, s=2, color='yellow')

    ax.text(0.1, 0.85, 'RM:', transform=ax.transAxes, fontweight='bold')
    ax.text(0.1, 0.8, f'{rm_text}$\\pm${rm_err_text} rad/m$^2$', transform=ax.transAxes, fontweight='bold')

    # Chi square label
    ax.text(0.47, 0.85, '$\\chi^2$:', transform=ax.transAxes, fontweight='bold')
    ax.text(0.47, 0.8, chi_string, transform=ax.transAxes, color='cyan', fontweight='bold')

    ax.text(0.73, 0.85, '$\\chi^2$ (table):', transform=ax.transAxes, fontweight='bold')
    ax.text(0.73, 0.8, chitable_string, transform=ax.transAxes, color='yellow', fontweight='bold')

    # M label
    ax.text(0.1, 0.15, 'M:', transform=ax.transAxes, fontweight='bold')
    ax.text(0.1, 0.1, m_string, transform=ax.transAxes, color='cyan', fontweight='bold')

    # Pixels label
    ax.text(0.73, 0.15, 'pixels', transform=ax.transAxes, fontweight='bold')
    ax.text(0.73, 0.1, str(n_pixels), transform=ax.transAxes, color='lime', fontweight='bold')

    if passfail:
        ax.text(0.1, 0.5, 'PASS', transform=ax.transAxes, color='lime', fontsize='large', fontweight='bold')
    else:
        ax.text(0.1, 0.5, 'FAIL', transform=ax.transAxes, color='red', fontsize='large', fontweight='bold')


# *************************************
# STOKES I MAP
# *************************************


def plot_stokes_i_map(long_arr, lat_arr, data, levels, x_gauss_rot1, y_gauss_rot1, x_gauss_rot2, y_gauss_rot2,
                      gauss_parameters, source_flag, x_long, y_lat, x_pix_max, y_pix_max, npix, ax):
    """This function plots the Stokes I Map.
    """
    ax.set_title('Stokes I Map')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_xlim(np.max(long_arr), np.min(long_arr))
    ax.set_ylim(np.min(lat_arr), np.max(lat_arr))

    ax.contour(long_arr, lat_arr, data, levels=levels, colors='darkgrey', linewidths=1,)

    # Over the same plot
    ax.contour(long_arr, lat_arr, data, levels=levels, colors='darkgrey', linewidths=1)

    ax.scatter(x_gauss_rot1 + gauss_parameters['x_mean'], y_gauss_rot1 + gauss_parameters['y_mean'], s=2, color='lime')

    # Plot locations of all candidates:
    ax.text(x_long[0, x_pix_max], y_lat[y_pix_max, 0], '*', color='yellow')
    ax.text(long_arr[0, npix - 1], lat_arr[npix - 1, 0], '*', color='lime')

    # Still the same plot!
    ax.scatter(x_gauss_rot2 + long_arr[npix, npix] - 0.08, y_gauss_rot2 + lat_arr[npix, npix], s=2, clip_on=False, color='red')
    ax.text(long_arr[npix, npix] - 0.068, lat_arr[npix, npix] + 0.02, 'Beam FWHM', color='red', fontweight='bold')

    ax.text(1.125, 0.2, 'Source Flag', transform=ax.transAxes, color='lime', fontweight='bold')
    ax.text(1.2, 0.1, str(source_flag), transform=ax.transAxes, size='xx-large', fontweight='bold', color='lime')

    ax.grid(False)


# *************************************
# PEAK PIXEL LINEAR FIT
# *************************************


def plot_peak_pixel_linear_fit(lx, pol_ang, rm_pix, drm, pol_err, predicted, probfit, ax):
    """This function plots the Peak Pixel Linear Fit.
    """
    ax.set_title(f'Peak Pixel Linear Fit\nPixel RM = {mf.nround(rm_pix)}$\\pm${mf.nround(drm)}')
    ax.set_xlabel('Lambda$^2$')
    ax.set_ylabel('Pol. Angle [deg]')
    ax.set_xlim(0.0435, 0.0455)
    ax.set_ylim(np.min(pol_ang) - 5, np.max(pol_ang) + 5)
    ax.set_xticks([0.0435, 0.0440, 0.0445, 0.0450])

    # Scatter plots
    ax.plot(lx, pol_ang, 'o', markersize=20, color='royalblue')
    ax.plot(lx, pol_ang, '^', markersize=15, color='lime')

    # Error bars
    pol_err = np.array([pol_err[0][0], pol_err[1][0], pol_err[2][0], pol_err[3][0]])
    ax.errorbar(lx, pol_ang, yerr=(2 * pol_err), capsize=5, ecolor='w', fmt='none')

    x_range = np.arange(101) * (0.0455 - 0.0435) / 100 + 0.0435

    # Fitted slope
    ax.plot(x_range, predicted, color='red', linestyle='--')
    ax.plot(x_range, predicted + 180, color='red', linestyle='--')
    ax.plot(x_range, predicted - 180, color='red', linestyle='--')

    ax.text(0.35, -0.23, f'Pixel Probfit = {mf.nround(probfit * 100)}%', transform=ax.transAxes)


# *************************************
# RM MAP
# *************************************


def plot_rm_map(rm_data, rm_text, rm_err_text, pa_text, pa_err_text, pi_units, gauss_parameters,
                x_long, y_lat, pi, x_fwhm, y_fwhm, pass_fail, wprob_t, ax):
    """This function plots the Rotation Measure Map.
    """
    ax.set_title(f'RM Map\nRM = {rm_text}$\\pm${rm_err_text}, dPA = {pa_text}$\\pm${pa_err_text}')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

    masked_array = np.ma.masked_where(rm_data == 0, rm_data)
    cmap = copy.copy(cm.get_cmap('RdBu'))
    cmap.set_bad(color='black')

    # Heatmap
    im = ax.pcolor(x_long, y_lat, masked_array, vmin=-500, vmax=500, cmap=cmap, shading='auto')
    plt.gca().invert_xaxis()

    # Overplotting the contours of PI
    levels = 3.0
    if pi_units != 'mJy/beam':
        levels = 0.3

    if gauss_parameters['amp'] / 2.0 > levels:
        levels = np.array([levels, gauss_parameters['amp'] / 2.0])
    else:
        levels = np.array([gauss_parameters['amp'] / 2.0])

    ax.contour(x_long, y_lat, pi, levels=levels, colors='darkgrey')

    # Overplotting the fwhm
    ax.scatter(x_fwhm, y_fwhm, s=2, color='lime')

    cbar = ax.figure.colorbar(im, ax=ax, ticks=[-500, -250, 0, 250, 500])
    cbar.ax.set_ylabel('RM', rotation=0, va="bottom")

    ax.text(1.25, 0.977, 'rad/m$^2$', transform=ax.transAxes)
    ax.text(1.29, -0.025, 'rad/m$^2$', transform=ax.transAxes)

    if pass_fail:
        ax.text(-0.2, -0.22, 'PASS', transform=ax.transAxes, color='lime', fontsize='large', fontweight='bold')
    else:
        ax.text(-0.2, -0.22, 'FAIL', transform=ax.transAxes, color='red', fontsize='large', fontweight='bold')

    ax.text(0.1, -0.22, f'Avg. Linfit Probfit = {round(wprob_t, 1)}%', transform=ax.transAxes, color='royalblue', fontsize='large', fontweight='bold')


# PLOT 1 - POL. INT. MAP PART 1
def pss_subplot_get4_cve(data, x_arr, y_arr, x_long, y_lat, delta, data_units, x_pi_max, y_pi_max, fwxm, num, source_flag, npix=11):
    """
    This function calculates a significant amount of data, which is primarily used in plotting the Polarised Intensity Map, but in many other paces too.

    ARGUMENTS:
    data (2D ndarray)    -- The polarised intensity data to be used in the calculations
    x_arr (2D ndarray)   -- The x pixel coordinates of the data
    y_arr (2D ndarray)   -- The y pixel coordinates of the data
    x_long (2D ndarray)  -- The x longitude coordinates of the data
    y_lat (2D ndarray)   -- The y latitude coordinates of the data
    delta (float)        -- The difference between each element in x_long and y_lat (should be 0.005 degrees in l/b coordinates)
    data_units (string)  -- The units of the data array
    x_pi_max (int)       -- The x pixel position of the source according to the Taylor17 source catalog
    y_pi_max (int)       -- The y pixel position of the source according to the Taylor17 source catalog
    fwxm (float)         -- The full width at half maximum (?) of the source
    long (float)         -- The galactic longitude of the source according to the Taylor17 source catalog
    lat (float)          -- The galactic latitude of the source according to the Taylor17 source catalog
    num (int)            -- The source number (ranges from 0 to the number of sources in Taylor17_candidates.dat file generated by generate_candidate_sourcelist.py)
    mosaic_name (string) -- The name of the mosaic (e.g. MA1)
    npix (int)           -- Defines the size (in pixels) of the regions of the stamp to be examined (default 11)

    RETURNS:
    x_a_2 (2D ndarray)            -- A 2D array containing the pixel numbers in x of the region used for calculations around the source (currently set to 21 x 21)
    y_a_2 (2D ndarray)            -- A 2D array containing the pixel numbers in y of the region used for calculations around the source (currently set to 21 x 21)
    whw (2D ndarray)              -- "Where half width", a mask indicating the fwhm within the subregion around the source
    w3s (2D ndarray)              -- "Where 3 * s", a mask indicating the 'bottom' of the source
    x_loc (int)                   -- The x pixel location of the peak of the gaussian fit
    y_loc (int)                   -- The y pixel location of the peak of the gaussian fit
    x_reg (2D ndarray)            -- The x array (in pixel units) indicating the FWHM region
    y_reg (2D ndarray)            -- The y array (in pixel units) indicating the FWHM region
    x_tick_vals (list)            -- A list of floats containing the x axis labels for plotting
    y_tick_vals (list)            -- A list of floats containing the y axis labels for plotting
    w_annulus (2D ndarray)        -- A mask indicating the annulus region
    gauss_parameters (dictionary) -- The parameters of the gaussian fit: amp, xmean, ymean, xfwhm, yfwhm, theta
    source_ok (bool)              -- A boolean indicating whether the source is acceptable
    """
    data_shape = data.shape
    try_max = 50
    num_pix = 20

    i_2 = x_pi_max
    if i_2 >= data_shape[0]:
        i_2 = data_shape[0] - 1
    j_2 = y_pi_max
    if j_2 >= data_shape[0]:
        j_2 = data_shape[0] - 1

    sub_id_x = np.array([i_2 - num_pix, i_2 + num_pix, j_2 - num_pix, j_2 + num_pix])

    if sub_id_x[0] < 0:
        sub_id_x[0] = 0
    if sub_id_x[1] >= data_shape[0]:
        sub_id_x[1] = data_shape[0] - 1
    if sub_id_x[2] < 0:
        sub_id_x[2] = 0
    if sub_id_x[3] >= data_shape[1]:
        sub_id_x[3] = data_shape[1] - 1

    data_2 = fh.cut_out_stamp(data, sub_id_x[0], sub_id_x[1], sub_id_x[2], sub_id_x[3])
    x_long_2 = fh.cut_out_stamp(x_long, sub_id_x[0], sub_id_x[1], sub_id_x[2], sub_id_x[3])
    x_a_2 = fh.cut_out_stamp(x_arr, sub_id_x[0], sub_id_x[1], sub_id_x[2], sub_id_x[3])
    y_lat_2 = fh.cut_out_stamp(y_lat, sub_id_x[0], sub_id_x[1], sub_id_x[2], sub_id_x[3])
    y_a_2 = fh.cut_out_stamp(y_arr, sub_id_x[0], sub_id_x[1], sub_id_x[2], sub_id_x[3])

    x_shape = x_long_2.shape
    x_arrs_2 = np.zeros(x_shape)
    y_arrs_2 = np.zeros(x_shape)
    for i in range(x_shape[0]):
        x_arrs_2[:, i] = i
    for i in range(x_shape[1]):
        y_arrs_2[i, :] = i

    max_x = x_shape[0] - 1
    max_y = x_shape[1] - 1

    if max_x > 25:
        max_x = 25
    if max_y > 25:
        max_y = 25

    x_longs = fh.cut_out_stamp(x_long_2, 15, max_x, 15, max_y)
    y_lats = fh.cut_out_stamp(y_lat_2, 15, max_x, 15, max_y)
    pi_sub = fh.cut_out_stamp(data_2, 15, max_x, 15, max_y)
    x_arrs = fh.cut_out_stamp(x_arrs_2, 15, max_x, 15, max_y)
    y_arrs = fh.cut_out_stamp(y_arrs_2, 15, max_x, 15, max_y)

    done = False
    try_num = 1

    # Initialize gaussian parameters
    amplitude = 0
    x_center_gauss = 0
    y_center_gauss = 0
    x_fwhm = 0
    y_fwhm = 0
    theta = 0

    # Initialize some arrays so the code doesn't get angry later on, these should never actually be accessed
    x_l_2 = fh.cut_out_stamp(x_long_2, sub_id_x[0], sub_id_x[1], sub_id_x[2], sub_id_x[3])
    y_b_2 = fh.cut_out_stamp(y_lat_2, sub_id_x[0], sub_id_x[1], sub_id_x[2], sub_id_x[3])
    data_fit = fh.cut_out_stamp(data_2, sub_id_x[0], sub_id_x[1], sub_id_x[2], sub_id_x[3])
    x_lb = np.zeros(data_fit.shape[0])
    y_lb = np.zeros(data_fit.shape[1])

    long_center_gauss = 0
    lat_center_gauss = 0

    gauss_pass = True

    try:
        while not done and try_num <= try_max:
            max_pi = np.max(pi_sub)
            eq_mask = pi_sub == max_pi
            idl_eq_mask = mf.idl_where(pi_sub == max_pi)

            # Checks to see if the 9x9 box is off the image
            data2_shape = data_2.shape

            sub_id_x[0] = x_arrs.flatten()[idl_eq_mask[0]] - npix
            sub_id_x[1] = x_arrs.flatten()[idl_eq_mask[0]] + npix
            sub_id_x[2] = y_arrs.flatten()[idl_eq_mask[0]] - npix
            sub_id_x[3] = y_arrs.flatten()[idl_eq_mask[0]] + npix

            if sub_id_x[0] < 0:
                sub_id_x[0] = 0
            if sub_id_x[1] >= data2_shape[0]:
                sub_id_x[1] = data2_shape[0] - 1
            if sub_id_x[2] < 0:
                sub_id_x[2] = 0
            if sub_id_x[3] >= data2_shape[1]:
                sub_id_x[3] = data2_shape[1] - 1

            data_fit = fh.cut_out_stamp(data_2, sub_id_x[0], sub_id_x[1], sub_id_x[2], sub_id_x[3])
            x_l_2 = fh.cut_out_stamp(x_long_2, sub_id_x[0], sub_id_x[1], sub_id_x[2], sub_id_x[3])
            y_b_2 = fh.cut_out_stamp(y_lat_2, sub_id_x[0], sub_id_x[1], sub_id_x[2], sub_id_x[3])

            x_pos_pix = mf.nround(x_arrs.flatten()[idl_eq_mask[0]])
            y_pos_pix = mf.nround(y_arrs.flatten()[idl_eq_mask[0]])
            x_pos = x_long_2[y_pos_pix, x_pos_pix]
            y_pos = y_lat_2[y_pos_pix, x_pos_pix]

            shape_y_2 = data_fit.shape
            x_lb = np.zeros(shape_y_2[0])
            y_lb = np.zeros(shape_y_2[1])
            x_lb[:] = x_l_2[0, :]
            y_lb[:] = y_b_2[:, 0]

            # Fit a 2D gaussian to the source, then ensure the centre is inside the correct region (defined by x_lb and y_lb)

            # est must have the form [amplitude, x_0, y_0, sigma_x, sigma_y, theta, offset]
            est = np.array([max_pi, 0.0, 0.0, 0.01, 0.01, 0.0, 0.0])

            amplitude, x_center_gauss, y_center_gauss, x_stddev, y_stddev, theta, yfit = ac.gauss_fit_2d(data_fit, x_lb - x_pos, y_lb - y_pos, est)

            x_fwhm = x_stddev
            y_fwhm = y_stddev
            spi = yfit.shape

            long_center_gauss = x_center_gauss + x_pos
            lat_center_gauss = y_center_gauss + y_pos

            if spi[0] != 0 and (long_center_gauss < np.min(x_lb) or long_center_gauss > np.max(x_lb) or lat_center_gauss < np.min(y_lb) or lat_center_gauss > np.max(y_lb)):
                failed = True
            else:
                failed = False

            # Finding the difference between the fitted gauss centre and the longitude and latitude of every pixel in the source,
            # then locating where this diff is at a minimum.
            dif_long = np.abs(x_longs - long_center_gauss)
            dif_lat = np.abs(y_lats - lat_center_gauss)
            min_dif_long = np.min(dif_long)
            min_dif_lat = np.min(dif_lat)
            w_long = mf.idl_where(dif_long == min_dif_long)
            w_lat = mf.idl_where(dif_lat == min_dif_lat)
            xpp = x_arrs.flatten()[w_long[0]]
            ypp = y_arrs.flatten()[w_lat[0]]

            # Check the distance between the fitted gauss centre and the real centre
            # If it's too big, the source fails
            if len(spi) != 0 and not failed and (np.abs(xpp - x_pos_pix) > 3 or np.abs(ypp - y_pos_pix) > 3):
                failed = True
            else:
                failed = False

            if len(spi) != 0 and not failed:
                done = True
            else:
                try_num += 1
                pi_sub[eq_mask] = 0.0
                npix = npix
                print(f'Try #{try_num}')

        if try_num >= try_max:
            gauss_pass = False

    except RuntimeError:
        gauss_pass = False

    if gauss_pass:
        print(f'\nGaussian fit succeeded with npix: {npix}, try: {try_num}')
    else:
        print(f'Failed to fit gaussian')
        # If the 4 flag (Gaussian fit failed) is not set already, set it
        source_flag = (source_flag | 4)

    x_a_2 = fh.cut_out_stamp(x_a_2, sub_id_x[0], sub_id_x[1], sub_id_x[2], sub_id_x[3])
    y_a_2 = fh.cut_out_stamp(y_a_2, sub_id_x[0], sub_id_x[1], sub_id_x[2], sub_id_x[3])

    print(f'\nGaussian Parameters:')
    print(f'Amplitude: {amplitude}')
    print(f'X Mean: {long_center_gauss}')
    print(f'Y Mean: {lat_center_gauss}')
    print(f'X Standard Deviation: {x_fwhm}')
    print(f'Y Standard Deviation: {y_fwhm}')
    print(f'Theta: {theta}\n')

    hw_aa = fwxm * x_fwhm
    hw_bb = fwxm * y_fwhm

    annv = 100  # Indicates the edge of the source, for annulus calculations, 1/100th max
    annw = np.sqrt(2.0 * (np.log(annv)))

    three_a = annw * x_fwhm  # Edge of source radii
    three_b = annw * y_fwhm

    x_l_2_p = (x_l_2 - long_center_gauss) * np.cos(theta) - (y_b_2 - lat_center_gauss) * np.sin(theta)
    y_b_2_p = (x_l_2 - long_center_gauss) * np.sin(theta) + (y_b_2 - lat_center_gauss) * np.cos(theta)

    whw = mf.idl_where(((x_l_2_p / hw_aa) ** 2 + (y_b_2_p / hw_bb) ** 2) <= 1.0)
    w3s = mf.idl_where(((x_l_2_p / three_a) ** 2 + (y_b_2_p / three_b) ** 2) <= 1.0)

    # Establishing the location of the point source:
    w_loc = mf.idl_where(np.logical_and(np.abs(long_center_gauss - x_long) < (delta / 2.0), np.abs(lat_center_gauss - y_lat) < (delta / 2.0)))

    w_loc_shape = w_loc.shape
    if w_loc_shape[0] == 0:  # If there are no elements of x_long/y_lat that match this condition
        w_loc = mf.idl_where(np.logical_and(np.abs(long_center_gauss - x_long) < (delta / 1.99), np.abs(lat_center_gauss - y_lat) < (delta / 1.99)))
        print(f'\nWARNING: locate center retry')

    w_loc_shape = w_loc.shape
    if w_loc_shape[0] != 0:
        x_loc = x_arr.flatten()[w_loc[0]]
        y_loc = y_arr.flatten()[w_loc[0]]
    else:
        x_loc = 0
        y_loc = 0

    levels = 0.5 + np.arange(40) * 0.5
    if data_units != 'mJy/beam':
        levels = 0.15 + np.arange(20) * 0.05

    # Get fwxm region
    x_gauss = (np.arange(1000) - 50) * x_fwhm * fwxm / 50.0
    # To avoid the warnings that pop up when a calculation results in a NaN, we calculate only the good pixels, and place the NaNs manually.
    radicand = 1.0 - (x_gauss / (x_fwhm * fwxm)) ** 2
    where_positive = radicand >= 0
    y_gauss = np.full(x_gauss.shape, np.nan)
    y_gauss[where_positive] = np.sqrt(radicand[where_positive]) * y_fwhm * fwxm
    x_gauss = np.array([x_gauss, x_gauss])
    y_gauss = np.array([y_gauss, -1 * y_gauss])

    # Rotating:
    x_gauss_rot1 = x_gauss * np.cos(theta) + y_gauss * np.sin(theta)
    y_gauss_rot1 = -1.0 * x_gauss * np.sin(theta) + y_gauss * np.cos(theta)

    # # Still on the same plot

    x_fwxm = x_gauss_rot1 + long_center_gauss
    y_fwxm = y_gauss_rot1 + lat_center_gauss
    x_tick_vals = np.array([np.max(x_l_2), x_center_gauss, np.min(x_l_2)])
    y_tick_vals = np.array([np.min(y_b_2), y_center_gauss, np.max(y_b_2)])

    whw_shape = whw.shape
    if len(whw_shape) != 0:
        x_reg = x_a_2.flatten()[whw]
        y_reg = y_a_2.flatten()[whw]
    else:
        x_reg = 0
        y_reg = 0

    if try_num <= try_max:
        source_ok = True
    else:
        source_ok = False

    diff_x = np.abs(x_arr[j_2, i_2] - x_loc)
    diff_y = np.abs(y_arr[j_2, i_2] - y_loc)

    if diff_x > 11 or diff_y > 11:
        source_ok = False
        print(f'diff_x: {diff_x}')
        print(f'diff_y: {diff_y}')

    # **************************************************************************
    # CALCULATING THE ANNULUS REGION:
    # **************************************************************************

    inc = 1.0 / 60  # One arc minute

    if x_fwhm < 0:
        sign_x_fwhm = -1.0
    else:
        sign_x_fwhm = 1.0

    if y_fwhm < 0:
        sign_y_fwhm = -1.0
    else:
        sign_y_fwhm = 1.0

    outer_edge_x = (annw * x_fwhm) + (inc * sign_x_fwhm)
    outer_edge_y = (annw * y_fwhm) + (inc * sign_y_fwhm)

    w_annulus = mf.idl_where(
        np.logical_and((((x_l_2_p / outer_edge_x) ** 2 + (y_b_2_p / outer_edge_y) ** 2) <= 1.0), (((x_l_2_p / three_a) ** 2 + (y_b_2_p / three_b) ** 2) >= 1.0)))

    # **************************************************************************
    # CALCULATING ANNULUS EDGE:
    # **************************************************************************

    x_gauss = (np.arange(100) - 50) * outer_edge_x / 50.0
    y_gauss = np.sqrt(1.0 - (x_gauss / outer_edge_x) ** 2) * outer_edge_y
    x_gauss = np.array([x_gauss, x_gauss])
    y_gauss = np.array([y_gauss, -1 * y_gauss])
    x_gauss_rot2 = x_gauss * np.cos(theta) + y_gauss * np.sin(theta)
    y_gauss_rot2 = -1 * x_gauss * np.sin(theta) + y_gauss * np.cos(theta)
    x_fwxm_ae = x_gauss_rot2 + long_center_gauss
    y_fwxm_ae = y_gauss_rot2 + lat_center_gauss

    # **************************************************************************
    # CALCULATING SOURCE EDGE:
    # **************************************************************************

    x_gauss = (np.arange(100) - 50) * x_fwhm * annw / 50.0
    y_gauss = np.sqrt(1.0 - (x_gauss / (x_fwhm * annw)) ** 2) * y_fwhm * annw
    x_gauss = np.array([x_gauss, x_gauss])
    y_gauss = np.array([y_gauss, -1 * y_gauss])
    x_gauss_rot3 = x_gauss * np.cos(theta) + y_gauss * np.sin(theta)
    y_gauss_rot3 = -1 * x_gauss * np.sin(theta) + y_gauss * np.cos(theta)
    x_fwxm_se = x_gauss_rot3 + long_center_gauss
    y_fwxm_se = y_gauss_rot3 + lat_center_gauss

    gauss_parameters = {'amp': amplitude,
                        'x_fwhm': x_fwhm,
                        'y_fwhm': y_fwhm,
                        'x_mean': long_center_gauss,
                        'y_mean': lat_center_gauss,
                        'theta': theta}

    pi_plot_data = {'x_lb': x_lb,
                    'y_lb': y_lb,
                    'x_l_2': x_l_2,
                    'y_b_2': y_b_2,
                    'data_fit': data_fit,
                    'levels': levels,
                    'num': num,
                    'x_gauss_rot': x_gauss_rot1,
                    'y_gauss_rot': y_gauss_rot1,
                    'x_center_gauss': long_center_gauss,
                    'y_center_gauss': lat_center_gauss,
                    'x_fwxm_ae': x_fwxm_ae,
                    'y_fwxm_ae': y_fwxm_ae,
                    'x_fwxm_se': x_fwxm_se,
                    'y_fwxm_se': y_fwxm_se}

    return x_a_2, y_a_2, whw, w3s, x_loc, y_loc, x_reg, y_reg, x_tick_vals, y_tick_vals, x_fwxm, y_fwxm, w_annulus, gauss_parameters, source_ok, pi_plot_data, source_flag


# PLOT 2 - STOKES I MAP
def pss_stokes_i_plot(stokes_i, x_long, y_lat, x_pix_max_i, y_pix_max_i, gauss_parameters, fwxm, x_pix_max, y_pix_max):
    """This function calculates values necessary for the plotting of the Stokes I Map.
    """
    npix = 11

    data = fh.cut_out_stamp(stokes_i, x_pix_max_i - npix + 1, x_pix_max_i + npix + 1, y_pix_max_i - npix + 1, y_pix_max_i + npix + 1)
    long_arr = fh.cut_out_stamp(x_long, x_pix_max_i - npix + 1, x_pix_max_i + npix + 1, y_pix_max_i - npix + 1, y_pix_max_i + npix + 1)
    lat_arr = fh.cut_out_stamp(y_lat, x_pix_max_i - npix + 1, x_pix_max_i + npix + 1, y_pix_max_i - npix + 1, y_pix_max_i + npix + 1)

    levels = 2 ** np.arange(20)

    x_l_b = long_arr[0, :]
    y_l_b = lat_arr[:, 0]

    # Get fwxm region
    x_gauss = (np.arange(1000) - 50) * gauss_parameters['x_fwhm'] * fwxm / 50.0

    # To avoid the warnings that pop up when a calculation results in a NaN, we calculate only the good pixels, and place the NaNs manually.
    radicand = 1.0 - (x_gauss / (gauss_parameters['x_fwhm'] * fwxm)) ** 2
    where_positive = radicand >= 0
    y_gauss = np.full(x_gauss.shape, np.nan)
    y_gauss[where_positive] = np.sqrt(radicand[where_positive]) * gauss_parameters['y_fwhm'] * fwxm

    x_gauss = np.array([x_gauss, x_gauss])
    y_gauss = np.array([y_gauss, -1 * y_gauss])

    # Rotating
    x_gauss_rot1 = x_gauss * np.cos(gauss_parameters['theta']) + y_gauss * np.sin(gauss_parameters['theta'])
    y_gauss_rot1 = -1 * x_gauss * np.sin(gauss_parameters['theta']) + y_gauss * np.cos(gauss_parameters['theta'])

    # Plot the shape of the beam:
    g_coords = SkyCoord(l=gauss_parameters['x_mean'], b=gauss_parameters['y_mean'], frame='galactic', unit='degree')
    ra = g_coords.fk5.ra.deg
    dec = g_coords.fk5.dec.deg

    major = 49 / 3600 / np.sin(dec) / 2.3548  # 49"/sin(dec) FWHM in Gaussian sigma
    minor = 49 / 3600 / 2.3548  # 49" FWHM, in Gaussian sigma

    pa, gl_src, gb_src = mf.get_position_angle(ra, dec)

    # Define Beam FWHM ellipse:
    x_gauss = (np.arange(200) - 100) * minor * fwxm / 100
    y_gauss = np.sqrt(1.0 - (x_gauss / (minor * fwxm)) ** 2) * major * fwxm
    x_gauss = np.array([x_gauss, x_gauss])
    y_gauss = np.array([y_gauss, -1 * y_gauss])
    x_gauss_rot2 = x_gauss * np.cos(pa) + y_gauss * np.sin(pa)
    y_gauss_rot2 = -1 * x_gauss * np.sin(pa) + y_gauss * np.cos(pa)

    si_plot_data = {'long_arr': long_arr,
                    'lat_arr': lat_arr,
                    'data': data,
                    'levels': levels,
                    'x_l_b': x_l_b,
                    'y_l_b': y_l_b,
                    'x_gauss_rot1': x_gauss_rot1,
                    'y_gauss_rot1': y_gauss_rot1,
                    'x_gauss_rot2': x_gauss_rot2,
                    'y_gauss_rot2': y_gauss_rot2,
                    'x_pix_max': x_pix_max,
                    'y_pix_max': y_pix_max,
                    'npix': npix}

    return si_plot_data


def get_subarrays_cve(stokes_i, qa, qb, qc, qd, ua, ub, uc, ud,
                      x_long, y_lat, x_loc, y_loc, x_a_2, y_a_2,
                      delta, pi, ston, whw, w3s, noise, w_annulus):
    """This function cuts out a series of postage stamps from the input arrays,
    as well as generates some X and Y pixel arrays
    """
    min_x = mf.nround(np.min(x_a_2))
    max_x = mf.nround(np.max(x_a_2))
    min_y = mf.nround(np.min(y_a_2))
    max_y = mf.nround(np.max(y_a_2))

    cut_pi = fh.cut_out_stamp(pi, min_x, max_x, min_y, max_y)
    cut_i = fh.cut_out_stamp(stokes_i, min_x, max_x, min_y, max_y)

    cut_qa = fh.cut_out_stamp(qa, min_x, max_x, min_y, max_y)
    cut_qb = fh.cut_out_stamp(qb, min_x, max_x, min_y, max_y)
    cut_qc = fh.cut_out_stamp(qc, min_x, max_x, min_y, max_y)
    cut_qd = fh.cut_out_stamp(qd, min_x, max_x, min_y, max_y)
    cut_ua = fh.cut_out_stamp(ua, min_x, max_x, min_y, max_y)
    cut_ub = fh.cut_out_stamp(ub, min_x, max_x, min_y, max_y)
    cut_uc = fh.cut_out_stamp(uc, min_x, max_x, min_y, max_y)
    cut_ud = fh.cut_out_stamp(ud, min_x, max_x, min_y, max_y)

    cut_noise = fh.cut_out_stamp(noise, min_x, max_x, min_y, max_y)

    cut_x_long = fh.cut_out_stamp(x_long, min_x, max_x, min_y, max_y)
    cut_y_lat = fh.cut_out_stamp(y_lat, min_x, max_x, min_y, max_y)

    cut_ston = fh.cut_out_stamp(ston, min_x, max_x, min_y, max_y)

    x_loc = mf.nround(x_loc)
    y_loc = mf.nround(y_loc)

    x_pos_long = x_long[y_loc, x_loc]
    y_pos_lat = y_lat[y_loc, x_loc]

    x_shape = cut_x_long.shape

    cut_x_arr = np.zeros(x_shape)
    cut_y_arr = np.zeros(x_shape)

    for i in range(x_shape[0]):
        cut_x_arr[:, i] = i
    for i in range(x_shape[1]):
        cut_y_arr[i, :] = i

    less_mask = mf.idl_where(np.logical_and(np.abs(x_pos_long - cut_x_long) < delta / 2, np.abs(y_pos_lat - cut_y_lat) < delta / 2))

    cut_x_loc = cut_x_arr.flatten()[less_mask[0]]
    cut_y_loc = cut_y_arr.flatten()[less_mask[0]]

    cut_x_reg = cut_x_arr.flatten()[whw]
    cut_y_reg = cut_y_arr.flatten()[whw]

    x3s = cut_x_arr.flatten()[w3s]
    y3s = cut_y_arr.flatten()[w3s]

    cut_x_ann = cut_x_arr.flatten()[w_annulus]
    cut_y_ann = cut_y_arr.flatten()[w_annulus]

    return (cut_pi, cut_i, cut_qa, cut_qb, cut_qc, cut_qd, cut_ua, cut_ub, cut_uc, cut_ud, cut_x_arr, cut_y_arr,
            cut_noise, cut_x_long, cut_y_lat, cut_ston, cut_x_loc, cut_y_loc, cut_x_reg, cut_y_reg, x3s, y3s, cut_x_ann, cut_y_ann)


def sbg_av3(cut_qa, cut_qb, cut_qc, cut_qd, cut_ua, cut_ub, cut_uc, cut_ud, cut_x_ann, cut_y_ann):
    """This function removes the background noise from the input array stamps.
    """

    cut_qa = mf.calc_background3(cut_qa, cut_x_ann, cut_y_ann)
    cut_qb = mf.calc_background3(cut_qb, cut_x_ann, cut_y_ann)
    cut_qc = mf.calc_background3(cut_qc, cut_x_ann, cut_y_ann)
    cut_qd = mf.calc_background3(cut_qd, cut_x_ann, cut_y_ann)
    cut_ua = mf.calc_background3(cut_ua, cut_x_ann, cut_y_ann)
    cut_ub = mf.calc_background3(cut_ub, cut_x_ann, cut_y_ann)
    cut_uc = mf.calc_background3(cut_uc, cut_x_ann, cut_y_ann)
    cut_ud = mf.calc_background3(cut_ud, cut_x_ann, cut_y_ann)

    return cut_qa, cut_qb, cut_qc, cut_qd, cut_ua, cut_ub, cut_uc, cut_ud


def unwrap_pa_and_rm_calcs4(cut_qa, cut_qb, cut_qc, cut_qd, cut_ua, cut_ub, cut_uc, cut_ud,
                            lambda2, crd, cut_dpsi_a, cut_dpsi_b, cut_dpsi_c, cut_dpsi_d,
                            cut_x_arr, cut_y_arr, lx):
    """This function uses the polarisation angles to calculate the rotation measure of each pixel, along with its error.
    """
    ua_shape = cut_ua.shape

    temp = np.zeros((2, ua_shape[0], ua_shape[1]))
    sigma_t = np.zeros((2, ua_shape[0], ua_shape[1]))
    prob_t = np.zeros(ua_shape)
    result = np.zeros((4, ua_shape[0], ua_shape[1]))

    # Unwrapping data
    u = np.array([cut_ua, cut_ub, cut_uc, cut_ud])
    q = np.array([cut_qa, cut_qb, cut_qc, cut_qd])

    z = q + (1j * u)
    z_norm = z / np.abs(z)

    # Checking for zeros in q and u which would make z_norm fail

    zero_mask = np.logical_and(q == 0, u == 0)
    if np.sum(zero_mask) > 0:  # If there is at least 1 zero in q or u
        print(f'****************************************** ZEROS IN Q AND U')
        z_norm[zero_mask] = 0.0 + (1j * 0.0)

    d_phase = z_norm[1:, :, :] * np.conj(z_norm[0:3, :, :])
    d_phase = np.arctan(np.imag(d_phase), np.real(d_phase))

    result[0, :, :] = np.arctan(np.imag(z_norm[0, :, :]), np.real(z_norm[0, :, :]))

    res_0 = result[0, :, :]
    w_nan = np.isnan(res_0)
    if np.sum(w_nan) > 0:
        # If there is at least one NaN, we will temporarily set it equal to zero
        res_0[w_nan] = 0.0

    neg_mask = res_0 < 0.0
    if np.sum(neg_mask) > 0:
        res_0[neg_mask] += (2 * np.pi)
    if np.sum(w_nan) > 0:
        # If there used to be any NaNs, we set them equal to zero a few lines ago, and now we set them to NaNs once again.
        res_0[w_nan] = np.nan
    result[0, :, :] = res_0

    for index in [0, 1, 2]:
        result[index + 1, :, :] = result[index, :, :] + d_phase[index, :, :]

    result = result / 2.0

    psi_err = np.array([cut_dpsi_a, cut_dpsi_b, cut_dpsi_c, cut_dpsi_d])

    # Done unwrapping, beginning rotation measure calculations
    finite_mask = mf.idl_where(np.isfinite(cut_ua))
    finite_mask_shape = finite_mask.shape
    i = cut_x_arr.flatten()[finite_mask].astype(int)
    j = cut_y_arr.flatten()[finite_mask].astype(int)

    nan_count = 0

    for k in np.arange(finite_mask_shape[0]):
        # Checking for nan's due to median removal in psi_err
        psi_where_nans = np.invert(np.isfinite(psi_err[:, j[k], i[k]]))

        psi_mask_no_nan = ~np.isnan(psi_err[:, j[k], i[k]])

        if np.sum(psi_where_nans) > 0:
            nan_count += 1
            print(f'Problem with nans at k = {k}')

            # Set nans equal to zero
            psi_err[psi_where_nans, j[k], i[k]] = 0
            psi_err[psi_where_nans, j[k], i[k]] = np.mean(psi_err[psi_mask_no_nan, j[k], i[k]])

        # Performing a linear fit on the polarisation angle
        def linear_function(x, m, b):
            return m * x + b

        lx = np.array(lx)
        fit_params, fit_cov = opt.curve_fit(linear_function, lx, result[:, j[k], i[k]], sigma=psi_err[:, j[k], i[k]])

        fit_slope, fit_yint = fit_params

        fit_data = linear_function(lx, fit_slope, fit_yint)

        weights = 1 / psi_err[:, j[k], i[k]] ** 2
        yint_sigma = np.sqrt(np.sum(weights * lx**2) / (np.sum(weights) * np.sum(weights * lx**2) - np.sum(weights * lx)**2))
        slope_sigma = np.sqrt(np.sum(weights) / (np.sum(weights) * np.sum(weights * lx**2) - np.sum(weights * lx)**2))

        # The chi-squared value is the sum of the squared difference between the observed data and the fit data divided by sigma squared
        chi_2 = np.sum((result[:, j[k], i[k]] - fit_data)**2 / psi_err[:, j[k], i[k]]**2)

        # This shadows IDL's LINFIT() PROB parameter. From the LINFIT documentation:
        # If PROB is greater than 0.1, the model parameters are “believable”. If PROB is less than 0.1, the accuracy of the model parameters is questionable.
        prob_good_fit = 1 - sp.gammainc(0.5 * (len(lx) - 2), 0.5 * chi_2)

        temp[:, j[k], i[k]] = np.array([fit_yint, fit_slope])

        sigma_t[:, j[k], i[k]] = np.array([yint_sigma, slope_sigma])
        prob_t[j[k], i[k]] = prob_good_fit

    # Completed RM calculations
    rm_array = temp[1, :, :]
    rm_error_array = sigma_t[1, :, :]
    intercept_array = temp[0, :, :].astype(float) * crd

    # Polarisation angle and initial angles
    psi = (lambda2 * rm_array * crd + intercept_array).astype(float)

    # Wrapping up the angle maps
    psi = wrap_data(psi, 0.0, 180.0)

    return rm_array, rm_error_array, psi, prob_t


def wrap_data(data, min_data, max_data):
    """This function ensures the input data falls within the region [min_data, max_data].
    """
    diff_data = max_data - min_data

    less_mask = data < min_data
    while np.sum(less_mask) > 0:
        data[less_mask] += diff_data
        less_mask = data < min_data

    greater_mask = data > max_data
    while np.sum(greater_mask) > 0:
        data[greater_mask] -= diff_data
        greater_mask = data > max_data

    return data


def fractional_polarisation_test(cut_stokes_i, cut_qa, cut_qb, cut_qc, cut_qd, cut_ua, cut_ub, cut_uc, cut_ud, cut_noise):
    """This function calculates the fractional polarisation of each of the 4 bands, and its error,
    and returns the chi-square residuals of the fractional polarisations"""
    # Fractional polarisations:
    m_a = np.sqrt(cut_ua**2 + cut_qa**2) / cut_stokes_i
    m_b = np.sqrt(cut_ub**2 + cut_qb**2) / cut_stokes_i
    m_c = np.sqrt(cut_uc**2 + cut_qc**2) / cut_stokes_i
    m_d = np.sqrt(cut_ud**2 + cut_qd**2) / cut_stokes_i

    # Uncertainties (assuming sigma_qu = sigma_i and same noise for all channels)
    dm_a = cut_noise * np.sqrt(1 + m_a**2) / cut_stokes_i
    dm_b = cut_noise * np.sqrt(1 + m_b**2) / cut_stokes_i
    dm_c = cut_noise * np.sqrt(1 + m_c**2) / cut_stokes_i
    dm_d = cut_noise * np.sqrt(1 + m_d**2) / cut_stokes_i

    mean_m = (m_a + m_b + m_c + m_d) / 4
    chi_square_residuals = ((m_a - mean_m)**2 / dm_a**2) + ((m_b - mean_m)**2 / dm_b**2) + ((m_c - mean_m)**2 / dm_c**2) + ((m_d - mean_m)**2 / dm_d**2)

    return chi_square_residuals


def avg_rm_calcs8(cut_pi, rotmeas, e_rotmeas, cut_ston, edge_threshold, cut_x_reg, cut_y_reg, cut_i, prob_t, fracpol_chi2_arr):
    """This function uses the rotation measure data to calculate various parameters
    about the source to assist in our analysis of the data.
    """
    source_ok = True

    cut_x_reg = cut_x_reg.astype(int)
    cut_y_reg = cut_y_reg.astype(int)

    # ------------------------------------------------------------
    # Beginning Averaging Calculations...
    # ------------------------------------------------------------

    # Calculate weight maps in rotation measure:
    weights = 1.0 / (e_rotmeas * e_rotmeas)

    noise_box = cut_ston[cut_y_reg, cut_x_reg]

    temp_pi = cut_pi[cut_y_reg, cut_x_reg]
    temp_si = cut_i[cut_y_reg, cut_x_reg]
    temp_wrm = weights[cut_y_reg, cut_x_reg]
    temp_rm = rotmeas[cut_y_reg, cut_x_reg]
    temp_e_rm = e_rotmeas[cut_y_reg, cut_x_reg]
    temp = np.copy(temp_pi)
    temp_prob = prob_t[cut_y_reg, cut_x_reg]
    temp_m_chi = fracpol_chi2_arr[cut_y_reg, cut_x_reg]

    noise_box_greater = mf.idl_where(noise_box >= edge_threshold)
    noise_box_greater_python = noise_box >= edge_threshold
    noise_box_greater_shape = noise_box_greater.shape
    noise_box_less = mf.idl_where(noise_box < edge_threshold)
    noise_box_less_shape = noise_box_less.shape

    # Identifying the minimum ston in region:

    if noise_box_greater_shape[0] != 0:
        temp_noise = noise_box.flatten()[noise_box_greater]
        w_ston = np.min(temp_noise)

        # WE MAY WANT TO CONSIDER AVERAGING ALL POINTS WITHIN THE FWHM AREA!

        if len(noise_box_less_shape) != 0:
            temp[noise_box_less] = np.nan  # Makes sure that the pixels have a good S:N
            temp[noise_box_greater_python] = temp_rm[noise_box_greater_python]

        # Calculating weighted rm averages:

        n_pixels = noise_box_greater_shape[0]

        rm_avc = np.sum(temp_rm.flatten()[noise_box_greater]) / n_pixels
        frac_pol = np.sum(temp_pi.flatten()[noise_box_greater] / temp_si.flatten()[noise_box_greater]) / n_pixels

        w_pi = np.max(temp_pi.flatten()[noise_box_greater])
        w_si = np.max(temp_si.flatten()[noise_box_greater])

        temp[noise_box_greater_python] = 1.0 / np.sqrt(np.sum(temp_wrm[noise_box_greater_python]))
        wrm_dev = temp.flatten()[noise_box_greater[0]]

        temp[noise_box_greater_python] = np.sum(temp_rm[noise_box_greater_python] * temp_wrm[noise_box_greater_python]) / np.sum(temp_wrm[noise_box_greater_python])
        wrm_avg = temp.flatten()[noise_box_greater[0]]

        temp_wrm_avg = temp.copy()  # To make sure that when we update temp, it doesn't also update temp_wrm_avg
        temp[noise_box_greater_python] = np.sqrt(np.sum((temp_rm[noise_box_greater_python] - rm_avc)**2) / (n_pixels - 1))

        wrm_rms = temp.flatten()[noise_box_greater[0]]
        temp[noise_box_greater_python] = np.sum(((temp_wrm_avg[noise_box_greater_python] - temp_rm[noise_box_greater_python])
                                                 / temp_e_rm[noise_box_greater_python])**2) / (n_pixels - 1)

        w_chi_2 = temp.flatten()[noise_box_greater[0]]
        if n_pixels == 1:
            w_chi_2 = 999.99

        temp[noise_box_greater_python] = np.sum(temp_e_rm[noise_box_greater_python]) / n_pixels

        temp[noise_box_greater_python] = np.sum(temp_prob[noise_box_greater_python]) / n_pixels
        w_prob_t = temp.flatten()[noise_box_greater[0]] * 100.0

        temp[noise_box_greater_python] = np.sum(temp_m_chi[noise_box_greater_python]) / n_pixels
        fracpol_chi2_avg = temp.flatten()[noise_box_greater[0]]

    else:
        print(f'\nWARNING!!!\nNo usable pixels in this source...')
        source_ok = False

        wrm_dev = 0
        wrm_avg = 0
        wrm_rms = 0
        w_chi_2 = 0
        n_pixels = 0
        w_ston = 0
        frac_pol = 0
        w_prob_t = 0
        w_pi = 0
        w_si = 0
        fracpol_chi2_avg = 0

    return wrm_dev, wrm_avg, wrm_rms, w_chi_2, n_pixels, w_ston, source_ok, frac_pol, w_prob_t, w_pi, w_si, fracpol_chi2_avg


# PLOT 1 - POL. INT. MAP PART 2
def display_and_store2(w_rmf, w_drmf, w_chi2f, n_pixels, frac_pol_av, degrees, chitable, min_pol_threshold, w_prob_t,
                       x_long, y_lat, xpixmax, ypixmax, x_loc, y_loc, source_num, source_flag):
    """This function generates a series of strings that will be displayed in the Polarised Intensity Map.
    """
    rm_text = str(mf.truncate(w_rmf))
    rm_err_text = str(mf.truncate(w_drmf))

    chi_value = w_chi2f
    chi_string = str(chi_value)
    chi_pos = chi_string.find('.')
    chi_string = chi_string[0:chi_pos + 5]

    mask = np.array(degrees) == n_pixels - 1
    chitable_string = str(chitable[mask][0])

    m_string = str(frac_pol_av)
    ms_pos = m_string.find('.')
    m_string = m_string[0:ms_pos + 5]

    if w_chi2f < chitable.flatten()[mask] and min_pol_threshold < frac_pol_av < 1.0 and n_pixels >= 5 and w_prob_t >= 10.0:
        passfail = True
    else:
        passfail = False

    x_loc = mf.nround(x_loc)
    y_loc = mf.nround(y_loc)

    # Check if fitted location is closest to selected candidate:
    temp_array = (x_long[0, xpixmax] - x_long[y_loc, x_loc]) ** 2 + (y_lat[ypixmax, 0] - y_lat[y_loc, x_loc]) ** 2
    temp = np.min(temp_array)
    temp_index = np.where(temp_array == temp)[0][0]

    if temp_index != source_num:
        source_flag = (source_flag | 128)
        passfail = False

    pi_plot_text = {'rm_text': rm_text,
                    'rm_err_text': rm_err_text,
                    'chi_string': chi_string,
                    'chitable_string': chitable_string,
                    'm_string': m_string,
                    'n_pixels': n_pixels,
                    'passfail': passfail}

    return passfail, source_flag, pi_plot_text


def calc_av_pa_err(cut_x_reg, cut_y_reg, cut_dpsi_a, cut_dpsi_b, cut_dpsi_c, cut_dpsi_d):
    """This function calculates the average and standard deviation of the error in the polarisation angle.
    """
    cut_x_reg = cut_x_reg.astype(int)
    cut_y_reg = cut_y_reg.astype(int)
    reg_shape = cut_x_reg.shape

    crd = 180.0 / np.pi

    t_a = np.sum(cut_dpsi_a[cut_y_reg, cut_x_reg]) * crd / reg_shape[0]
    t_b = np.sum(cut_dpsi_b[cut_y_reg, cut_x_reg]) * crd / reg_shape[0]
    t_c = np.sum(cut_dpsi_c[cut_y_reg, cut_x_reg]) * crd / reg_shape[0]
    t_d = np.sum(cut_dpsi_d[cut_y_reg, cut_x_reg]) * crd / reg_shape[0]

    t_t = np.array([t_a, t_b, t_c, t_d])

    av_err = np.mean(t_t)
    dav_err = np.std(t_t) / 2.0

    return av_err, dav_err


# PLOT 4 - RM PLOT
def plot_rm_full3(data, x_long, y_lat, delta, rm_value, drm_value, pa_err, dpa_err, ston):
    """This function prepares data to be displayed in the Rotation Measure Map.
    """
    rm_data = data

    mask = data < -500.0
    rm_data[mask] = -500.0

    mask = data > 500.0
    rm_data[mask] = 500.0

    mask = ston <= 3.0
    rm_data[mask] = 0.0

    min_x = np.min(x_long)
    max_x = np.max(x_long)
    min_y = np.min(y_lat)
    max_y = np.max(y_lat)

    d_2 = delta / 2.0

    rm_text = str(mf.truncate(rm_value))
    rm_err_text = str(mf.nround(drm_value))
    pa_text = str(mf.truncate(pa_err))
    pa_err_text = str(mf.nround(dpa_err))

    rm_plot_data = {'rm_data': rm_data,
                    'rm_text': rm_text,
                    'rm_err_text': rm_err_text,
                    'pa_text': pa_text,
                    'pa_err_text': pa_err_text,
                    'min_x': min_x,
                    'max_x': max_x,
                    'min_y': min_y,
                    'max_y': max_y,
                    'd_2': d_2}

    return rm_plot_data


# PLOT 3 - PEAK PIXEL LINEAR FIT
def plot_strongest_linfit(pi, ua, ub, uc, ud, qa, qb, qc, qd, dpsi_c, dpsi_d, whw, rotmeas, e_rotmeas, psi, prob_t):
    """This function prepares data to be displayed in the Peak Pixel Linear Fit.
    """
    pi_array = pi.flatten()[whw]
    temp = np.max(pi_array)
    temp_index = np.where(pi_array == temp)[0]  # imax

    lambda_2 = (3.0 * 10**8 / (1420.4060 * 10**6))**2

    q_pix = np.array([[[qa.flatten()[whw[temp_index]]]], [[qb.flatten()[whw[temp_index]]]], [[qc.flatten()[whw[temp_index]]]], [[qd.flatten()[whw[temp_index]]]]])
    u_pix = np.array([[[ua.flatten()[whw[temp_index]]]], [[ub.flatten()[whw[temp_index]]]], [[uc.flatten()[whw[temp_index]]]], [[ud.flatten()[whw[temp_index]]]]])

    pol_ang = (unwrap_angles(q_pix, u_pix) * 180 / np.pi).flatten()
    pol_err = np.array([dpsi_c.flatten()[whw[temp_index]], dpsi_c.flatten()[whw[temp_index]], dpsi_c.flatten()[whw[temp_index]], dpsi_d.flatten()[whw[temp_index]]]) * 180 / np.pi

    rm_pix = rotmeas.flatten()[whw[temp_index]]
    drm = e_rotmeas.flatten()[whw[temp_index]]
    pol_ang_0 = psi.flatten()[whw[temp_index]]
    probfit = prob_t.flatten()[whw[temp_index]]

    x_range = (np.arange(101) * (0.0455 - 0.0435) / 100) + 0.0435
    predicted = (rm_pix * (x_range - lambda_2) * 180 / np.pi + pol_ang_0)

    lin_fit_plot_data = {'pol_ang': pol_ang,
                         'rm_pix': rm_pix,
                         'drm': drm,
                         'pol_err': pol_err,
                         'predicted': predicted,
                         'probfit': probfit}

    return lin_fit_plot_data


def unwrap_angles(q_arr, u_arr):
    """This function is Jo-Anne's algorithm to unwrap adjacent channels.
    """
    shape = q_arr.shape
    result = np.zeros(shape)

    z = q_arr + (1j * u_arr)
    z_norm = z / np.abs(z)

    d_phase = z_norm[1:, :, :] * np.conj(z_norm[0:shape[0] - 1, :, :])
    d_phase = np.arctan(np.imag(d_phase), np.real(d_phase))
    result[0, :, :] = np.arctan(np.imag(z_norm[0, :, :]), np.real(z_norm[0, :, :]))

    result_0 = result[0, :, :]
    result_0 = np.where(np.isnan(result_0), 0, result_0)  # Setting NaNs equal to zero
    result[0, :, :] = result_0

    for index in range(shape[0] - 1):
        result[index + 1, :, :] = result[index, :, :] + d_phase[index, :, :]

    result = result / 2.0

    return result


def parse_flag(flag):
    """This flag takes in the numerical form of the source flag and returns a list of flags (in words) that were active.
    """
    flag_list = []

    # & is the python bitwise AND
    # If you & the full flag with a flag that you want to check, the result will be 0 if it isn't set
    # and equal to the flag if it is set.
    # e.g. 5 & 1 = 1, meaning the 1 flag is set, and 5 & 4 = 4, meaning the 4 flag is set,
    # while 5 & 1024 = 0, meaning the 1024 flag is not set
    if flag & 1 != 0:
        flag_list.append('Revisit later')
    if flag & 2 != 0:
        flag_list.append('Manual fail')
    if flag & 4 != 0:
        flag_list.append('Gaussian fit failed')
    if flag & 8 != 0:
        flag_list.append('Too few/many pixels with sufficiant S:N')
    if flag & 16 != 0:
        flag_list.append('Fractional polarization too low/high.')
    if flag & 32 != 0:
        flag_list.append('Failed RM-averaging Chi-square test')
    if flag & 64 != 0:
        flag_list.append('Failed average linfit Chi-square test')
    if flag & 128 != 0:
        flag_list.append('False detection from neighbour')
    if flag & 256 != 0:
        flag_list.append('Mark presence of unidenfified double')
    if flag & 512 != 0:
        flag_list.append('Go back one source')
    if flag & 1024 != 0:
        flag_list.append('Retry source')

    return flag_list


def is_mod_flag_valid(input_flag, morph):
    """This function performs some basic error validation to determine if the user input when setting/unsetting flags is valid.
    For example, '16', '5', and '13' are all valid inputs, but 'bread', '-9999', or '3.14159' are not valid inputs.
    """
    flag_valid = True

    try:  # If the input flag is a number (a float or an integer)
        float_flag = float(input_flag)

        if not float_flag.is_integer():  # If the input flag is a float but not an integer (e.g. 3.14159), the input is invalid
            flag_valid = False
        else:  # If the input flag is an integer
            if (not morph) and (float_flag < 0 or float_flag > 2047):  # The maximum value of a flag alteration is the sum of all the possible flags
                flag_valid = False
            if morph and (float_flag < 0 or float_flag > 31):  # The maximum value of a morphology flag alteration is the sum of all the possible morphology flags
                flag_valid = False

    except ValueError:  # If the input flag is not a number (e.g. 'bread'), the input is invalid
        flag_valid = False

    return flag_valid


def main(input_directory, output_directory, chitable_directory):
    print(f'input directory: {input_directory}')
    print(f'output directory: {output_directory}')

    print(f'\n\nVERIFY THESE INPUTS: ')
    stokes_i_threshold = 1.2 / 1000
    min_pol_threshold = 0.02
    print(f'\n\nCurrent minimum polarisation threshold is: {min_pol_threshold}')
    alpha = 0.003  # fraction of I to be included in PA error
    print(f'\nAlpha factor in PA error calculations: {alpha}')
    output_ext = 'final_003I'
    print(f'\nOutput file extension: {output_ext}')
    edge_threshold = 5.0
    print(f'\nEdge threshold is {edge_threshold} sigma')
    fwxm_v = 2.0
    print(f'\nCurrently set to FULL WIDTH, MALF MAX')
    fwxm = np.sqrt(2.0 * np.log(fwxm_v))
    print(f'FWXM factor: {fwxm}')

    # **************************************************************************
    # READING IN THE DATA:
    # **************************************************************************

    # Reading in data from PI  table
    mosaic_name = ''
    mosaic_caps = ''
    mosaic_path_exists = False
    while not mosaic_path_exists:
        mosaic_caps = input('\nEnter the name of the mosaic you would like to analyze in all caps [MA1]: ')
        if mosaic_caps == '':
            mosaic_caps = 'MA1'
        mosaic_lower = mosaic_caps.lower()
        if mosaic_lower[0] == 'm':
            mosaic_name = mosaic_lower[1:]
        else:
            mosaic_name = mosaic_lower

        mosaic_path = f'{input_directory}{mosaic_name}'
        if not os.path.isdir(mosaic_path):
            print(f'\nThere is no input data corresponding to this mosaic. Please try again.')
        else:
            mosaic_path_exists = True
            print(f'\nCalculations for {mosaic_caps}:')

    sourcelist_name = input('Enter the table to use [_Taylor17_candidates]: ')
    if sourcelist_name == '':
        sourcelist_name = '_Taylor17_candidates'

    sourcelist_path = f'{output_directory}{mosaic_name}/M{mosaic_name.upper()}{sourcelist_name}.dat'  # Default: ../Data/output_data/a1/MA1_Taylor17_candidates.dat

    if not os.path.exists(sourcelist_path):
        print(f'\nThere is no polarised candidate sourcelist for this mosaic. One must be generated before Rotation Measure analysis can begin.')
    else:
        pi_data = read_pi_source_list_cd(sourcelist_path)
        num_sources = pi_data['len']

        # Reading in data from Stokes Q and U FITS files

        stokes, header = read_qu_data_cve(input_directory, mosaic_name)

        # Readying in from Chi Table

        degrees, chi2, chitable = read_chitable(chitable_directory)

        # **************************************************************************
        # DEFINING THE DATA SETS:
        # **************************************************************************

        lc = np.zeros(num_sources)
        bc = np.zeros(num_sources)
        wrmc = np.zeros(num_sources)
        wdrmc = np.zeros(num_sources)
        rmsc = np.zeros(num_sources)
        pic = np.zeros(num_sources)
        snc = np.zeros(num_sources)
        mc = np.zeros(num_sources)
        chi2c = np.zeros(num_sources)
        npixels = np.zeros(num_sources)
        sic = np.zeros(num_sources)
        probchi = np.zeros(num_sources)
        dpaav = np.zeros(num_sources)
        flag = np.zeros(num_sources)
        morphology = np.zeros(num_sources)
        rm_peakpix = np.zeros(num_sources)
        drm_peakpix = np.zeros(num_sources)
        fracpol_chi2_src = np.zeros(num_sources)

        x_array, y_array, x_long, y_lat = fh.make_xy_arrays(header['I'])

        # **************************************************************************
        # DETERMINING THE FREQUENCIES FOR THE FOUR BANDS:
        # **************************************************************************

        crd = 180 / np.pi

        freq_a = float(header['U_A']['OBSFREQ'])
        freq_b = float(header['U_B']['OBSFREQ'])
        freq_c = float(header['U_C']['OBSFREQ'])
        freq_d = float(header['U_D']['OBSFREQ'])

        la2 = (3 * 10 ** 8 / freq_a) ** 2
        lb2 = (3 * 10 ** 8 / freq_b) ** 2
        lc2 = (3 * 10 ** 8 / freq_c) ** 2
        ld2 = (3 * 10 ** 8 / freq_d) ** 2

        lx = [la2, lb2, lc2, ld2]

        freq_ok = check_freq2(freq_a, freq_b, freq_c, freq_d, header['Q_A'], header['Q_B'], header['Q_C'], header['Q_D'])
        # NOTE: Some mosaics (MM1-2, MN1-2) have a problem with the band C frequences. It's safe to skip past, because the correct numbers are used.

        if not freq_ok:
            do_continue = input(f'Do you want to continue anyway? (Y/N) [N] ')
            if do_continue.lower() == 'y' or do_continue.lower() == 'yes':
                do_continue = True
            else:
                do_continue = False
        else:
            do_continue = True

        if do_continue:
            lambda_default = 3 * 10 ** 8 / (1420.4060 * 10 ** 6)
            lambda_2 = lambda_default ** 2

            # **************************************************************************
            # THINGS FOR PSS_SUB:
            # **************************************************************************
            delta = np.abs(float(header['I']['CDELT1']))

            # **************************************************************************
            print('\nBeginning Calculations...\n')
            # **************************************************************************

            box_halfwidth = 20
            pi_units = 'mJy/beam'

            i = 0
            while i < num_sources:
                source_num = i
                source_flag = 0

                print(f'Calculations for source #{source_num}')

                xpix_max_i = pi_data['xpixmax'][i]
                ypix_max_i = pi_data['ypixmax'][i]

                # Because the box sizes are different between the noise calculation and the Gauss fit,
                # it's convenient to define a full-size PI array, fill part of it with the FG-subtracted and
                # de-biased PI, then feed that into the old code so it can extract its own box.
                # Same for S:N array and noise
                temp_pi = np.zeros(x_array.shape)
                ston = np.zeros(x_array.shape)
                noise = np.zeros(x_array.shape)

                x_min = mf.nround(xpix_max_i) - box_halfwidth
                x_max = mf.nround(xpix_max_i) + box_halfwidth + 1
                y_min = mf.nround(ypix_max_i) - box_halfwidth
                y_max = mf.nround(ypix_max_i) + box_halfwidth + 1

                stamp = {}
                arrays = [x_array, y_array, stokes['I'], stokes['Q_A'], stokes['Q_B'], stokes['Q_C'], stokes['Q_D'], stokes['U_A'], stokes['U_B'], stokes['U_C'], stokes['U_D']]
                array_label = ['xarr', 'yarr', 'I', 'Q_A', 'Q_B', 'Q_C', 'Q_D', 'U_A', 'U_B', 'U_C', 'U_D']
                for array in range(len(arrays)):
                    stamp[array_label[array]] = fh.cut_out_stamp(arrays[array],
                                                                 mf.nround(xpix_max_i) - box_halfwidth,
                                                                 mf.nround(xpix_max_i) + box_halfwidth,
                                                                 mf.nround(ypix_max_i) - box_halfwidth,
                                                                 mf.nround(ypix_max_i) + box_halfwidth)
                    # The stamps have been stored in a dictionary, just like the fits data was. To access simply call, for example, stamp['xarr'], or stamp['I']

                # Noise calculations:
                # The main purpose of this is to generate the noise-arr. The foreground subtraction isn't used here.
                g_coords = SkyCoord(l=pi_data['lmax'][i], b=pi_data['bmax'][i], frame='galactic', unit='degree')
                ra = g_coords.fk5.ra.deg
                dec = g_coords.fk5.dec.deg

                annulus_pixels = ac.calculate_annulus(ra, dec, stamp['xarr'], stamp['yarr'], xpix_max_i, ypix_max_i, stamp['I'], stokes_i_threshold)
                foreground_pixels = annulus_pixels[0]

                foreground_vector, sigma_qu = ac.estimate_local_noise(foreground_pixels,
                                                                      stamp['Q_A'],
                                                                      stamp['Q_B'],
                                                                      stamp['Q_C'],
                                                                      stamp['Q_D'],
                                                                      stamp['U_A'],
                                                                      stamp['U_B'],
                                                                      stamp['U_C'],
                                                                      stamp['U_D'])

                pi_debiased, noise_arr, ston_arr = ac.construct_new_ston_cutout(stamp['I'],
                                                                                stamp['Q_A'],
                                                                                stamp['Q_B'],
                                                                                stamp['Q_C'],
                                                                                stamp['Q_D'],
                                                                                stamp['U_A'],
                                                                                stamp['U_B'],
                                                                                stamp['U_C'],
                                                                                stamp['U_D'],
                                                                                foreground_vector,
                                                                                sigma_qu)

                temp_pi[y_min:y_max, x_min:x_max] = pi_debiased * 1000
                ston[y_min:y_max, x_min:x_max] = ston_arr
                noise[y_min:y_max, x_min:x_max] = noise_arr

                (x_a_2, y_a_2, whw, w3s, x_loc, y_loc, x_reg, y_reg, x_tick_vals, y_tick_vals, x_fwhm, y_fwhm,
                 w_annulus, gauss_parameters, source_ok, pi_plot_data, source_flag) = pss_subplot_get4_cve(temp_pi,
                                                                                                           x_array,
                                                                                                           y_array,
                                                                                                           x_long,
                                                                                                           y_lat,
                                                                                                           delta,
                                                                                                           pi_units,
                                                                                                           xpix_max_i,
                                                                                                           ypix_max_i,
                                                                                                           fwxm,
                                                                                                           source_num,
                                                                                                           source_flag)

                si_plot_data = pss_stokes_i_plot(stokes['I'] * 1000, x_long, y_lat, xpix_max_i, ypix_max_i,
                                                 gauss_parameters, fwxm, pi_data['xpixmax'], pi_data['ypixmax'])

                # do_flagging = input(f'Do you want to activate pixel flagging? (Y/N) [N]: ')
                # do_flagging = ''
                # if do_flagging.lower() == 'y' or do_flagging.lower() == 'yes':
                #     good_fit = False
                #     while not good_fit:
                #         # Fit Gaussian to PI, get annulus
                #         (x_a_2, y_a_2, whw, w3s, x_loc, y_loc, x_reg, y_reg, x_tick_vals, y_tick_vals, x_fwhm, y_fwhm,
                #          w_annulus, gauss_parameters, source_ok, pi_plot_data, source_flag) = pss_subplot_get4_cve(temp_pi,
                #                                                                                                    x_array,
                #                                                                                                    y_array,
                #                                                                                                    x_long,
                #                                                                                                    y_lat,
                #                                                                                                    delta,
                #                                                                                                    pi_units,
                #                                                                                                    xpix_max_i,
                #                                                                                                    ypix_max_i,
                #                                                                                                    fwxm,
                #                                                                                                    pi_data["lmax"][source_num],
                #                                                                                                    pi_data["bmax"][source_num],
                #                                                                                                    source_num,
                #                                                                                                    source_flag)
                #     si_plot_data = pss_stokes_i_plot(stokes['I'] * 1000, x_long, y_lat,
                #                                      xpix_max_i, ypix_max_i, gauss_parameters, fwxm, pi_data['xpixmax'], pi_data['ypixmax'])

                if whw.shape[0] == 0:
                    source_ok = False
                if w_annulus.shape[0] == 0:
                    source_ok = False
                if not source_ok:
                    # If the 4 flag (Gaussian fit failed) is not set already, set it
                    source_flag = (source_flag | 4)

                if source_ok:
                    print('\nSource OK in Main program...')

                    (cut_pi, cut_i, cut_qa, cut_qb, cut_qc, cut_qd,
                     cut_ua, cut_ub, cut_uc, cut_ud, cut_xarr, cut_yarr,
                     cut_noise, cut_x_long, cut_y_lat, cut_ston,
                     cut_x_loc, cut_y_loc, cut_x_reg, cut_y_reg,
                     x3s, y3s, cut_x_ann, cut_y_ann) = get_subarrays_cve(stokes['I'],
                                                                         stokes['Q_A'],
                                                                         stokes['Q_B'],
                                                                         stokes['Q_C'],
                                                                         stokes['Q_D'],
                                                                         stokes['U_A'],
                                                                         stokes['U_B'],
                                                                         stokes['U_C'],
                                                                         stokes['U_D'],
                                                                         x_long,
                                                                         y_lat,
                                                                         x_loc,
                                                                         y_loc,
                                                                         x_a_2,
                                                                         y_a_2,
                                                                         delta,
                                                                         temp_pi,
                                                                         ston,
                                                                         whw,
                                                                         w3s,
                                                                         noise,
                                                                         w_annulus)

                    stamp['xarr'] = cut_xarr
                    stamp['yarr'] = cut_yarr

                    if not source_ok:  # I don't think this conditional will ever be true
                        # If the 8 flag (Too few/many pixels with sufficient S:N) is not set already, set it
                        source_flag = (source_flag | 8)

                    # **************************************************************************
                    # CALCULATIONS WITH BACKGROUND REMOVED:
                    # **************************************************************************

                    print(f'\n**** BACKGROUND CALCS ****')

                    cut_qa, cut_qb, cut_qc, cut_qd, cut_ua, cut_ub, cut_uc, cut_ud = sbg_av3(cut_qa, cut_qb, cut_qc, cut_qd, cut_ua, cut_ub, cut_uc, cut_ud, cut_x_ann, cut_y_ann)

                    print(f'\n**** ----------------- ****')

                    # Error in polarisation angles:
                    cut_dpsi_a = cut_noise / (2 * np.sqrt(cut_ua ** 2 + cut_qa ** 2))
                    cut_dpsi_b = cut_noise / (2 * np.sqrt(cut_ub ** 2 + cut_qb ** 2))
                    cut_dpsi_c = cut_noise / (2 * np.sqrt(cut_uc ** 2 + cut_qc ** 2))
                    cut_dpsi_d = cut_noise / (2 * np.sqrt(cut_ud ** 2 + cut_qd ** 2))

                    rm_array_rbg, rm_error_rbg, psi_rbg, prob_t = unwrap_pa_and_rm_calcs4(cut_qa, cut_qb, cut_qc, cut_qd, cut_ua, cut_ub, cut_uc, cut_ud,
                                                                                          lambda_2, crd, cut_dpsi_a, cut_dpsi_b, cut_dpsi_c, cut_dpsi_d,
                                                                                          stamp['xarr'], stamp['yarr'], lx)

                    frac_pol_chi2_arr = fractional_polarisation_test(cut_i, cut_qa, cut_qb, cut_qc, cut_qd, cut_ua, cut_ub, cut_uc, cut_ud, cut_noise)

                    (w_drmf_rbg, w_rmf_rbg, w_rm_rmsf_rbg, w_chi2f_rbg, n_pixels_rbg, w_ston_rbg, source_ok,
                     frac_pol_av, w_prob_t, w_pi, w_si, fracpol_chi2_avg) = avg_rm_calcs8(cut_pi,
                                                                                          rm_array_rbg,
                                                                                          rm_error_rbg,
                                                                                          cut_ston,
                                                                                          edge_threshold,
                                                                                          cut_x_reg,
                                                                                          cut_y_reg,
                                                                                          cut_i * 1000,
                                                                                          prob_t,
                                                                                          frac_pol_chi2_arr)

                    if n_pixels_rbg > 100:  # Sometimes this part of the code fits huge Gaussians with > 100 pixels and the chisq table can't handle it
                        source_ok = False
                        # If the 4 flag (Gaussian fit failed) is not set already, set it
                        source_flag = (source_flag | 4)

                    if not source_ok:
                        # If the 8 flag (Too few/many pixels with sufficient S:N) is not set already, set it
                        source_flag = (source_flag | 8)

                    if source_ok:
                        pass_fail, source_flag, pi_plot_text = display_and_store2(w_rmf_rbg, w_drmf_rbg, w_chi2f_rbg, n_pixels_rbg, frac_pol_av,
                                                                                  degrees, chitable, min_pol_threshold, w_prob_t, x_long, y_lat,
                                                                                  pi_data['xpixmax'], pi_data['ypixmax'], x_loc, y_loc, i, source_flag)

                        av_err, dav_err = calc_av_pa_err(cut_x_reg, cut_y_reg, cut_dpsi_a, cut_dpsi_b, cut_dpsi_c, cut_dpsi_d)

                        rm_plot_data = plot_rm_full3(rm_array_rbg, cut_x_long, cut_y_lat, delta, w_rmf_rbg, w_drmf_rbg, av_err, dav_err, cut_ston)

                        linear_fit_plot_data = plot_strongest_linfit(cut_pi, cut_ua, cut_ub, cut_uc, cut_ud, cut_qa, cut_qb, cut_qc, cut_qd,
                                                                     cut_dpsi_c, cut_dpsi_d, whw, rm_array_rbg, rm_error_rbg, psi_rbg, prob_t)

                        pi_array = cut_pi.flatten()[whw]
                        temp = np.max(pi_array)
                        temp_index = np.where(pi_array == temp)[0]

                        rm_peakpix[source_num] = rm_array_rbg.flatten()[whw[temp_index]]
                        drm_peakpix[source_num] = rm_error_rbg.flatten()[whw[temp_index]]

                        # Recording the rest of the data for output tables

                        x_loc = mf.nround(x_loc)
                        y_loc = mf.nround(y_loc)

                        lc[source_num] = x_long[y_loc, x_loc]
                        bc[source_num] = y_lat[y_loc, x_loc]
                        wrmc[source_num] = w_rmf_rbg
                        wdrmc[source_num] = w_drmf_rbg
                        rmsc[source_num] = w_rm_rmsf_rbg
                        dpaav[source_num] = av_err
                        chi2c[source_num] = w_chi2f_rbg
                        npixels[source_num] = n_pixels_rbg
                        sic[source_num] = w_si
                        mc[source_num] = frac_pol_av
                        pic[source_num] = w_pi
                        snc[source_num] = w_ston_rbg
                        probchi[source_num] = w_prob_t
                        fracpol_chi2_src[source_num] = fracpol_chi2_avg

                        if n_pixels_rbg < 5:
                            # If the 8 flag (Too few/many pixels with sufficient S:N) is not set already, set it
                            source_flag = (source_flag | 8)
                        if frac_pol_av <= min_pol_threshold or frac_pol_av > 1.0:
                            # If the 16 flag (Fractional polarisation too low/high) is not set already, set it
                            source_flag = (source_flag | 16)
                        if w_chi2f_rbg >= chitable[n_pixels_rbg - 1]:
                            # If the 32 flag (Failed RM-averaging Chi-square test) is not set already, set it
                            source_flag = (source_flag | 32)
                        if w_prob_t < 10.0:
                            # If the 64 flag (Failed average linfit Chi-square test) is not set already, set it
                            source_flag = (source_flag | 64)

                        # **************************************************************************
                        # PLOTTING / FLAGGING SECTION:
                        # **************************************************************************

                        plt.style.use('dark_background')
                        plt.rcParams['figure.constrained_layout.use'] = True
                        fig = plt.figure(figsize=(13, 10), dpi=80)
                        spec2 = gridspec.GridSpec(ncols=2, nrows=2, figure=fig)
                        ax1 = fig.add_subplot(spec2[0, 0])
                        ax2 = fig.add_subplot(spec2[0, 1])
                        ax3 = fig.add_subplot(spec2[1, 0])
                        ax4 = fig.add_subplot(spec2[1, 1])

                        # *************************************
                        # POL INT MAP
                        # *************************************

                        plot_pol_int_map(pi_plot_data['x_l_2'], pi_plot_data['y_b_2'],
                                         pi_plot_data['data_fit'],
                                         pi_plot_data['levels'],
                                         mosaic_name,
                                         pi_plot_data['num'],
                                         pi_plot_data['x_gauss_rot'], pi_plot_data['y_gauss_rot'],
                                         pi_plot_data['x_center_gauss'], pi_plot_data['y_center_gauss'],
                                         x_long, y_lat,
                                         x_loc, y_loc,
                                         pi_plot_data['x_fwxm_ae'], pi_plot_data['y_fwxm_ae'],
                                         pi_plot_data['x_fwxm_se'], pi_plot_data['y_fwxm_se'],
                                         pi_plot_text['rm_text'], pi_plot_text['rm_err_text'],
                                         pi_plot_text['chi_string'],
                                         pi_plot_text['chitable_string'],
                                         pi_plot_text['m_string'],
                                         pi_plot_text['n_pixels'],
                                         # pi_plot_text['passfail'],
                                         pass_fail,
                                         ax1)

                        # *************************************
                        # STOKES I MAP
                        # *************************************

                        plot_stokes_i_map(si_plot_data['long_arr'], si_plot_data['lat_arr'],
                                          si_plot_data['data'],
                                          si_plot_data['levels'],
                                          si_plot_data['x_gauss_rot1'], si_plot_data['y_gauss_rot1'],
                                          si_plot_data['x_gauss_rot2'], si_plot_data['y_gauss_rot2'],
                                          gauss_parameters,
                                          source_flag,
                                          x_long, y_lat,
                                          xpix_max_i, ypix_max_i,
                                          si_plot_data['npix'],
                                          ax2)

                        # *************************************
                        # PEAK PIXEL LINEAR FIT
                        # *************************************

                        plot_peak_pixel_linear_fit(lx,
                                                   linear_fit_plot_data['pol_ang'],
                                                   linear_fit_plot_data['rm_pix'], linear_fit_plot_data['drm'],
                                                   linear_fit_plot_data['pol_err'],
                                                   linear_fit_plot_data['predicted'],
                                                   linear_fit_plot_data['probfit'],
                                                   ax3)

                        # *************************************
                        # RM MAP
                        # *************************************

                        plot_rm_map(rm_plot_data['rm_data'],
                                    rm_plot_data['rm_text'], rm_plot_data['rm_err_text'],
                                    rm_plot_data['pa_text'], rm_plot_data['pa_err_text'],
                                    pi_units,
                                    gauss_parameters,
                                    cut_x_long, cut_y_lat,
                                    cut_pi,
                                    x_fwhm, y_fwhm,
                                    pass_fail,
                                    w_prob_t,
                                    ax4)

                        # Making the plots show up, the code will keep running until the plt.show() line, at which point execution will pause
                        # In order to continue on to the next source, the plot window will have to be exited manually
                        plt.show(block=False)

                        # Flags:
                        print('\n\nFlag key: 0 - Good')
                        print('          1 - Revisit later')
                        print('          2 - Manual fail (+4 for morphology mismatch, +8 for no PI contrast, +32 for gradient)')
                        print('          4 - Gaussian fit failed')
                        print('          8 - Too few/many pixels with sufficient S:N')
                        print('         16 - Fractional polarization too low/high')
                        print('         32 - Failed RM-averaging Chi-square test')
                        print('         64 - Failed average linfit Chi-square test')
                        print('        128 - False detection from neighbour')
                        print('        256 - Mark presence of unidenfified double (please combine with "1" flag))')
                        print('        512 - Go back one source')
                        print('       1024 - Retry source')

                        print(f'\nCurrent flag: {source_flag}')
                        for ind_flag in parse_flag(source_flag):
                            print(f'            - {ind_flag}')

                        mod_flag = ''
                        mod_flag_valid = False
                        while not mod_flag_valid:
                            mod_flag = input(f'Set/unset flags? [no change]: ')
                            if mod_flag == '':
                                mod_flag = 0
                                mod_flag_valid = True
                            else:
                                if is_mod_flag_valid(mod_flag, morph=False):
                                    mod_flag_valid = True
                                else:
                                    print('\nInput is invalid, please press return or enter an integer between 0 and 2047.\n')

                        # Alter flag
                        mod_flag = int(mod_flag)
                        # ^ is the python bitwise XOR, it will set any flags that the user wants set and unset any flags that the user wants unset.
                        # e.g. 1 ^ 9 = 8, as the 9 (1 + 8) will unset the 1 flag since it is already set, and set the 8 flag since it isn't set yet.
                        flag[i] = (source_flag ^ mod_flag)

                        if source_flag & 512:  # If the user set the 'Go back one source' flag
                            i -= 1
                            continue  # De-increment to previous source
                        if source_flag & 1024:  # If the user set the 'Retry source' flag
                            continue  # Don't increment to next source, meaning we run through this source again

                        if flag[i] == 0:  # If there are no issues with the source
                            print('\n\n --------------------------------')
                            print('\n Set morphology flag(s):')
                            print(' 0 - Unresolved, isolated source and polarisation matches Stokes I')
                            print(' 1 - (Clearly) Extended source')
                            print(' 2 - Resolved double/multiple')
                            print(' 4 - PI is subset of Stokes I shape')
                            print(' 8 - Additional polarized component(s) seen')
                            print('16 - Offset between PI and I')

                            mod_morph_flag_valid = False
                            while not mod_morph_flag_valid:
                                morphology_flag = input('Set flags? [0]: ')
                                if morphology_flag == '':
                                    morphology_flag = 0
                                    morphology[i] = morphology_flag
                                    mod_morph_flag_valid = True
                                else:
                                    if is_mod_flag_valid(morphology_flag, morph=True):
                                        mod_morph_flag_valid = True
                                    else:
                                        print('\nInput is invalid, please press return or enter an integer between 0 and 31.\n')

                        print('\nPlease close the plot window to continue to the next source.\n\n')

                        plt.show()

                    else:
                        print('\n\nProblems with this source - IGNORED\n\n')
                        lc[source_num] = pi_data['lmax'][i]
                        bc[source_num] = pi_data['bmax'][i]
                else:
                    print('\n\nProblems with this source - IGNORED\n\n')
                    lc[source_num] = pi_data['lmax'][i]
                    bc[source_num] = pi_data['bmax'][i]

                i += 1  # Move on to next source

            # **************************************************************************
            # WRITING OUT THE RESULTS:
            # **************************************************************************

            outname = input('Enter output file name [recalc]: ')
            if outname == '':
                outname = 'recalc'

            out_dir = f'{output_directory}{mosaic_name}'

            # If the (mosaic specific) directory we want to put this list in doesn't exist yet, make it:
            Path(out_dir).mkdir(parents=True, exist_ok=True)

            file_out = f'_RMlist_{outname}.dat'
            out_path = f'{out_dir}/{mosaic_caps}{file_out}'
            out_path2 = f'{out_dir}/{mosaic_caps}_RMlist_{outname}_ONLY_GOOD.dat'

            # If either of these files already exists, we delete them so we can make them from scratch
            if os.path.exists(out_path):
                os.remove(out_path)
            if os.path.exists(out_path2):
                os.remove(out_path2)

            with open(out_path, 'w') as write_file, open(out_path2, 'w') as write_file2:
                write_file.write('l         b      WRM  WdRM   dPAav   prob   Chi^2   PI     SI       M     S:N  #pix   P/F  Morph   mChi2   RM_peak   dRM_peak')
                write_file.write(f'\n-- degrees --  -- rad/m^2 --  deg                 {pi_units}')
                write_file.write('\n')

                write_file2.write('l         b      WRM  WdRM   dPAav   prob   Chi^2   PI     SI       M     S:N  #pix   P/F  Morph   mChi2   RM_peak   dRM_peak')
                write_file2.write(f'\n-- degrees --  -- rad/m^2 --  deg                 {pi_units}')
                write_file2.write('\n')

                for i in range(num_sources):
                    write_file.write(f'\n{mf.string_normalise(str(round(lc[i], 3)), 7)}'
                                     f' {mf.string_normalise(str(round(bc[i], 3)), 6, negatives=True)}'
                                     f'  {mf.string_normalise(str(mf.nround(wrmc[i])), 4, negatives=True)}'
                                     f'    {mf.string_normalise(str(mf.nround(wdrmc[i])), 2)}'
                                     f'     {mf.string_normalise(str(mf.nround(dpaav[i])), 2)}'
                                     f'    {mf.string_normalise(str(round(probchi[i], 1)), 4)}'
                                     f'   {mf.string_normalise(str(round(chi2c[i], 3)), 5)} '
                                     f' {mf.string_normalise(str(round(pic[i], 2)), 6)}'
                                     f' {mf.string_normalise(str(round(sic[i], 2)), 6)}'
                                     f'  {mf.string_normalise(str(round(mc[i], 3)), 5)}'
                                     f'  {mf.string_normalise(str(round(snc[i], 2)), 5)}'
                                     f'  {mf.string_normalise(str(mf.nround(npixels[i])), 2)}'
                                     f'    {mf.string_normalise(str(mf.nround(flag[i])), 4)}'
                                     f'  {mf.string_normalise(str(mf.nround(morphology[i])), 2)} '
                                     f'    {mf.string_normalise(str(round(fracpol_chi2_src[i], 1)), 5)}'
                                     f'    {mf.string_normalise(str(mf.nround(rm_peakpix[i])), 4, negatives=True)}'
                                     f'       {mf.string_normalise(str(mf.nround(drm_peakpix[i])), 2)}')

                    if flag[i] == 0 and fracpol_chi2_src[i] <= 5:
                        write_file2.write(f'\n{mf.string_normalise(str(round(lc[i], 3)), 7)}'
                                          f' {mf.string_normalise(str(round(bc[i], 3)), 6, negatives=True)}'
                                          f'  {mf.string_normalise(str(mf.nround(wrmc[i])), 4, negatives=True)}'
                                          f'    {mf.string_normalise(str(mf.nround(wdrmc[i])), 2)}'
                                          f'     {mf.string_normalise(str(mf.nround(dpaav[i])), 2)}'
                                          f'    {mf.string_normalise(str(round(probchi[i], 1)), 4)}'
                                          f'   {mf.string_normalise(str(round(chi2c[i], 3)), 5)} '
                                          f' {mf.string_normalise(str(round(pic[i], 2)), 6)}'
                                          f' {mf.string_normalise(str(round(sic[i], 2)), 6)}'
                                          f'  {mf.string_normalise(str(round(mc[i], 3)), 5)}'
                                          f'  {mf.string_normalise(str(round(snc[i], 2)), 5)}'
                                          f'  {mf.string_normalise(str(mf.nround(npixels[i])), 2)}'
                                          f'    {mf.string_normalise(str(mf.nround(flag[i])), 4)}'
                                          f'  {mf.string_normalise(str(mf.nround(morphology[i])), 2)} '
                                          f'    {mf.string_normalise(str(round(fracpol_chi2_src[i], 1)), 5)}'
                                          f'    {mf.string_normalise(str(mf.nround(rm_peakpix[i])), 4, negatives=True)}'
                                          f'       {mf.string_normalise(str(mf.nround(drm_peakpix[i])), 2)}')

            # Output a list of sources that need to be revisited:
            revisit_list = np.bitwise_and(flag.astype(int), 1) == 1
            if np.sum(revisit_list) > 0:
                lmax = pi_data['lmax'][revisit_list]
                bmax = pi_data['bmax'][revisit_list]
                xpixmax = pi_data['xpixmax'][revisit_list]
                ypixmax = pi_data['ypixmax'][revisit_list]
                pimax = pi_data['pimax'][revisit_list]
                simax = pi_data['simax'][revisit_list]
                stonmax = pi_data['stonmax'][revisit_list]

                revisit_name = input('Enter output file name [revisit]: ')
                if revisit_name == '':
                    revisit_name = 'revisit'

                revisit_file_out = f'_sourcelist_{revisit_name}.dat'
                revisit_out_path = f'{out_dir}/{mosaic_caps}{revisit_file_out}'

                # If this file already exists, we delete it so we can make it from scratch
                if os.path.exists(revisit_out_path):
                    os.remove(revisit_out_path)

                with open(revisit_out_path, 'w') as write_file:
                    write_file.write(f'Polarised Intensity source list for {mosaic_caps}')
                    write_file.write(f'\n')
                    write_file.write(f'\n   l       b     xpix  ypix       PI       SI     S:N')
                    write_file.write(f'\n-- degrees --                  {pi_units}')
                    write_file.write(f'\n')

                    max_long = np.max(lmax)
                    while max_long > 0.0:
                        mask = mf.idl_where(lmax == max_long)
                        write_file.write(f'\n{mf.string_normalise(str(round(lmax.flatten()[mask[0]], 3)), 7)}'
                                         f' {mf.string_normalise(str(round(bmax.flatten()[mask[0]], 3)), 6, negatives=True)}'
                                         f'    {mf.string_normalise(str(mf.nround(xpixmax.flatten()[mask[0]])), 4)}'
                                         f' {mf.string_normalise(str(mf.nround(ypixmax.flatten()[mask[0]])), 4)}'
                                         f'     {mf.string_normalise(str(round(pimax.flatten()[mask[0]], 2)), 6)}'
                                         f'    {mf.string_normalise(str(round(simax.flatten()[mask[0]], 2)), 6)}'
                                         f' {mf.string_normalise(str(round(stonmax.flatten()[mask[0]], 2)), 5)}')
                        where_current = np.logical_and(lmax == lmax.flatten()[mask[0]], bmax == bmax.flatten()[mask[0]])
                        lmax[where_current] = 0.0
                        max_long = np.max(lmax)

            print('\nRotation Measure analysis complete!')
