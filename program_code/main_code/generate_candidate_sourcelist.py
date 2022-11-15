import numpy as np
from astropy.coordinates import SkyCoord
from pathlib import Path
import os
from main_code.subroutines import array_calculations as ac
from main_code.subroutines import fits_handling as fh
from main_code.subroutines import misc_functions as mf

boundary_size = 20
signal_threshold = 4.8
box_half_width = 20
outfile = 'Taylor17_candidates.dat'
stokes_I_threshold = 1.2 / 1000


def main(raw_data_path, output_data_path, taylor17_path):
    # Determining which mosaic to analyze

    mosaic_path = ''
    mosaic_path_exists = False
    while not mosaic_path_exists:
        mosaic_caps = input('Enter the name of the mosaic you would like to analyze in all caps [MA1]: ')
        if mosaic_caps == '':
            mosaic_name = 'a1'
        else:
            mosaic_lower = mosaic_caps.lower()
            if mosaic_lower[0] == 'm':
                mosaic_name = mosaic_lower[1:]
            else:
                mosaic_name = mosaic_lower

        mosaic_path = f'{raw_data_path}{mosaic_name}'

        if not os.path.isdir(mosaic_path):
            print(f'\nThere is no input data corresponding to this mosaic. Please try again.\n')
        else:
            mosaic_path_exists = True

    # Reading FITS files

    stokes = {}
    fits_header = None  # Initializing the header to None so the code below doesn't complain about it possibly not being initialized

    for band in ['I', 'Q_A', 'Q_B', 'Q_C', 'Q_D', 'U_A', 'U_B', 'U_C', 'U_D']:
        stokes_header, stokes_data = fh.read_fits(f'{mosaic_path}/m{mosaic_name}_1420_MHz_{band}_image.fits')

        if band == 'I':
            fits_header = stokes_header

        stokes[band] = stokes_data
        # The data from each fits file is now stored in a dictionary. To access the data simply call, for example, stokes['I'], or stokes['Q_A']

    # Calculating edges of mosaic

    # Left side (CDELT1 is negative)
    l_max = (boundary_size + 1 - fits_header['CRPIX1']) * fits_header['CDELT1'] + fits_header['CRVAL1']
    # Right side (CDELT1 is negative)
    l_min = (fits_header['NAXIS1'] - boundary_size - fits_header['CRPIX1']) * fits_header['CDELT1'] + fits_header['CRVAL1']
    # Bottom side
    b_min = (boundary_size + 1 - fits_header['CRPIX2']) * fits_header['CDELT2'] + fits_header['CRVAL2']
    # Top side
    b_max = (fits_header['NAXIS2'] - boundary_size - fits_header['CRPIX2']) * fits_header['CDELT2'] + fits_header['CRVAL2']

    # Reading Taylor17 catalogue

    cgps_ident = []
    ra = []
    dec = []
    int_i = []
    pa = []
    with open(taylor17_path, 'r') as taylor17_catalogue:
        for source in taylor17_catalogue:
            source_split = source.split()

            cgps_ident.append(source_split[0] + ' ' + source_split[1])
            ra.append(float(source_split[2]))
            dec.append(float(source_split[5]))
            int_i.append(float(source_split[8]))
            pa.append(int(source_split[25]))

    # Convert RA and Dec to galactic coordinates

    ra = np.array(ra)
    dec = np.array(dec)
    ra_dec_coords = SkyCoord(ra=ra, dec=dec, frame='fk5', unit='deg')
    g_long = ra_dec_coords.galactic.l.deg
    g_lat = ra_dec_coords.galactic.b.deg

    # Defining the PA vector in equatorial unit length

    # pa = np.array(pa)
    # mu_ra = np.sin(pa * np.pi / 180)
    # mu_dec = np.cos(pa * np.pi / 180)

    # Select sources in the field

    source_names = []
    source_lat = []
    source_long = []
    source_ra = []
    source_dec = []
    source_int_i = []
    for i in range(len(cgps_ident)):
        if l_max > g_long[i] > l_min and b_max > g_lat[i] > b_min:
            source_names.append(cgps_ident[i])
            source_lat.append(g_lat[i])
            source_long.append(g_long[i])
            source_ra.append(ra[i])
            source_dec.append(dec[i])
            source_int_i.append(int_i[i])

    print(f'\nOut of {len(cgps_ident)} Taylor 17 sources, {len(source_names)} are in the field of view')

    # Make pixel/coordinate location arrays

    x_array, y_array, x_long, y_lat = fh.make_xy_arrays(fits_header)

    long_value = fits_header['CRVAL1']
    long_pix = fits_header['CRPIX1']
    long_delta = fits_header['CDELT1']

    lat_value = fits_header['CRVAL2']
    lat_pix = fits_header['CRPIX2']
    lat_delta = fits_header['CDELT2']

    # Inspecting sources

    print('\nInspecting sources...')

    polarized_list = []
    ston_list = []
    peak_pi_list = []
    # pa_list = []
    for i in range(len(source_names)):  # [1505]:
        # Cut out box around source

        center_x = ((source_long[i] - long_value) / long_delta) + long_pix - 1  # The - 1 was in the IDL code, some kind of 'off by one' error I guess
        center_y = ((source_lat[i] - lat_value) / lat_delta) + lat_pix - 1

        stamp_names = ['xarr', 'yarr', 'xlong', 'ylat', 'I', 'Q_A', 'Q_B', 'Q_C', 'Q_D', 'U_A', 'U_B', 'U_C', 'U_D']
        arrays_to_stamp = [x_array, y_array, x_long, y_lat, stokes['I'],
                           stokes['Q_A'], stokes['Q_B'], stokes['Q_C'], stokes['Q_D'],
                           stokes['U_A'], stokes['U_B'], stokes['U_C'], stokes['U_D']]

        stamp = {}

        for stamp_name in range(len(stamp_names)):
            stamp[stamp_names[stamp_name]] = fh.cut_out_stamp(arrays_to_stamp[stamp_name],
                                                              mf.nround(center_x) - box_half_width,
                                                              mf.nround(center_x) + box_half_width,
                                                              mf.nround(center_y) - box_half_width,
                                                              mf.nround(center_y) + box_half_width)  # The + 1 is because IDL indexes arrays differently than python
            # The stamps have been stored in a dictionary, just like the fits data was. To access simply call, for example, stamp['xarr'], or stamp['I']

        # Calculate annulus

        annulus_pixels = ac.calculate_annulus(source_ra[i], source_dec[i], stamp['xarr'], stamp['yarr'], center_x, center_y, stamp['I'], stokes_I_threshold)
        foreground_pixels = annulus_pixels[0]
        source_pixels = annulus_pixels[1]

        if np.sum(np.isnan(stamp['Q_A'][foreground_pixels == 1])) < np.sum(foreground_pixels):
            # If the foreground area of the Stokes QA stamp has at least one valid pixel (isn't entirely NaNs)

            # Estimate local noise in annulus

            foreground_vector, sigma_qu = ac.estimate_local_noise(foreground_pixels,
                                                                  stamp['Q_A'],
                                                                  stamp['Q_B'],
                                                                  stamp['Q_C'],
                                                                  stamp['Q_D'],
                                                                  stamp['U_A'],
                                                                  stamp['U_B'],
                                                                  stamp['U_C'],
                                                                  stamp['U_D'])

            # Calculate polarized intensity (w/ foreground subtraction), S:N

            pi_debiased, noise_array, ston_array = ac.construct_new_ston_cutout(stamp['I'],
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

            # If it's above threshold, add to catalog

            # See how many on-source pixels have S:N > 5 (just checking source pixels, to exclude neighbours)

            # Check for peak S:N > 5 on source:
            source_ston = np.where(np.isnan(ston_array), -9999, ston_array) * np.where(np.isnan(source_pixels), -9999, source_pixels)
            peak_ston = np.amax(source_ston)
            peak_index_y, peak_index_x = np.where(source_ston == peak_ston)
        else:
            # If every pixel in the foreground area of the Stokes QA stamp is a NaN, we can't calculate anything useful using this source. We'll make some default values
            # so we know the source won't get selected
            peak_ston = -1 * 1.7976931348623157e+308  # The maximum negative value a float can have in Python
            pi_debiased = np.zeros(stamp['Q_A'].shape) - 1.7976931348623157e+308
            peak_index_y, peak_index_x = 20, 20

        if peak_ston >= signal_threshold:  # Just testing for peak S:N
            polarized_list.append(i)
            ston_list.append(peak_ston)
            peak_pi_list.append(pi_debiased[peak_index_y, peak_index_x])

    # Construct candidate data arrays:
    indices = np.array(polarized_list)
    polarized_ston = np.array(ston_list)
    ston_sort = polarized_ston.argsort()
    indices = indices[ston_sort[::-1]]  # Sorting sources by descending S:N
    polarized_ston = polarized_ston[ston_sort[::-1]]
    peak_pi_list = np.array(peak_pi_list)[ston_sort[::-1]]

    polarized_g_long = np.array(source_long)[indices]
    polarized_g_lat = np.array(source_lat)[indices]
    polarized_x_pixels = ((polarized_g_long - fits_header['CRVAL1']) / fits_header['CDELT1']) + fits_header['CRPIX1'] - 1
    polarized_y_pixels = ((polarized_g_lat - fits_header['CRVAL2']) / fits_header['CDELT2']) + fits_header['CRPIX2'] - 1
    polarized_stokes_i = np.array(source_int_i)[indices]
    polarized_pi = peak_pi_list * 1000  # peak_pi_list is in units of Jy / beam, we want mJy / beam

    out_dir = f'{output_data_path}{mosaic_name}'

    # If the (mosaic specific) directory we want to put this list in doesn't exist yet, make it:
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # Output to table of mosaic Stokes I sources
    with open(f'{out_dir}/M{mosaic_name.upper()}_Taylor17_candidates.dat', "w") as write_file:
        write_file.write(f'Polarized source candidate list for field M{mosaic_name.upper()}')
        write_file.write(f'\nGenerated from Taylor+17 catalog, threshold (sigma) = {signal_threshold}')
        write_file.write(f'\n ')
        write_file.write(f'\n    l        b    xpix  ypix       PI       SI      S/N')
        write_file.write(f'\n--  degrees   --                mJy/beam')
        write_file.write(f'\n ')

        for source in range(polarized_g_long.size):
            write_file.write(f'\n{mf.string_normalise(str(round(polarized_g_long[source], 3)), 7)}'
                             f'   {mf.string_normalise(str(round(polarized_g_lat[source], 3)), 6, negatives=True)}'
                             f'   {mf.string_normalise(str(mf.nround(polarized_x_pixels[source])), 4)}'
                             f'  {mf.string_normalise(str(mf.nround(polarized_y_pixels[source])), 4)}'
                             f'     {mf.string_normalise(str(round(polarized_pi[source, 0], 2)), 5)}'
                             f'    {mf.string_normalise(str(round(polarized_stokes_i[source], 2)), 5)}'
                             f'   {mf.string_normalise(str(round(polarized_ston[source], 2)), 5)}')

    print('\nCandidate sourcelist generated!')
