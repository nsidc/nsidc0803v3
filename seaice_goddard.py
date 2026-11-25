"""Implement the Goddard (NASA Team) sea ice concentration algorithm

Sample usage:

    python seaice_goddard.py psn25 20240315 20240315
  creates:
    /share/apps/nsidc0803/nsidc0803/dev/scotts/nt_0803v3/2024/03/nt_0803v3_psn25_20240315.dat

    python seaice_goddard.py pss25 20240315 20240315
  creates:
   /share/apps/nsidc0803/nsidc0803/dev/scotts/nt_0803v3/2024/03/nt_0803v3_pss25_20240315.dat
"""


import sys
import os
import numpy
from netCDF4 import Dataset
import dateutil.parser
import numpy as np
import datetime as dt


# This sets the ntsiconc to a uint8 value of 250
siconc_scaling = 2.5
maxconc = 100.0
c100value = siconc_scaling * maxconc
minconc = 0.0

gridid_to_hemi = {
    'psn25': 'north',
    'pss25': 'south',
}
# These are specific to the JAXA L1R files
#   chan res combines the channel and the resampled-resolution fields
chan_to_chanres = {
    'h19': 'h19r06',
    'v19': 'v19r06',
    'v22': 'v22r06',
    'h37': 'h37r06',
    'v37': 'v37r06',
}
gridid_to_H = {
    'psn25': 'N',
    'pss25': 'S',
}

nt_tiepoints = {
    'nsidc0001_f17': {
        'psn25': {
            'tp19vw': 184.9,
            'tp19vf': 248.4,
            'tp19vm': 220.7,
            'tp19hw': 113.4,
            'tp19hf': 232.0,
            'tp19hm': 196.0,
            'tp37vw': 207.1,
            'tp37vf': 242.3,
            'tp37vm': 188.5,
        },
        'pss25': {
            'tp19vw': 184.9,
            'tp19vf': 253.1,
            'tp19vm': 244.0,
            'tp19hw': 113.4,
            'tp19hf': 237.8,
            'tp19hm': 211.9,
            'tp37vw': 207.1,
            'tp37vf': 246.6,
            'tp37vm': 212.6,
        },
    },
    # These were initially just copied from the 0001 F17 coefficient
    #'sfmlr': {
    'cal': {
        'psn25': {
            'tp19vw': 184.9,
            'tp19vf': 248.4,
            'tp19vm': 220.7,
            'tp19hw': 113.4,
            'tp19hf': 232.0,
            'tp19hm': 196.0,
            'tp37vw': 207.1,
            'tp37vf': 242.3,
            'tp37vm': 188.5,
        },
        'pss25': {
            'tp19vw': 184.9,
            'tp19vf': 253.1,
            'tp19vm': 244.0,
            'tp19hw': 113.4,
            'tp19hf': 237.8,
            'tp19hm': 211.9,
            'tp37vw': 207.1,
            'tp37vf': 246.6,
            'tp37vm': 212.6,
        },
    },
}

def get_nt_tiepoints(gridid, date, source):
    """Return the NT tie points for these values"""
    tiepoints = nt_tiepoints[source][gridid]

    return tiepoints


# Note: this assumes only 25km grids will be accessed
NSIDC0001_LOWRES_GRIDS = {
    'psn25': 'N25km',
    'pss25': 'S25km',
}


def ncvar(sat, chan):
    """Return the netCDF4 variable name"""
    if chan == 'h19':
        return f'TB_{sat}_19H'
    elif chan == 'v19':
        return f'TB_{sat}_19V'
    elif chan == 'v22':
        return f'TB_{sat}_22V'
    elif chan == 'h37':
        return f'TB_{sat}_37H'
    elif chan == 'v37':
        return f'TB_{sat}_37V'
    else:
        raise RuntimeError(f'Could not calc ncvar: {sat=}, {chan=}')


def read_ncvar(ds, ncgroup, ncvar):
    return ds.groups[ncgroup].variables[ncvar]


def get_tbs_nsidc0001(
    gridid,
    date,
    tb_list,
    srcdir_='/home/scotts/Desktop/data/nsidc0001/{gridid}/{year}',
    bfn_='NSIDC0001_TB_PS_{nsidcgrid}_{ymd}_{ver}.nc',
    ver='v6.0',
    ncgroup='F17',
):
    """Return a dictionary of the 0001 TB fields"""
    year = date.strftime('%Y')
    ymd = date.strftime('%Y%m%d')
    nsidcgrid = NSIDC0001_LOWRES_GRIDS[gridid]

    srcdir = srcdir_.format(
        gridid=gridid,
        year=year,
    )
    bfn = bfn_.format(
        nsidcgrid=nsidcgrid,
        ymd=ymd,
        ver=ver,
    )

    fn = f'{srcdir}/{bfn}'
    try:
        assert os.path.isfile(fn)
    except AssertionError:
        print('no such 0001 file: {fn}')
        breakpoint()

    ds = Dataset(fn)
    tbs = {}
    for tb in tb_list:
        tbs[tb] = np.squeeze(read_ncvar(ds, ncgroup, ncvar(ncgroup, tb))).astype(np.float32)

    return tbs


def get_tbs_sfmlr(
    gridid,
    date,
    tb_list,
    srcdir_='/home/scotts/Desktop/data/tb_gw1am2_sfmlr/{gridid}/{year}/{month}',
    bfn_='tb_gw1am2_sfmlr_{gridid}_{chanres}_{ymd}.dat',
):
    """Return a dictionary of the 0001 TB fields"""
    year = date.strftime('%Y')
    month = date.strftime('%m')
    ymd = date.strftime('%Y%m%d')

    srcdir = srcdir_.format(
        gridid=gridid,
        year=year,
        month=month,
    )

    xdim, ydim = get_dims(gridid)
    tbs = {}
    for tb in tb_list:
        chanres = chan_to_chanres[tb]

        bfn = bfn_.format(
            gridid=gridid,
            chanres=chanres,
            ymd=ymd,
        )
        fn = f'{srcdir}/{bfn}'
        field = np.fromfile(fn, dtype=np.float32).reshape(ydim, xdim)

        tbs[tb] = field

    return tbs


def get_tbs_cal(
    gridid,
    date,
    tb_list,
    # TODO: This is an intentional "error" until the mount gets fixed
    #srcdir_='/share/apps/nsidc0802/nsidc0802/dev/scotts/tb_gw1am2_cal/{gridid}/{year}/{month}',
    srcdir_='/share/apps/nsidc0803/nsidc0802/dev/scotts/tb_gw1am2_cal/{gridid}/{year}/{month}',
    bfn_='tb_gw1am2_cal_{gridid}_{chanres}_{ymd}.dat'
):
    """Return a dictionary of the 0001 TB fields"""
    year = date.strftime('%Y')
    month = date.strftime('%m')
    ymd = date.strftime('%Y%m%d')

    srcdir = srcdir_.format(
        gridid=gridid,
        year=year,
        month=month,
    )

    xdim, ydim = get_dims(gridid)
    tbs = {}
    for tb in tb_list:
        chanres = chan_to_chanres[tb]

        bfn = bfn_.format(
            gridid=gridid,
            chanres=chanres,
            ymd=ymd,
        )
        fn = f'{srcdir}/{bfn}'
        field = np.fromfile(fn, dtype=np.float32).reshape(ydim, xdim)

        tbs[tb] = field

    return tbs


def get_tbs(gridid, date, source):
    """Return the necessary TB fields
          tb19h, tb19v, tb22v, tb37v
    """
    tb_list = ['h19', 'v19', 'v22', 'v37']
    tbs = None
    if source == 'nsidc0001_f17':
        tbs = get_tbs_nsidc0001(gridid, date, tb_list)
    elif source == 'sfmlr':
        tbs = get_tbs_sfmlr(gridid, date, tb_list)
    elif source == 'cal':
        tbs = get_tbs_cal(gridid, date, tb_list)
    else:
        raise RuntimeError(f'Cannot read TBs from source: {source}')

    # Add an array of "v19" with no zeros, for use in denominator calcs
    v19_no0 = tbs['v19'].copy()
    v19_no0[v19_no0 == 0] = 1
    tbs['v19_no0'] = v19_no0

    return tbs


def compute_invalid(tbs):
    """Return a mask where data values are invalid"""
    is_invalid = None
    for tb in tbs:
        if 'no0' in tb:
            break
        if is_invalid is None:
            is_invalid = tbs[tb] == 0
        else:
            is_invalid = is_invalid | (tbs[tb] == 0)

    return is_invalid


def lookup_gr_threshold(source, gridid, chan):
    """Determine the gradient ratio threshold
    For v22:   0.045
    For v37:
      F17, SH: 0.053
      other:   0.050
    """
    if chan == 'v22':
        # This is gr2
        return 0.045
    elif chan == 'v37':
        # This is gr1
        if (source == 'nsidc0001_f17') and (gridid == 'pss25'):
            # gr1 for F17 in SH
            return 0.053
        else:
            return 0.050

    raise RuntimeError(f'bad chan in lookup_gr_threshold()?: {chan}')


def calc_nt_weather(gridid, source, tb_comp_chan, tbs):
    """Calculate the NT weather filter"""
    gradient_ratio_threshold = lookup_gr_threshold(source, gridid, tb_comp_chan)

    v19 = tbs['v19']
    v19_no0 = tbs['v19_no0']
    tb_comp = tbs[tb_comp_chan]

    is_bad_tb = (v19 == 0) | (tb_comp == 0)

    gradient_ratio = (tb_comp - v19) / (tb_comp + v19_no0)

    # It's not weather if it's bad TB
    is_weather = (gradient_ratio > gradient_ratio_threshold) & (~is_bad_tb)

    return is_weather, gradient_ratio


def calc_polarization_ratio(tbs):
    """Return the polarization ratio"""
    v19 = tbs['v19']
    v19_no0 = tbs['v19_no0']
    h19 = tbs['h19']

    polarization_ratio = (v19 - h19) / (v19_no0 + h19)

    return polarization_ratio


def rescale(field, new_max):
    """Rescale from field with range of 0 to 1 ...
    to range of 0 to new_max
    """
    return field * new_max


def compute_nt_coefficients(gridid, source):
    """Define the "C" values that are used in the NASA Team algorithm
    Note: tiepoint keys are like:
        tp19vw
    """
    tiepoints = get_nt_tiepoints(gridid, date, source)

    # Differences: 19v vs 19h
    DW1 = tiepoints['tp19vw'] - tiepoints['tp19hw']
    DF1 = tiepoints['tp19vf'] - tiepoints['tp19hf']
    DM1 = tiepoints['tp19vm'] - tiepoints['tp19hm']

    # Sums: 19v vs 19h
    SW1 = tiepoints['tp19vw'] + tiepoints['tp19hw']
    SF1 = tiepoints['tp19vf'] + tiepoints['tp19hf']
    SM1 = tiepoints['tp19vm'] + tiepoints['tp19hm']

    # Differences: 37v vs 19h
    DW2 = tiepoints['tp37vw'] - tiepoints['tp19vw']
    DF2 = tiepoints['tp37vf'] - tiepoints['tp19vf']
    DM2 = tiepoints['tp37vm'] - tiepoints['tp19vm']

    # Sums: 37v vs 19h
    SW2 = tiepoints['tp37vw'] + tiepoints['tp19vw']
    SF2 = tiepoints['tp37vf'] + tiepoints['tp19vf']
    SM2 = tiepoints['tp37vm'] + tiepoints['tp19vm']

    # Define NASA Team coefficients as a dictionary
    #   for ease of indexing
    C = {}
    C[1] = DM1 * DW2 - DM2 * DW1
    C[2] = DM2 * SW1 - DW2 * SM1
    C[3] = DW1 * SM2 - DM1 * SW2
    C[4] = SM1 * SW2 - SM2 * SW1
    C[5] = DF1 * (DM2 - DW2) + DW1 * (DF2 - DM2) + DM1 * (DW2 - DF2)
    C[6] = DF2 * (SM1 - SW1) + DW2 * (SF1 - SM1) + DM2 * (SW1 - SF1)
    C[7] = DF1 * (SW2 - SM2) + DW1 * (SM2 - SF2) + DM1 * (SF2 - SW2)
    C[8] = SF2 * (SW1 - SM1) + SW2 * (SM1 - SF1) + SM2 * (SF1 - SW1)
    C[9] = DF2 * DW1 - DF1 * DW2
    C[10] = DW2 * SF1 - DF2 * SW1
    C[11] = SW2 * DF1 - DW1 * SF2
    C[12] = SF2 * SW1 - SF1 * SW2

    return C


def force_rov(field, minval, maxval, dtype=np.uint8):
    """Cause a field to have minimum and maximum values"""
    is_zero = field == 0
    is_nan = np.isnan(field)
    field[field < minval] = minval
    field[field > maxval] = maxval
    field[is_zero | is_nan] = 0

    if field.dtype != dtype:
        field = np.round(field)
        field = field.astype(dtype)

    return field


def get_dims(gridid):
    if gridid == 'psn25':
        return 304, 448
    elif gridid == 'pss25':
        return 316, 332


def get_landmask(gridid):
    xdim, ydim = get_dims(gridid)
    mask_fn = f'nt0051_landmask_{gridid}.dat'
    mask = np.fromfile(mask_fn, dtype=np.uint8).reshape(ydim, xdim)

    return mask


def get_nearsie(gridid, date):
    """SAmple:
    /Desktop/data/nsidc0079_siem/psn25/2024/nsidc0079_siemask_psn25_20240101.dat
    """
    nearsie_dir_ = '/home/scotts/Desktop/data/nsidc0079_siem/{gridid}/{year}'
    nearsie_bfn_ = 'nsidc0079_siemask_{gridid}_{ymd}.dat'

    year = date.strftime('%Y')
    ymd = date.strftime('%Y%m%d')
    nearsie_dir = nearsie_dir_.format(
        gridid=gridid,
        year=year,
    )
    nearsie_bfn = nearsie_bfn_.format(
        gridid=gridid,
        ymd=ymd,
    )

    xdim, ydim = get_dims(gridid)
    nearsie_fn = f'{nearsie_dir}/{nearsie_bfn}'
    nearsie = np.fromfile(nearsie_fn, dtype=np.uint8).reshape(ydim, xdim)

    return nearsie


def get_climate_mask(gridid, date):
    """Return the nsidc0051 mask for this grid and date
        icelakeclim.nsidc0622.N01_v1.byte
        icelakeclim.nasateam.S02_v2.byte
    """
    month = date.strftime('%m')
    srcdir = '/home/scotts/Desktop/calib_0802_0803/nt_siconc/nsidc0740_ancillary'

    if gridid == 'psn25':
        bfn = f'icelakeclim.nsidc0622.N{month}_v1.byte'
    elif gridid == 'pss25':
        bfn = f'icelakeclim.nasateam.S{month}_v2.byte'

    climate_mask_fn = f'{srcdir}/{bfn}'
    xdim, ydim = get_dims(gridid)

    climate_mask = np.fromfile(climate_mask_fn, dtype=np.uint8).reshape(ydim, xdim)

    return climate_mask


# def apply_nt_spillover(conc_orig, shoremap, minic, match_CDRv4_cdralgos=False):
def apply_nt_spillover(conc_orig, shoremap, minic):
    """
    Note: This was adapted from the routine in seaice_cdr
    but new adj123 (=shoremap) fields were calculated

    Note: The original code for matching CDRv4_cdralgos has been removed

    Apply the NASA Team land spillover routine.

    conc is 0 to > 100.0
      (per this code, negative value means missing data)

    shoremap is 0: ocean, 1: land, 2: coast,
                 3: coast, 4: near_coast, 5: far_coast
      in original, shoremap is big-endian int16, but nothing in the
        computation requires it to be anything other than int-like

    minic is 0 to 100.0
      (in original binary file, minic is 0-1000 which is conc * 10)
      but as included here, it is float64 with values 0 to 96.3 (=~100.0)

    It appears that there is an assumption that land is always >= 15% conc
    TODO: Perhaps this should be looking for ocean cells that are >= 15%?

    NOTE/TODO: the last time this code was run, the output did not look
    correct. There may be a problem with the algorithm, or with the input minic
    field. Review this code and output carefully to ensure it works as expected.
    NOTE/TODO: the last time this code was run, the output did not look
    correct. There may be a problem with the algorithm, or with the input minic
    field. Review this code and output carefully to ensure it works as expected.
    """
    ydim, xdim = conc_orig.shape

    # Create a 0 to 100 range version of conc_orig
    conc100 = np.zeros((ydim, xdim), dtype=np.uint8)
    is_conc = conc_orig <= 250
    # conc100[is_conc] = conc_orig[is_conc] / 2.5
    # print('NOTE: Not scaling conc for spillover correction')
    conc100 = conc_orig.copy()

    is_flagged = conc_orig > 250
    conc100[is_flagged] = conc_orig[is_flagged]

    # converting type allows this to go negative without rolling over
    newice250 = conc_orig.copy().astype(np.int16)

    is_at_coast = shoremap == 1
    is_near_coast = shoremap == 2
    is_far_coastal = shoremap == 3

    # Note: the minic array has to be rescaled from the original 0-100 to 0-250
    # this means that thresholds of 20, 40, 60 are now: 50, 100, 150
    #mod_minic = np.multiply(minic, 2.5).astype(np.uint8)
    mod_minic = minic.copy()

    # mod_minic[is_at_coast & (minic > 60)] = 60
    # mod_minic[is_near_coast & (minic > 40)] = 40
    # mod_minic[is_far_coastal & (minic > 20)] = 20

    #mod_minic[is_at_coast & (minic > 150)] = 150
    #mod_minic[is_near_coast & (minic > 100)] = 100
    #mod_minic[is_far_coastal & (minic > 50)] = 50

    # Count number of nearby low ice conc
    n_low = np.zeros_like(conc_orig, dtype=np.uint8)

    # TODO: The original NASA code allows low-conc values over land to
    #       count toward the number of nearby low-conc values needed to
    #       cause a grid cell to be considered spillover.  This seems like
    #       an error. (https://github.com/nsidc/pm_icecon/issues/63)
    # Note: This scheme does not work if the land values have
    #       no concentration value

    conc_equiv = conc_orig.copy()
    # If the mean concentration value over the land is low, then
    # then it's probably been set to zero and should not be used
    # to determine whether spillover should be applied

    # if match_CDRv4_cdralgos:
    #     # The cdralgos version of the NT land spillover algorithm excludes
    #     # grid cells near the edge of the grid by looping over range (3 to dim-3)
    #     grid_edge = np.zeros(conc_equiv.shape, dtype=np.uint8)

    #     grid_edge[:3, :] = 1
    #     grid_edge[-3:, :] = 1
    #     grid_edge[:, :3] = 1
    #     grid_edge[:, -3:] = 1
    #     is_not_grid_edge = grid_edge == 0

    # Note: the sea ice extent threshold needs to be scaled to 0-250, hence the *2.5
    #sie_threshold = 15 * 2.5
    sie_threshold = 15

    for joff in range(-3, 3 + 1):
        for ioff in range(-3, 3 + 1):
            offmax = max(abs(ioff), abs(joff))

            rolled = np.roll(conc_equiv, (joff, ioff), axis=(0, 1))

            # if match_CDRv4_cdralgos:
            #     is_rolled_low = (rolled < 15) & (rolled >= 0) & is_not_grid_edge
            # else:
            #     is_rolled_low = (rolled < 15) & (rolled >= 0)
            is_rolled_low = (rolled < sie_threshold) & (rolled >= 0)

            if offmax <= 3:
                n_low[is_rolled_low & is_at_coast] += 1

            if offmax <= 2:
                n_low[is_rolled_low & is_near_coast] += 1

            if offmax <= 1:
                n_low[is_rolled_low & is_far_coastal] += 1

    # Note: there are meaningless differences "at the edge" in these counts
    # because the spatial interpolation is not identical along the border

    # Note: original shoremap values were 3, 4, 5
    #       here, they are 1, 2, 3
    # where_reduce_ice = (n_low >= 3) & (shoremap > 2)

    is_not_farocean = ((shoremap >= 1) & (shoremap <= 3))

    where_reduce_ice = (n_low >= 3) & is_not_farocean
    newice250[where_reduce_ice] -= mod_minic[where_reduce_ice]

    where_ice_overreduced = (conc_orig >= 0) & (newice250 < 0) & is_not_farocean
    newice250[where_ice_overreduced] = 0

    # # Preserve missing data (conc value of -10)
    # where_missing = (conc < 0) & where_reduce_ice & is_not_farocean
    # newice[where_missing] = conc[where_missing]

    # # This rounds the floating point values for easier comparison to CDR v4 vals
    # # which are scaled short ints
    # newice = np.round(newice, 2)
    newice = newice250.astype(np.uint8)

    return newice


def load_minimum_iceconc_pmalgos(gridid):
    """Read the pmalgos CMIN file"""
    if gridid == 'psn25':
        fn = './cmin_goddard_1992.N11.byte'
        minic = np.fromfile(fn, dtype=np.uint8).reshape(448, 304)
        return minic
    elif gridid == 'pss25':
        fn = './cmin_goddard_1992.S11.byte'
        minic = np.fromfile(fn, dtype=np.uint8).reshape(332, 316)
        return minic

    print('No minic file found for gridid: {gridid}')
    return None


def load_minimum_iceconc_cdralgos(gridid):
    """Read the pmalgos CMIN file
    This version uses the files from the cdralgos repository
    which were 2-byte integers scaled from 0-1000, so
    need to be divided by 4 for use with 0-250 conc field
    Also, the file has a 300 byte header, which is 150 shorts..."""
    if gridid == 'psn25':
        fn_byte = './concmin-north-byte.dat'
        if os.path.isfile(fn_byte):
            minic_byte = np.fromfile(fn_byte, dtype=np.uint8).reshape(448, 304)
            return minic_byte
        else:
            fn = './concmin-north-le.dat'
            minic = np.fromfile(fn, dtype=np.int16)[150:].reshape(448, 304)
            # Save a byte version
            minic_div4 = np.divide(minic.astype(np.float32), 4.0) + 0.5
            minic_byte = minic_div4.astype(np.uint8)
            minic_byte.tofile(fn_byte)
            return minic_byte
    elif gridid == 'pss25':
        #fn = './concmin-south-le.dat'
        #minic = np.fromfile(fn, dtype=np.int16)[150:].reshape(332, 316)
        fn_byte = './concmin-south-byte.dat'
        if os.path.isfile(fn_byte):
            minic_byte = np.fromfile(fn_byte, dtype=np.uint8).reshape(332, 316)
            return minic_byte
        else:
            fn = './concmin-south-le.dat'
            minic = np.fromfile(fn, dtype=np.int16)[150:].reshape(332, 316)
            # Save a byte version
            minic_div4 = np.divide(minic.astype(np.float32), 4.0) + 0.5
            minic_byte = minic_div4.astype(np.uint8)
            minic_byte.tofile(fn_byte)
            return minic_byte
    else:
        print('No minic file found for gridid: {gridid}')
        return None

    print('should not be able to reach this...')
    breakpoint()
    # minic_rescaled = np.divide(minic, 4)
    # #print('check minic_rescaled...')
    # #breakpoint()
    # minic250 = minic_rescaled.astype(np.uint8)

    # return minic250

def load_shoremap(gridid):
    """Read the pmalgos CMIN file"""
    if gridid == 'psn25':
        fn = 'shoremap_psn25.dat'
        shoremap = np.fromfile(fn, dtype=np.uint8).reshape(448, 304)
        return shoremap
    elif gridid == 'pss25':
        fn = 'shoremap_pss25.dat'
        shoremap = np.fromfile(fn, dtype=np.uint8).reshape(332, 316)
        return shoremap

    print('No shoremap file found for gridid: {gridid}')
    return None


def load_validice_cdralgos(gridid, date):
    """Load the monthly valid-ice masks for this date
    Note: these are landmask dependent!
    """
    month = date.strftime('%m')
    xdim, ydim = get_dims(gridid)
    hemi = gridid_to_hemi[gridid]

    fn_validice = f'./cdralgos_validice/validice-{hemi}-{month}-byte.dat'
    validice_mask = np.fromfile(fn_validice, dtype=np.uint8).reshape(ydim, xdim)

    return validice_mask


def load_nt_had_ice(gridid, date):
    """Load the fields which are non-zero wherever NT had ice in that month
    Note: these are landmask dependent!
    """
    month = date.strftime('%m')
    xdim, ydim = get_dims(gridid)
    hemi = gridid_to_hemi[gridid]

    fn_validice = f'./nt_had_ice/nt_had_ice_{gridid}_{date.strftime("%m")}.dat'
    validice_mask = np.fromfile(fn_validice, dtype=np.uint8).reshape(ydim, xdim)

    return validice_mask


def calc_nt_siconc(gridid, date, source):
    """Calculate the NASA Team siconc"""
    landmask = get_landmask(gridid)
    # nearsie = get_nearsie(gridid, date)
    # climate_mask = get_climate_mask(gridid, date)
    climate_mask = load_nt_had_ice(gridid, date)

    ymd = date.strftime('%Y%m%d')
    tbs = get_tbs(gridid, date, source)

    is_invalid = compute_invalid(tbs)
    is_suspect_tb = tbs['v19'] < tbs['h19']
    is_weather_v37, gr1 = calc_nt_weather(gridid, source, 'v37', tbs)
    is_weather_v22, _ = calc_nt_weather(gridid, source, 'v22', tbs)

    coef = compute_nt_coefficients(gridid, source)

    polarization_ratio = calc_polarization_ratio(tbs)
    prgr = polarization_ratio * gr1
    anf = coef[1] + coef[2] * polarization_ratio + coef[3] * gr1 + coef[4] * prgr
    anm = coef[9] + coef[10] * polarization_ratio + coef[11] * gr1 + coef[12] * prgr
    dd = coef[5] + coef[6] * polarization_ratio + coef[7] * gr1 + coef[8] * prgr

    # print('Applying nearsie mask...')
    # anf[nearsie == 0] = 0
    # anm[nearsie == 0] = 0

    fyr = rescale(anf/dd, c100value)
    myr = rescale(anm/dd, c100value)

    tot = fyr + myr

    totout = force_rov(tot, minconc, c100value)
    myrout = force_rov(myr, minconc, c100value)
    myrout[myrout > totout] = totout[myrout > totout]

    # Apply the various filters
    totout[is_invalid] = 0
    totout[is_suspect_tb] = 0
    totout[is_weather_v37] = 0
    totout[is_weather_v22] = 0
    totout[climate_mask == 0] = 0

    myrout[is_invalid] = 0
    myrout[is_suspect_tb] = 0
    myrout[is_weather_v37] = 0
    myrout[is_weather_v22] = 0
    totout[climate_mask == 0] = 0

    #print('Applying landmask...')

    is_flagged = landmask > 250
    totout[is_flagged] = landmask[is_flagged]
    myrout[is_flagged] = landmask[is_flagged]

    # Apply the land spillover
    #print('applying NT land spillover')
    minic = load_minimum_iceconc_cdralgos(gridid)
    shoremap = load_shoremap(gridid)
    corrected = apply_nt_spillover(totout, shoremap, minic)

    #print('Applying landmask...')
    #is_flagged = landmask > 250
    #totout[is_flagged] = landmask[is_flagged]
    #myrout[is_flagged] = landmask[is_flagged]
    corrected[is_flagged] = landmask[is_flagged]

    return totout, myrout, corrected


def load_nsidc0803(gridid, date, raw=True):
    """Get the nsidc0803 siconc field for this grid and date"""
    srcdir_ = '/home/scotts/Desktop/data/nsidc0803/{gridid}/{year}'
    bfn_ = 'NSIDC-0803_SEAICE_AMSR2_{H}_{ymd}_{ver}.nc'

    year = date.strftime('%Y')
    ymd = date.strftime('%Y%m%d')
    ver = 'v2.0'

    srcdir = srcdir_.format(
        gridid=gridid,
        year=year,
    )
    bfn = bfn_.format(
        H=gridid_to_H[gridid],
        ymd=ymd,
        ver=ver,
    )
    ifn = f'{srcdir}/{bfn}'
    ds = Dataset(ifn)
    if raw:
        ds.set_auto_maskandscale(False)
    siconc = np.squeeze(np.asarray(ds.variables['ICECON']))

    return siconc

def load_area(gridid):
    """Load the area values for this grid"""
    ngrid = NSIDC0001_LOWRES_GRIDS[gridid]
    fn_area = f'/home/scotts/Desktop/data/nsidc0771/NSIDC0771_CellArea_PS_{ngrid}_v1.1.nc'
    ds = Dataset(fn_area)
    area = np.asarray(ds.variables['cell_area']).astype(np.float32) / 1000000 / 1000000

    return area


def load_nsidc0051(gridid, date, raw=True):
    """Get the nsidc0051 siconc field for this grid and date
    ~/Desktop/data/nsidc0051/2024/|head
        NSIDC0051_SEAICE_PS_N25km_20240101_v2.0.nc
    """
    srcdir_ = '/home/scotts/Desktop/data/nsidc0051/{year}'
    bfn_ = 'NSIDC0051_SEAICE_PS_{H}25km_{ymd}_{ver}.nc'

    year = date.strftime('%Y')
    ymd = date.strftime('%Y%m%d')
    ver = 'v2.0'
    ncvar = 'F17_ICECON'

    srcdir = srcdir_.format(
        gridid=gridid,
        year=year,
    )
    bfn = bfn_.format(
        H=gridid_to_H[gridid],
        ymd=ymd,
        ver=ver,
    )
    ifn = f'{srcdir}/{bfn}'
    ds = Dataset(ifn)
    if raw:
        ds.set_auto_maskandscale(False)
    siconc = np.squeeze(np.asarray(ds.variables[ncvar]))

    return siconc


def output_nasateam_values(gridid, date, area, outdir='./output'):
    """Run info for this gridid and date"""

    # Manually choose the sources
    # source = sys.argv[3]

    os.makedirs(outdir, exist_ok=True)
    ymd = date.strftime('%Y%m%d')

    xdim, ydim = get_dims(gridid)

    fn = f'{outdir}/nt_0803_{gridid}_{ymd}.dat'
    if os.path.isfile(fn):
        nt0803 = np.fromfile(fn, dtype=np.uint8).reshape(ydim, xdim)
    else:
        nt0803 = load_nsidc0803(gridid, date)
        breakpoint()
        nt0803.tofile(fn)

    fn = f'{outdir}/nt_0051_{gridid}_{ymd}.dat'
    if os.path.isfile(fn):
        nt0051 = np.fromfile(fn, dtype=np.uint8).reshape(ydim, xdim)
    else:
        nt0051 = load_nsidc0051(gridid, date)
        breakpoint()
        nt0051.tofile(fn)

    # totout, myrout, corrected = calc_nt_siconc(gridid, date, source)
    fn = f'{outdir}/nt_0001_{gridid}_{ymd}.dat'
    if os.path.isfile(fn):
        nt0001 = np.fromfile(fn, dtype=np.uint8).reshape(ydim, xdim)
    else:
        totout, myrout, corrected = calc_nt_siconc(gridid, date, 'nsidc0001_f17')
        nt0001 = corrected
        breakpoint()
        nt0001.tofile(fn)

    fn = f'{outdir}/nt_sfmlr_{gridid}_{ymd}.dat'
    if os.path.isfile(fn):
        sfmlr = np.fromfile(fn, dtype=np.uint8).reshape(ydim, xdim)
    else:
        totout_sfmlr, myrout_sfmlr, corrected_sfmlr = calc_nt_siconc(gridid, date, 'sfmlr')
        sfmlr = corrected_sfmlr
        breakpoint()
        sfmlr.tofile(fn)

    if gridid == 'psn25':
        polehole = np.fromfile('polehole_mask_F17.dat', dtype=np.uint8).reshape(ydim, xdim)
    else:
        polehole = np.zeros((ydim, xdim), dtype=np.uint8)
    is_polehole = polehole > 0

    # print('skipping comparison-mask by nearsie')

    # nearsie = get_nearsie(gridid, date)
    # nt0051[nearsie == 0] = 0
    # nt0803[nearsie == 0] = 0
    # sfmlr[nearsie == 0] = 0

    #concdff_sfmlr = sfmlr.astype(np.int16) - nt0051.astype(np.int16)
    #concdff_nt0803 = nt0803.astype(np.int16) - nt0051.astype(np.int16)

    # Set pole hole to 100% for sea ice extent comparisons
    nt0051[is_polehole] = 250
    nt0803[is_polehole] = 250
    nt0001[is_polehole] = 250
    sfmlr[is_polehole] = 250

    is_sie_nt0051 = (nt0051 >= 38) & (nt0051 <= 250)
    is_sie_nt0803 = (nt0803 >= 38) & (nt0803 <= 250)
    is_sie_nt0001 = (nt0001 >= 38) & (nt0001 <= 250)
    is_sie_sfmlr = (sfmlr >= 38) & (sfmlr <= 250)

    # Set pole hole to 0% for sea ice area comparisons
    nt0051[is_polehole] = 0
    nt0803[is_polehole] = 0
    nt0001[is_polehole] = 0
    sfmlr[is_polehole] = 0

    is_sic_nt0051 = (nt0051 >= 0) & (nt0051 <= 250)
    is_sic_nt0803 = (nt0803 >= 0) & (nt0803 <= 250)
    is_sic_nt0001 = (nt0001 >= 0) & (nt0001 <= 250)
    is_sic_sfmlr = (sfmlr >= 0) & (sfmlr <= 250)

    sifrac_nt0051 = np.zeros((ydim, xdim), dtype=np.float32)
    sifrac_nt0051[is_sic_nt0051] = 0.004 * nt0051[is_sic_nt0051] 

    sifrac_nt0803 = np.zeros((ydim, xdim), dtype=np.float32)
    sifrac_nt0803[is_sic_nt0803] = 0.004 * nt0803[is_sic_nt0803] 

    sifrac_nt0001 = np.zeros((ydim, xdim), dtype=np.float32)
    sifrac_nt0001[is_sic_nt0001] = 0.004 * nt0001[is_sic_nt0001] 

    sifrac_sfmlr = np.zeros((ydim, xdim), dtype=np.float32)
    sifrac_sfmlr[is_sic_sfmlr] = 0.004 * sfmlr[is_sic_sfmlr] 

    total_sie_nt0051 = np.sum(area[is_sie_nt0051])
    total_sie_nt0803 = np.sum(area[is_sie_nt0803])
    total_sie_nt0001 = np.sum(area[is_sie_nt0001])
    total_sie_sfmlr = np.sum(area[is_sie_sfmlr])

    total_sia_nt0051 = np.sum(area[is_sic_nt0051] * sifrac_nt0051[is_sic_nt0051])
    total_sia_nt0803 = np.sum(area[is_sic_nt0803] * sifrac_nt0803[is_sic_nt0803])
    total_sia_nt0001 = np.sum(area[is_sic_nt0001] * sifrac_nt0001[is_sic_nt0001])
    total_sia_sfmlr = np.sum(area[is_sic_sfmlr] * sifrac_sfmlr[is_sic_sfmlr])


    #siediff_sfmlr = sie_sfmlr.astype(np.int16) - sie_nt0051.astype(np.int16)
    #siediff_nt0803 = sie_nt0803.astype(np.int16) - sie_nt0051.astype(np.int16)

    outstr = ''
    outstr += f'{gridid}'
    outstr += f',{date.year},{date.month},{date.day}'

    outstr += f',{total_sie_nt0051:.4f}'
    outstr += f',{total_sie_nt0803:.4f}'
    outstr += f',{total_sie_nt0001:.4f}'
    outstr += f',{total_sie_sfmlr:.4f}'

    outstr += f',{total_sia_nt0051:.4f}'
    outstr += f',{total_sia_nt0803:.4f}'
    outstr += f',{total_sia_nt0001:.4f}'
    outstr += f',{total_sia_sfmlr:.4f}'

    print(outstr, flush=True)


if __name__ == '__main__':
    gridid = sys.argv[1]
    date_0 = dateutil.parser.parse(sys.argv[2]).date()
    date_1 = dateutil.parser.parse(sys.argv[3]).date()


    # Note: Initially a mismount on the 0802v3 VM has tbs on:
    #tbdir = '/share/apps/nsidc0803/nsidc0802/dev/scotts/...'
    # TODO: there should be a unified place to set these dir and file names
    outdir = '/share/apps/nsidc0803/nsidc0803/dev/scotts/nt_0803v3'
    date = date_0
    while date <= date_1:
        year = date.strftime('%Y')
        month = date.strftime('%m')
        ymd = date.strftime('%Y%m%d')
        out_subdir = f'{year}/{month}'

        #fn = f'{outdir}/nt_0803v3_{gridid}_{ymd}.dat'
        fn = f'{outdir}/{out_subdir}/nt_0803v3_{gridid}_{ymd}.dat'
        os.makedirs(os.path.dirname(fn), exist_ok=True)

        if os.path.isfile(fn):
            print(f'File exists: {fn}')
        else:
            #totout_sfmlr, myrout_sfmlr, corrected_sfmlr = calc_nt_siconc(gridid, date, 'sfmlr')
            print(f'Computing NT siconc for: {gridid} {date}')
            raw_ntsiconc, myi_ntsiconc, corrected_ntsiconc = calc_nt_siconc(gridid, date, 'cal')

            # Set pole hole to polehole sentinel value for Northern Hemisphere
            if gridid == 'psn25':
                ydim, xdim = corrected_ntsiconc.shape
                polehole = np.fromfile('polehole_mask_F17.dat', dtype=np.uint8).reshape(ydim, xdim)
                is_polehole = polehole > 0
                corrected_ntsiconc[is_polehole] = 251

            corrected_ntsiconc.tofile(fn)
            print(f'Wrote: {fn}')

        date += dt.timedelta(days=1)
