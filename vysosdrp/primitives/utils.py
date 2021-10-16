from pathlib import Path
from datetime import datetime, timedelta
import logging
import sys
import re
import subprocess
from matplotlib import pyplot as plt
from matplotlib.dates import HourLocator, MinuteLocator, DateFormatter
plt.style.use('classic')
import pymongo

import numpy as np
from astropy.io import fits
from astropy import units as u
from astropy import stats
from astropy.time import Time
import astropy.coordinates as c
from astropy.wcs import WCS
from astropy.table import Table, Column
from astropy.modeling import models, fitting
from astropy.wcs.utils import proj_plane_pixel_scales
import ephem
import ccdproc
import photutils
import sep

from keckdata import fits_reader, VYSOS20

from keckdrpframework.primitives.base_primitive import BasePrimitive
from keckdrpframework.models.arguments import Arguments


##-----------------------------------------------------------------------------
## find_master
##-----------------------------------------------------------------------------
def build_master_file_name(kd, master_type, date_string):
    if master_type.lower() in ['bias', 'zero']:
        master_file_name = f"MasterBias_{date_string}.fits"
    elif master_type.lower() in ['dark']:
        exptime = int(kd.exptime())
        master_file_name = f"MasterDark_{exptime:03d}s_{date_string}.fits"
    elif master_type.lower() in ['flat']:
        master_file_name = f"MasterFlat_{kd.filter()}_{date_string}.fits"
    else:
        master_file_name = None
    return master_file_name


def find_master(master_directory, master_type, kd):
    # Find master bias file
    if master_directory is not None:
        master_directory = Path(master_directory)
    else:
        return None
    if master_directory.exists() is False:
        return None

    # Build expected file name
    date_string = kd.obstime().strftime('%Y%m%dUT')
    master_file_name = build_master_file_name(kd, master_type, date_string)
    master_file = master_directory.joinpath(master_file_name)

    # Go hunting for the files
    if master_file.exists() is True:
        return master_file
    else:
        # Look for bias within 10 days
        count = 0
        while master_file.exists() is False and count <= 10:
            count += 1
            # Days before
            date_string = (kd.obstime()-timedelta(count)).strftime('%Y%m%dUT')
            master_file_name = build_master_file_name(kd, master_type, date_string)
            master_file = master_directory.joinpath(master_file_name)
            if master_file.exists() is True:
                return master_file
            # Days after
            date_string = (kd.obstime()+timedelta(count)).strftime('%Y%m%dUT')
            master_file_name = build_master_file_name(kd, master_type, date_string)
            master_file = master_directory.joinpath(master_file_name)
            if master_file.exists() is True:
                return master_file
        if master_file.exists() is False:
            return None
        return master_file


##-----------------------------------------------------------------------------
## Evaluate pre and post conditions
##-----------------------------------------------------------------------------
def pre_condition(primitive, name, condition,
                  fail_level=logging.DEBUG,
                  success_level=logging.DEBUG):
    if condition is True:
        primitive.log.log(success_level,
            f'Precondition for {primitive.__class__.__name__} "{name}" satisfied')
    else:
        primitive.log.log(fail_level,
            f'Precondition for {primitive.__class__.__name__} "{name}" failed')
    return condition


def post_condition(primitive, name, condition,
                   fail_level=logging.WARNING,
                   success_level=logging.DEBUG):
    if condition is True:
        primitive.log.log(success_level,
            f'Postcondition for {primitive.__class__.__name__} "{name}" satisfied')
    else:
        primitive.log.log(fail_level,
            f'Postcondition for {primitive.__class__.__name__} "{name}" failed')
    return condition


##-----------------------------------------------------------------------------
## Function: download_vizier
##-----------------------------------------------------------------------------
def download_vizier(pointing, radius, catalog='UCAC4', band='i', maglimit=None):
    from astroquery.vizier import Vizier
    catalogs = {'UCAC4': 'I/322A', 'Gaia': 'I/345/gaia2'}
    if catalog not in catalogs.keys():
        print(f'{catalog} not in {catalogs.keys()}')
        raise NotImplementedError
    if band not in ['r', 'i']:
        print(f'Band {band} not supported')
        raise NotImplementedError

    columns = {'UCAC4': ['_RAJ2000', '_DEJ2000', 'rmag', 'imag'],
               'Gaia': ['RA_ICRS', 'DE_ICRS', 'Gmag', 'RPmag']}
    ra_colname = {'UCAC4': '_RAJ2000',
                  'Gaia': 'RA_ICRS'}
    dec_colname = {'UCAC4': '_DEJ2000',
                   'Gaia': 'DE_ICRS'}
    mag_colname = {'UCAC4': f'{band}mag',
                   'Gaia': 'RPmag'}
    filter_string = '>0' if maglimit is None else f"<{maglimit}"
    column_filter = {mag_colname[catalog]: filter_string}

    v = Vizier(columns=columns[catalog],
               column_filters=column_filter)
    v.ROW_LIMIT = 2e4

    try:
        stars = Table(v.query_region(pointing, catalog=catalogs[catalog],
                                     radius=c.Angle(radius, "deg"))[0])
        stars.add_column( Column(data=stars[ra_colname[catalog]], name='RA') )
        stars.add_column( Column(data=stars[dec_colname[catalog]], name='DEC') )
        stars.add_column( Column(data=stars[mag_colname[catalog]], name='mag') )
    except:
        stars = None
    return stars


##-----------------------------------------------------------------------------
## Function: get_panstarrs
##-----------------------------------------------------------------------------
def get_panstarrs(cfg, field_name, pointing, filter, maglimit=None, log=None):
    catalogname = cfg['Photometry'].get('calibration_catalog')
    band = {'PSi': 'i', 'PSr': 'r'}[filter]
    if maglimit is None: maglimit = 25

    ## First check if we have a pre-downloaded catalog for this field
    local_catalog_path = Path(cfg['Photometry'].get('local_catalog_path', '.'))
    local_catalog_file = local_catalog_path.joinpath(f'{field_name}_{band}{maglimit*10:03.0f}.cat')
    if local_catalog_file.exists() is True:
        ## Read local file
        if log: log.debug(f'  Reading {local_catalog_file}')
        pscat = Table.read(local_catalog_file, format='ascii.csv')
    else:
        ## Download
        if log: log.debug(f'  Downloading from Mast')
        radius = 0.35 # Allow for some telescope pointing error
        from astroquery.mast import Catalogs
        cols = ['objName', 'objID', 'objInfoFlag', 'qualityFlag', 'raMean',
                'decMean', 'raMeanErr', 'decMeanErr', 'epochMean', 'nDetections',
                'ng', 'nr', 'ni', 'gMeanApMag', 'gMeanApMagErr', 'gMeanApMagStd',
                'gMeanApMagNpt', 'gFlags', 'rMeanApMag', 'rMeanApMagErr',
                'rMeanApMagStd', 'rMeanApMagNpt', 'rFlags', 'iMeanApMag',
                'iMeanApMagErr', 'iMeanApMagStd', 'iMeanApMagNpt', 'iFlags']
        if band == 'i':
            pscat = Catalogs.query_region(pointing, radius=radius,
                             catalog="Panstarrs", table="mean", data_release="dr2",
                             sort_by=[("desc", f"{band}MeanApMag")], columns=cols,
                             iMeanApMag=[("gte", 0), ("lte", maglimit)],
                             )
        elif band == 'r':
            pscat = Catalogs.query_region(pointing, radius=radius,
                             catalog="Panstarrs", table="mean", data_release="dr2",
                             sort_by=[("desc", f"{band}MeanApMag")], columns=cols,
                             rMeanApMag=[("gte", 0), ("lte", maglimit)],
                             )
        elif band == 'g':
            pscat = Catalogs.query_region(pointing, radius=radius,
                             catalog="Panstarrs", table="mean", data_release="dr2",
                             sort_by=[("desc", f"{band}MeanApMag")], columns=cols,
                             gMeanApMag=[("gte", 0), ("lte", maglimit)],
                             )
        else:
            pscat = Catalogs.query_region(pointing, radius=radius,
                             catalog="Panstarrs", table="mean", data_release="dr2",
                             columns=cols,
                             )
        if log: log.debug(f'  Got {len(pscat)} entries total')
        if log: log.debug(f'  Got {len(pscat)} entries with {band}-band magnitudes')
        if log: log.debug(f'  Writing {local_catalog_file}')
        pscat.write(local_catalog_file, format='ascii.csv')

    # Filter based on magnitude
    if maglimit is not None:
        pscat = pscat[pscat[f'{band}MeanApMag'] <= maglimit]

    return pscat


##-----------------------------------------------------------------------------
## Function: sigma_clipping_line_fit
##-----------------------------------------------------------------------------
def sigma_clipping_line_fit(xdata, ydata, nsigma=3, maxiter=7, maxcleanfrac=0.3,
                            intercept_fixed=False, intercept0=0, slope0=1,
                            log=None):
        if log: log.debug('  Running sigma_clipping_line_fit')
        npoints = len(xdata)
        if log: log.debug(f'  npoints = {npoints}')
        fit = fitting.LinearLSQFitter()
        line_init = models.Linear1D(slope=slope0, intercept=intercept0)
        line_init.intercept.fixed = intercept_fixed
        fitted_line = fit(line_init, xdata, ydata)
        deltas = ydata - fitted_line(xdata)
        mean, median, std = stats.sigma_clipped_stats(deltas)
        cleaned = np.array(abs(deltas) < nsigma*std)
#         if log: log.debug(cleaned)
        if log: log.debug(f'  fitted slope = {fitted_line.slope.value:3g}')
        if log: log.debug(f'  std = {std:4g}')
        if log: log.debug(f'  n_cleaned = {np.sum(cleaned)}')
        for iteration in range(1, maxiter+1):
            last_std = std
            new_fit = fit(line_init, xdata[cleaned], ydata[cleaned])
            deltas = ydata - new_fit(xdata)
            mean, median, std = stats.sigma_clipped_stats(deltas)
            cleaned = cleaned | np.array(abs(deltas) < nsigma*std)
            if np.sum(~cleaned)/npoints > maxcleanfrac:
                if log: log.debug(f'  Exceeded maxcleanfrac of {maxcleanfrac}')
                return fitted_line
            if std > last_std:
                if log: log.debug(f'  StdDev increased')
                return fitted_line
            else:
                fitted_line = new_fit
#             if log: log.debug(cleaned)
            if log: log.debug(f'  {iteration} fitted slope = {fitted_line.slope.value:3g}')
            if log: log.debug(f'  {iteration} std = {std:4g}')
            if log: log.debug(f'  {iteration} n_cleaned = {np.sum(cleaned)}')

        return fitted_line


##-----------------------------------------------------------------------------
## Function: estimate_f0
##-----------------------------------------------------------------------------
def estimate_f0(A, band='i'):
    '''
    1 Jy = 1.51e7 photons sec^-1 m^-2 (dlambda/lambda)^-1
    https://archive.is/20121204144725/http://www.astro.utoronto.ca/~patton/astro/mags.html#selection-587.2-587.19
    band cent    dl/l    Flux0   Reference
    U    0.36    0.15    1810    Bessel (1979)
    B    0.44    0.22    4260    Bessel (1979)
    V    0.55    0.16    3640    Bessel (1979)
    R    0.64    0.23    3080    Bessel (1979)
    I    0.79    0.19    2550    Bessel (1979)
    J    1.26    0.16    1600    Campins, Reike, & Lebovsky (1985)
    H    1.60    0.23    1080    Campins, Reike, & Lebovsky (1985)
    K    2.22    0.23    670     Campins, Reike, & Lebovsky (1985)
    g    0.52    0.14    3730    Schneider, Gunn, & Hoessel (1983)
    r    0.67    0.14    4490    Schneider, Gunn, & Hoessel (1983)
    i    0.79    0.16    4760    Schneider, Gunn, & Hoessel (1983)
    z    0.91    0.13    4810    Schneider, Gunn, & Hoessel (1983)
    '''
    tabledata = {'band': ['U', 'B', 'V', 'R', 'I', 'J', 'H', 'K', 'g', 'r', 'i', 'z'],
                 'cent': [0.36, 0.44, 0.55, 0.64, 0.79, 1.26, 1.60, 2.22, 0.52, 0.67, 0.79, 0.91],
                 'dl/l': [0.15, 0.22, 0.16, 0.23, 0.19, 0.16, 0.23, 0.23, 0.14, 0.14, 0.16, 0.13],
                 'Flux0': [1810, 4260, 3640, 3080, 2550, 1600, 1080, 670 , 3730, 4490, 4760, 4810],
                }
    t = Table(tabledata)
    band = t[t['band'] == band]
    dl = 0.16 # dl/l (for i band)
    f0 = band['Flux0'] * 1.51e7 * A * band['dl/l'] # photons / sec
    return f0[0]


##-----------------------------------------------------------------------------
## Primitive: Record
##-----------------------------------------------------------------------------
class Record(BasePrimitive):
    """
    This is a template for primitives, which is usually an action.

    The methods in the base class can be overloaded:
    - _pre_condition
    - _post_condition
    - _perform
    - apply
    - __call__
    """

    def __init__(self, action, context):
        BasePrimitive.__init__(self, action, context)
        self.log = context.pipeline_logger
        self.cfg = self.context.config.instrument

    def _pre_condition(self):
        """Check for conditions necessary to run this process"""
        checks = [pre_condition(self, 'Skip image is not set',
                                not self.action.args.skip),
                  pre_condition(self, 'norecord configuration is not set',
                                not self.cfg['Telescope'].getboolean('norecord', False)),
                  pre_condition(self, 'connected to mongoDB',
                                self.action.args.images is not None),
                 ]
        return np.all(checks)

    def _post_condition(self):
        """Check for conditions necessary to verify that the process run correctly"""
        checks = []
        return np.all(checks)

    def _perform(self):
        """
        Returns an Argument() with the parameters that depends on this operation.
        """
        self.log.info(f"Running {self.__class__.__name__} action")

        # Comple image info to store
        self.image_info = {
            'filename': self.action.args.fitsfile,
            'telescope': self.action.args.kd.instrument,
            'compressed': Path(self.action.args.kd.fitsfilename).suffix == '.fz',
            }
        self.log.info(self.image_info)
        # From Header
        if self.action.args.kd.get('OBJECT', None) is not None:
            self.image_info['target name'] = self.action.args.kd.get('OBJECT')
        if self.action.args.kd.get('EXPTIME', None) is not None:
            self.image_info['exptime'] = self.action.args.kd.get('EXPTIME')
        if self.action.args.kd.obstime() is not None:
            self.image_info['date'] = self.action.args.kd.obstime()
        if self.action.args.kd.get('FILTER', None) is not None:
            self.image_info['filter'] = self.action.args.kd.get('FILTER')
        if self.action.args.kd.get('AZIMUTH', None) is not None:
            self.image_info['az'] = float(self.action.args.kd.get('AZIMUTH'))
        if self.action.args.kd.get('ALTITUDE', None) is not None:
            self.image_info['alt'] = float(self.action.args.kd.get('ALTITUDE'))
        if self.action.args.kd.get('AIRMASS', None) is not None:
            self.image_info['airmass'] = float(self.action.args.kd.get('AIRMASS'))
        # From Science Image Analysis
        if hasattr(self.action.args, 'header_pointing'):
            if self.action.args.header_pointing is not None:
                self.image_info['header_RA'] = self.action.args.header_pointing.ra.deg
                self.image_info['header_DEC'] = self.action.args.header_pointing.dec.deg
        if hasattr(self.action.args, 'moon_alt'):
            if self.action.args.moon_alt is not None:
                self.image_info['moon_alt'] = self.action.args.moon_alt
        if hasattr(self.action.args, 'moon_separation'):
            if self.action.args.moon_separation is not None:
                self.image_info['moon_separation'] = self.action.args.moon_separation
        if hasattr(self.action.args, 'moon_illum'):
            if self.action.args.moon_illum is not None:
                self.image_info['moon_illumination'] = self.action.args.moon_illum
        if hasattr(self.action.args, 'fwhm'):
            if self.action.args.fwhm is not None:
                self.image_info['FWHM_pix'] = self.action.args.fwhm
        if hasattr(self.action.args, 'ellipticity'):
            if self.action.args.ellipticity is not None:
                self.image_info['ellipticity'] = self.action.args.ellipticity
        if hasattr(self.action.args, 'n_objects'):
            if self.action.args.n_objects is not None:
                self.image_info['n_stars'] = self.action.args.n_objects
        if hasattr(self.action.args, 'zero_point'):
            if self.action.args.zero_point is not None:
                self.image_info['zero point'] = self.action.args.zero_point
        if hasattr(self.action.args, 'throughput'):
            if self.action.args.throughput is not None:
                self.image_info['throughput'] = self.action.args.throughput
        if hasattr(self.action.args, 'sky_background'):
            if self.action.args.sky_background is not None:
                self.image_info['sky background'] = self.action.args.sky_background
        if hasattr(self.action.args, 'perr'):
            if self.action.args.perr is not None and not np.isnan(self.action.args.perr):
                self.image_info['perr_arcmin'] = self.action.args.perr.to(u.arcmin).value
        if hasattr(self.action.args, 'wcs'):
            if self.action.args.wcs is not None:
                self.image_info['wcs'] = str(self.action.args.wcs.to_header()).strip()
        if hasattr(self.action.args, 'jpegfile'):
            if self.action.args.jpegfile is not None:
                self.image_info['jpegs'] = [f"{self.action.args.jpegfile.name}"]
        # From Flat Image Analysis
        if hasattr(self.action.args, 'image_stats'):
            if self.action.args.image_stats is not None:
                self.image_info['sky background'] = self.action.args.image_stats[1]

        # Log this info
        for key in self.image_info.keys():
            self.log.debug(f'  {key}: {self.image_info[key]}')

        # Remove old entries for this image file
        deletion = self.action.args.images.delete_many( {'filename': self.action.args.fitsfile} )
        self.log.debug(f'  Deleted {deletion.deleted_count} previous entries for {self.action.args.fitsfile}')

        # Save new entry for this image file
        self.log.debug('  Adding image info to mongo database')
        ## Save document
        try:
            inserted_id = self.action.args.images.insert_one(self.image_info).inserted_id
            self.log.debug(f"  Inserted document id: {inserted_id}")
        except:
            e = sys.exc_info()[0]
            self.log.error('Failed to add new document')
            self.log.error(e)
        self.action.args.mongoclient.close()

        return self.action.args


##-----------------------------------------------------------------------------
## Primitive: UpdateDirectory
##-----------------------------------------------------------------------------
class UpdateDirectory(BasePrimitive):
    """
    This is a template for primitives, which is usually an action.

    The methods in the base class can be overloaded:
    - _pre_condition
    - _post_condition
    - _perform
    - apply
    - __call__
    """

    def __init__(self, action, context):
        BasePrimitive.__init__(self, action, context)
        self.log = context.pipeline_logger

    def _pre_condition(self):
        """Check for conditions necessary to run this process"""
        checks = []
        return np.all(checks)

    def _post_condition(self):
        """Check for conditions necessary to verify that the process run correctly"""
        checks = []
        return np.all(checks)

    def _perform(self):
        """
        Returns an Argument() with the parameters that depends on this operation.
        """
        self.log.info(f"Running {self.__class__.__name__} action")

        newdir = Path(self.action.args.input).expanduser()
        if newdir.exists() is False:
            self.log.info(f"  Creating directory {newdir}")
            newdir.mkdir(parents=True)
        self.log.info(f"  Updating directory to {newdir}")

        if self.context.data_set is not None:
            self.context.data_set.remove_all()
        self.context.data_set.change_directory(f"{newdir}")

        if self.action.args.flats is not True and self.action.args.cals is not True:
            self.log.info(f'  Changing UT date, so clearing lists of flats and cals')
            self.context.biases = list()
            self.context.darks = dict()
            self.context.flats = dict()

        files = [f.name for f in newdir.glob('*.fts')]
        self.log.info(f"  Ingesting {len(files)} files")
        for file in files:
            self.log.debug(f"  Appending {file}")
            self.context.data_set.append_item(file)

        self.context.data_set.start_monitor()

        return self.action.args


##-----------------------------------------------------------------------------
## Primitive: SetFileType
##-----------------------------------------------------------------------------
class SetFileType(BasePrimitive):
    """
    This is a template for primitives, which is usually an action.

    The methods in the base class can be overloaded:
    - _pre_condition
    - _post_condition
    - _perform
    - apply
    - __call__
    """

    def __init__(self, action, context):
        BasePrimitive.__init__(self, action, context)
        self.log = context.pipeline_logger
        self.cfg = self.context.config.instrument

    def _pre_condition(self):
        """Check for conditions necessary to run this process"""
        checks = []
        return np.all(checks)

    def _post_condition(self):
        """Check for conditions necessary to verify that the process run correctly"""
        checks = []
        return np.all(checks)

    def _perform(self):
        """
        Returns an Argument() with the parameters that depends on this operation.
        """
        self.log.info(f"Running {self.__class__.__name__} action")

        self.log.info(f'  Data directory is: {self.action.args.input}')
        if '/Volumes/VYSOSData' in self.action.args.input:
            file_type = "*.fts.fz"
        elif '/Users/vysosuser/V20Data/Images' in self.action.args.input:
            file_type = "*.fts"

        self.log.info(f'  Setting file type to "{file_type}"')
        if self.context.data_set is not None:
            self.context.data_set.file_type = file_type

        return self.action.args



##-----------------------------------------------------------------------------
## Primitive: SetOverwrite
##-----------------------------------------------------------------------------
class SetOverwrite(BasePrimitive):
    """
    This is a template for primitives, which is usually an action.

    The methods in the base class can be overloaded:
    - _pre_condition
    - _post_condition
    - _perform
    - apply
    - __call__
    """

    def __init__(self, action, context):
        BasePrimitive.__init__(self, action, context)
        self.log = context.pipeline_logger
        self.cfg = self.context.config.instrument

    def _pre_condition(self):
        """Check for conditions necessary to run this process"""
        checks = []
        return np.all(checks)

    def _post_condition(self):
        """Check for conditions necessary to verify that the process run correctly"""
        checks = []
        return np.all(checks)

    def _perform(self):
        """
        Returns an Argument() with the parameters that depends on this operation.
        """
        self.log.info(f"Running {self.__class__.__name__} action")
        self.log.info(f"  Setting overwrite to True")
        self.cfg.set('Telescope', 'overwrite', value='True')

        return self.action.args


if __name__ == '__main__':
    pass
