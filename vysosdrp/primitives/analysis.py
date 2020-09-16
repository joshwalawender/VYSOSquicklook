from pathlib import Path
from datetime import datetime, timedelta
import sys
import re
import subprocess
from matplotlib import pyplot as plt
import subprocess

import numpy as np
from astropy.io import fits
from astropy import units as u
from astropy import stats
from astropy.time import Time
import astropy.coordinates as c
from astropy.wcs import WCS
from astropy.table import Table, Column, MaskedColumn
import ccdproc
import photutils

from keckdata import fits_reader, VYSOS20

from keckdrpframework.primitives.base_primitive import BasePrimitive
from keckdrpframework.models.arguments import Arguments

from .utils import pre_condition, post_condition, download_vizier, get_panstarrs, sigma_clipping_line_fit, estimate_f0


##-----------------------------------------------------------------------------
## Primitive: MoonInfo
##-----------------------------------------------------------------------------
class MoonInfo(BasePrimitive):
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

        lat=c.Latitude(self.action.args.kd.get('SITELAT'), unit=u.degree)
        lon=c.Longitude(self.action.args.kd.get('SITELONG'), unit=u.degree)
        height=float(self.action.args.kd.get('ALT-OBS')) * u.meter
        loc = c.EarthLocation(lon, lat, height)
        temperature=float(self.action.args.kd.get('AMBTEMP'))*u.Celsius
        pressure=self.cfg['Telescope'].getfloat('pressure', 700)*u.mbar
        altazframe = c.AltAz(location=loc, obstime=self.action.args.kd.obstime(),
                             temperature=temperature,
                             pressure=pressure)
        moon = c.get_moon(Time(self.action.args.kd.obstime()), location=loc)
        sun = c.get_sun(Time(self.action.args.kd.obstime()))

        moon_alt = ((moon.transform_to(altazframe).alt).to(u.degree)).value
        moon_separation = (moon.separation(self.action.args.header_pointing).to(u.degree)).value\
                    if self.action.args.header_pointing is not None else None

        # Moon illumination formula from Meeus, ÒAstronomical 
        # Algorithms". Formulae 46.1 and 46.2 in the 1991 edition, 
        # using the approximation cos(psi) \approx -cos(i). Error 
        # should be no more than 0.0014 (p. 316). 
        moon_illum = 50*(1 - np.sin(sun.dec.radian)*np.sin(moon.dec.radian)\
                     - np.cos(sun.dec.radian)*np.cos(moon.dec.radian)\
                     * np.cos(sun.ra.radian-moon.ra.radian))

        self.action.args.moon_alt = moon_alt
        self.action.args.moon_separation = moon_separation
        self.action.args.moon_illum = moon_illum

        return self.action.args


##-----------------------------------------------------------------------------
## Primitive: SolveAstrometry
##-----------------------------------------------------------------------------
class SolveAstrometry(BasePrimitive):
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
                  pre_condition(self, 'WCS not already solved',
                                self.action.args.wcs is None),
                 ]
        force_solve = self.cfg['Astrometry'].getboolean('force_solve', False)
        return np.all(checks) if force_solve is False else True

    def _post_condition(self):
        """Check for conditions necessary to verify that the process run correctly"""
        checks = []
        return np.all(checks)

    def _perform(self):
        """
        Returns an Argument() with the parameters that depends on this operation.
        """
        self.log.info(f"Running {self.__class__.__name__} action")

        nx, ny = self.action.args.kd.pixeldata[0].shape
        estimated_pixel_scale = self.cfg['Telescope'].getfloat('pixel_scale', 1)
        search_radius = self.cfg['Astrometry'].getfloat('search_radius', 1)
        solve_field = self.cfg['Astrometry'].get('solve_field', '/usr/local/bin/solve-field')
        wcs_output_file = Path('~/tmp.wcs').expanduser()
        axy_output_file = Path('~/tmp.axy').expanduser()
        solvetimeout = self.cfg['Astrometry'].getint('solve_timeout', 120)
        cmd = [f'{solve_field}', '-p', '-O', '-N', 'none', '-B', 'none',
               '-U', 'none', '-S', 'none', '-M', 'none', '-R', 'none',
               '--axy', f'{axy_output_file}', '-W', f'{wcs_output_file}',
               '-z', '2',
               '-L', f'{0.9*estimated_pixel_scale}',
               '-H', f'{1.1*estimated_pixel_scale}',
               '-u', 'arcsecperpix',
               '-t', f"{self.cfg['Astrometry'].getfloat('tweak_order', 2)}",
               '-3', f'{self.action.args.header_pointing.ra.deg}',
               '-4', f'{self.action.args.header_pointing.dec.deg}',
               '-5', f'{search_radius}',
               '-l', f'{solvetimeout}',
               ]
        if self.cfg['Astrometry'].get('astrometry_cfg_file', None) is not None:
            cmd.extend(['-b', self.cfg['Astrometry'].get('astrometry_cfg_file')])

        if self.action.args.fitsfilepath.suffix == '.fz':
            self.log.info(f'  Uncompressing {self.action.args.fitsfilepath.name}')
            tmpfile = Path('~/tmp.fts').expanduser().absolute()
            if tmpfile.exists():
                self.log.debug(f'  Removing old {tmpfile}')
                tmpfile.unlink()
            funpackcmd = ['funpack', '-O', f'{tmpfile}',
                          f'{self.action.args.fitsfilepath}']
            self.log.debug(f'  funpack command: {funpackcmd}')
            funpack_result = subprocess.run(funpackcmd,
                                            stdout=subprocess.PIPE,
                                            stderr=subprocess.PIPE)
            cmd.append(f'{tmpfile}')
        else:
            cmd.append(f'{self.action.args.fitsfilepath}')

        self.log.debug(f"  Solve astrometry command: {' '.join(cmd)}")

        try:
            result = subprocess.run(cmd, timeout=solvetimeout,
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE)
        except subprocess.TimeoutExpired as e:
            self.log.warning('The solve-field process timed out')
            return self.action.args

        self.log.debug(f"  Returncode = {result.returncode}")
        for line in result.stdout.decode().split('\n'):
            self.log.debug(line)
        if result.returncode != 0:
            self.log.warning(f'Astrometry solve failed:')
            for line in result.stdout.decode().split('\n'):
                self.log.warning(line)
            for line in result.stderr.decode().split('\n'):
                self.log.warning(line)
            return self.action.args

        if wcs_output_file.exists() is True:
            self.log.debug(f"  Found {wcs_output_file}")
        else:
            raise FileNotFoundError(f"Could not find {wcs_output_file}")
        if axy_output_file.exists() is True:
            self.log.debug(f"  Found {axy_output_file}. Deleteing.")
            axy_output_file.unlink()
        # Open wcs output
        self.log.debug(f"  Creating astropy.wcs.WCS object")
        output_wcs = WCS(f'{wcs_output_file}')
        self.log.debug(f"Deleteing {wcs_output_file}.")
        wcs_output_file.unlink()
        if output_wcs.is_celestial is False:
            self.log.info(f"  Could not parse resulting WCS as celestial")
            return self.action.args

        self.log.info(f"  Solve complete")
        self.action.args.wcs_header = output_wcs.to_header_string()
        self.action.args.wcs = output_wcs

        # Determine Pointing Error
        r, d = self.action.args.wcs.all_pix2world([nx/2.], [ny/2.], 1)
        self.action.args.wcs_pointing = c.SkyCoord(r[0], d[0], frame='fk5',
                                          equinox='J2000',
                                          unit=(u.deg, u.deg),
                                          obstime=self.action.args.kd.obstime())
        self.action.args.perr = self.action.args.wcs_pointing.separation(
                                     self.action.args.header_pointing)
        self.log.info(f'Pointing error = {self.action.args.perr.to(u.arcmin):.1f}')

        return self.action.args


##-----------------------------------------------------------------------------
## Primitive: SolveAstrometryOnline
##-----------------------------------------------------------------------------
class SolveAstrometryOnline(BasePrimitive):
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
                  pre_condition(self, 'WCS not already solved',
                                self.action.args.wcs is None),
                 ]
        force_solve = self.cfg['Astrometry'].getboolean('force_solve', False)
        return np.all(checks) if force_solve is False else True

    def _post_condition(self):
        """Check for conditions necessary to verify that the process run correctly"""
        checks = []
        return np.all(checks)

    def _perform(self):
        """
        Returns an Argument() with the parameters that depends on this operation.
        """
        self.log.info(f"Running {self.__class__.__name__} action")
        from astroquery.exceptions import TimeoutError as astrometryTimeout
        from astroquery.astrometry_net import AstrometryNet

        ast = AstrometryNet()
        ast.api_key = self.cfg['Astrometry'].get('api_key', None)
        solve_timeout = self.cfg['Astrometry'].getint('solve_timeout', 90)
        nx, ny = self.action.args.kd.pixeldata[0].data.shape

        if self.action.args.n_objects >= 100:
            stars = self.action.args.objects[:100]
        else:
            stars = self.action.args.objects

        try:
            self.log.debug(f"  Running astrometry.net solve")
            wcs_header = ast.solve_from_source_list(stars['x'], stars['y'],
                                                    nx, ny,
                                                    solve_timeout=solve_timeout,
                                                    center_dec=self.action.args.header_pointing.dec.deg,
                                                    center_ra=self.action.args.header_pointing.ra.deg,
                                                    radius=0.8,
                                                    scale_est=0.44,
                                                    scale_err=0.02,
                                                    scale_units='arcsecperpix',
                                                    tweak_order=2,
                                                    )
        except astrometryTimeout as e:
            self.log.warning('Astrometry solve timed out')
            return self.action.args
        except Exception as e:
            self.log.warning('Astrometry solve failed')
            self.log.warning(e)
            return self.action.args

        if wcs_header == {}:
            self.log.info(f"  Solve failed")
            return self.action.args

        self.log.info(f"  Solve complete")
        self.action.args.wcs_header = wcs_header
        self.action.args.wcs = WCS(wcs_header)

        # Determine Pointing Error
#         pixel_scale = np.mean(proj_plane_pixel_scales(self.action.args.wcs))*60*60
        r, d = self.action.args.wcs.all_pix2world([nx/2.], [ny/2.], 1)
        self.action.args.wcs_pointing = c.SkyCoord(r[0], d[0], frame='fk5',
                                          equinox='J2000',
                                          unit=(u.deg, u.deg),
                                          obstime=self.action.args.kd.obstime())
        self.action.args.perr = self.action.args.wcs_pointing.separation(
                                     self.action.args.header_pointing)
        self.log.info(f'Pointing error = {self.action.args.perr.to(u.arcmin):.1f}')

        return self.action.args


##-----------------------------------------------------------------------------
## Primitive: GetCalibrationStars
##-----------------------------------------------------------------------------
class GetCalibrationStars(BasePrimitive):
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
        catalog = self.cfg['Photometry'].get('calibration_catalog', '')
#         known_catalogs = ['Gaia', 'UCAC4', 'PanSTARRS']
        known_catalogs = ['PanSTARRS']
        checks = [pre_condition(self, 'Skip image is not set',
                                not self.action.args.skip),
                  pre_condition(self, 'Found existing WCS',
                                self.action.args.wcs is not None),
                  pre_condition(self, f'Catalog {catalog} is known',
                                catalog in known_catalogs),
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

        try:
            pscat = get_panstarrs(self.cfg,
                                  self.action.args.kd.get('OBJECT'),
                                  self.action.args.header_pointing.to_string('decimal'),
                                  self.action.args.kd.filter(),
                                  maglimit=self.cfg['Photometry'].getfloat('calibration_maglimit', 25),
                                  log=self.log,
                                  )
        except Exception as e:
            self.log.debug(e)
            pscat = None

        self.action.args.calibration_catalog = pscat
        ncat = len(self.action.args.calibration_catalog) if self.action.args.calibration_catalog is not None else 0
        self.log.info(f"  Found {ncat} catalog entries for calibration")

        return self.action.args


##-----------------------------------------------------------------------------
## Primitive: CalibratePhotometry
##-----------------------------------------------------------------------------
# class CalibratePhotometry(BasePrimitive):
#     """
#     This is a template for primitives, which is usually an action.
# 
#     The methods in the base class can be overloaded:
#     - _pre_condition
#     - _post_condition
#     - _perform
#     - apply
#     - __call__
#     """
# 
#     def __init__(self, action, context):
#         BasePrimitive.__init__(self, action, context)
#         self.log = context.pipeline_logger
#         self.cfg = self.context.config.instrument
# 
#     def _pre_condition(self):
#         """Check for conditions necessary to run this process"""
#         checks = [pre_condition(self, 'Skip image is not set',
#                                 not self.action.args.skip),
#                   pre_condition(self, 'Calibration catalog exists',
#                                 self.action.args.calibration_catalog is not None),
#                   pre_condition(self, 'Calibration catalog has been associated',
#                                 self.action.args.associated_calibrators is not None),
#                  ]
#         return np.all(checks)
# 
#     def _post_condition(self):
#         """Check for conditions necessary to verify that the process run correctly"""
#         checks = []
#         return np.all(checks)
# 
#     def _perform(self):
#         """
#         Returns an Argument() with the parameters that depends on this operation.
#         """
#         self.log.info(f"Running {self.__class__.__name__} action")
#         nx, ny = self.action.args.kd.pixeldata[0].shape



##-----------------------------------------------------------------------------
## Primitive: CalibratePhotometry
##-----------------------------------------------------------------------------
class CalibratePhotometry(BasePrimitive):
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
                  pre_condition(self, 'Calibration catalog exists',
                                self.action.args.calibration_catalog is not None),
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
        nx, ny = self.action.args.kd.pixeldata[0].shape

        ## Do photometry measurement
        self.log.debug(f'  Add pixel positions to catalog')
        x, y = self.action.args.wcs.all_world2pix(self.action.args.calibration_catalog['raMean'],
                                                  self.action.args.calibration_catalog['decMean'], 1)
        self.action.args.calibration_catalog.add_column(Column(data=x, name='x'))
        self.action.args.calibration_catalog.add_column(Column(data=y, name='y'))

        buffer = 10
        in_image = (self.action.args.calibration_catalog['x'] > buffer)\
                   & (self.action.args.calibration_catalog['x'] < nx-buffer)\
                   & (self.action.args.calibration_catalog['y'] > buffer)\
                   & (self.action.args.calibration_catalog['y'] < ny-buffer)
        self.log.debug(f'  Only {np.sum(in_image)} stars are within image pixel boundaries')
        self.action.args.calibration_catalog = self.action.args.calibration_catalog[in_image]

        positions = [p for p in zip(self.action.args.calibration_catalog['x'],
                                    self.action.args.calibration_catalog['y'])]

        self.log.debug(f'  Attemping shape measurement for {len(positions)} stars')
        fwhm = list()
        elliptiticty = list()
        orientation = list()
        xcentroid = list()
        ycentroid = list()
        for i, entry in enumerate(self.action.args.calibration_catalog):
            xi = int(x[i])
            yi = int(y[i])
            if xi > 10 and xi < nx-10 and yi > 10 and yi < ny-10:
                im = self.action.args.kd.pixeldata[0].data[yi-10:yi+10,xi-10:xi+10]
                properties = photutils.data_properties(im)
                fwhm.append(2.355*(properties.semimajor_axis_sigma.value**2\
                            + properties.semiminor_axis_sigma.value**2)**0.5)
                orientation.append(properties.orientation.to(u.deg).value)
                elliptiticty.append(properties.elongation)
                xcentroid.append(properties.xcentroid.value)
                ycentroid.append(properties.ycentroid.value)
            else:
                fwhm.append(np.nan)
                orientation.append(np.nan)
                elliptiticty.append(np.nan)
                xcentroid.append(np.nan)
                ycentroid.append(np.nan)
        wnan = np.isnan(fwhm)
        nmasked = np.sum(wnan)
        self.log.info(f"  Measured {len(fwhm)-nmasked} indivisual FWHM values")
        self.action.args.calibration_catalog.add_column(MaskedColumn(
                data=fwhm, name='FWHM', mask=wnan))
        self.action.args.calibration_catalog.add_column(MaskedColumn(
                data=orientation, name='orientation', mask=wnan))
        self.action.args.calibration_catalog.add_column(MaskedColumn(
                data=elliptiticty, name='elliptiticty', mask=wnan))
        self.action.args.calibration_catalog.add_column(MaskedColumn(
                data=xcentroid, name='xcentroid', mask=wnan))
        self.action.args.calibration_catalog.add_column(MaskedColumn(
                data=ycentroid, name='ycentroid', mask=wnan))


        self.log.debug(f'  Attemping aperture photometry for {len(positions)} stars')
        FWHM_pix = self.action.args.fwhm if self.action.args.fwhm is not None else 8
        ap_radius = int(2.0*FWHM_pix)
        star_apertures = photutils.CircularAperture(positions, ap_radius)
        sky_apertures = photutils.CircularAnnulus(positions,
                                                  r_in=int(np.ceil(1.5*ap_radius)),
                                                  r_out=int(np.ceil(2.0*ap_radius)))
        self.log.debug(f'  Running photutils.aperture_photometry')
        phot_table = photutils.aperture_photometry(
                               self.action.args.kd.pixeldata[0].data,
                               [star_apertures, sky_apertures])

        self.log.debug(f'  Subtracting sky flux')
        phot_table['sky'] = phot_table['aperture_sum_1'] / sky_apertures.area()
        med_sky = np.nanmedian(phot_table['sky'])
        self.log.info(f'  Median Sky = {med_sky:.0f} e-/pix')
        self.action.args.sky_background = med_sky
        self.action.args.calibration_catalog.add_column(phot_table['sky'])
        bkg_sum = phot_table['aperture_sum_1'] / sky_apertures.area() * star_apertures.area()
        final_sum = (phot_table['aperture_sum_0'] - bkg_sum)/self.action.args.kd.exptime()
        phot_table['flux'] = final_sum
        self.action.args.calibration_catalog.add_column(phot_table['flux'])
        wzero = (self.action.args.calibration_catalog['flux'] < 0)
        self.log.debug(f'  Masking {np.sum(wzero)} stars with <0 flux')
        self.action.args.calibration_catalog['flux'].mask = wzero


        # Estimate flux from catalog magnitude
        self.log.debug(f'  Estimate flux from catalog magnitude')
        d_telescope = self.cfg['Telescope'].getfloat('d_primary_mm', 508)
        d_obstruction = self.cfg['Telescope'].getfloat('d_obstruction_mm', 127)
        A = 3.14*(d_telescope/2/1000)**2 - 3.14*(d_obstruction/2/1000)**2 # m^2
        self.action.args.f0 = estimate_f0(A, band=self.action.args.band) # photons / sec
        catflux = self.action.args.f0\
                  * 10**(-self.action.args.calibration_catalog[f'{self.action.args.band}MeanApMag']/2.5)
        self.action.args.calibration_catalog.add_column(Column(data=catflux*u.photon/u.second,
                                                               name='catflux'))

        self.log.debug(f'  Fit, clipping brightest 5% and faintest 5% of stars')
        bad = (self.action.args.calibration_catalog['flux'].mask == True)
        nclip = int(np.floor(0.05*len(self.action.args.calibration_catalog[~bad])))
        fitted_line = sigma_clipping_line_fit(self.action.args.calibration_catalog[~bad]['catflux'][nclip:-nclip].data,
                                              self.action.args.calibration_catalog[~bad]['flux'][nclip:-nclip].data,
                                              intercept_fixed=True)
        self.log.info(f"  Slope (e-/photon) = {fitted_line.slope.value:.3g}")
        self.action.args.zero_point_fit = fitted_line
        deltas = self.action.args.calibration_catalog['flux'].data\
               - fitted_line(self.action.args.calibration_catalog['catflux'].data)
        mean, med, std = stats.sigma_clipped_stats(deltas)
        self.log.info(f"  Fit StdDev = {std:.2g}")

        return self.action.args


##-----------------------------------------------------------------------------
## Primitive: GetTargetStars
##-----------------------------------------------------------------------------
class GetTargetStars(BasePrimitive):
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
        catalog = self.cfg['Photometry'].get('target_catalog', None)
        known_catalogs = ['Gaia', 'UCAC4']
        checks = [pre_condition(self, 'Skip image is not set',
                                not self.action.args.skip),
                  pre_condition(self, 'Found existing WCS',
                                self.action.args.wcs is not None),
                  pre_condition(self, 'Found existing WCS pointing',
                                self.action.args.wcs_pointing is not None),
                  pre_condition(self, f'Catalog {catalog} is known',
                                catalog in known_catalogs),
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

        pscat = get_panstarrs(self.cfg,
                              self.action.args.kd.get('OBJECT'),
                              self.action.args.wcs_pointing.to_string('decimal'),
                              self.action.args.kd.filter(),
                              maglimit=self.cfg['Photometry'].getfloat('target_maglimit', 17),
                              log=self.log,
                              )

        self.action.args.target_catalog = pscat
        ncat = len(self.action.args.target_catalog) if self.action.args.target_catalog is not None else 0
        self.log.info(f"  Found {ncat} entries in target catalog")

        return self.action.args


##-----------------------------------------------------------------------------
## Primitive: ImageStats
##-----------------------------------------------------------------------------
class ImageStats(BasePrimitive):
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

        self.action.args.image_stats = (stats.sigma_clipped_stats(self.action.args.kd.pixeldata[0]))
        return self.action.args


if __name__ == '__main__':
    pass

