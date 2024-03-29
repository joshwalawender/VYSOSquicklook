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
import sep

from keckdrpframework.primitives.base_primitive import BasePrimitive
from keckdrpframework.models.arguments import Arguments

from .utils import pre_condition, post_condition, sigma_clipping_line_fit, estimate_f0, mode


##-----------------------------------------------------------------------------
## Primitive: MakeSourceMask
##-----------------------------------------------------------------------------
class MakeSourceMask(BasePrimitive):
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
                  pre_condition(self, 'Extraction requested',
                                self.cfg['Extract'].getboolean('do_extraction', False) is True),
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
        snr = 5
        npixels = 5
        self.action.args.source_mask = [None]*len(self.action.args.kd.pixeldata)
        for i,pd in enumerate(self.action.args.kd.pixeldata):
            source_mask = photutils.make_source_mask(pd, snr, npixels)
            self.action.args.source_mask[i] = source_mask

        return self.action.args


##-----------------------------------------------------------------------------
## Primitive: ExtractStars
##-----------------------------------------------------------------------------
class ExtractStars(BasePrimitive):
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
                  pre_condition(self, 'Background has been generated',
                                self.action.args.background is not None),
                  pre_condition(self, 'Extraction requested',
                                self.cfg['Extract'].getboolean('do_extraction', False) is True),
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

        exptime = float(self.action.args.kd.get('EXPTIME'))

        # Photutils StarFinder
#         extract_fwhm = self.cfg['Extract'].getfloat('extract_fwhm', 5)
#         thresh = self.cfg['Extract'].getint('extract_threshold', 9)
#         pixel_scale = self.cfg['Telescope'].getfloat('pixel_scale', 1)
#         pd = self.action.args.kd.pixeldata[0]
#         mean, median, std = stats.sigma_clipped_stats(pd.data, sigma=3.0)
#         # daofind = photutils.DAOStarFinder(fwhm=extract_fwhm, threshold=thresh*std)  
#         starfind = photutils.IRAFStarFinder(thresh*std, extract_fwhm)  
#         sources = starfind(pd.data)
#         self.log.info(f'  Found {len(sources):d} stars')
#         self.log.info(sources.keys())
# 
#         self.action.args.objects = sources
#         self.action.args.objects.sort('flux')
#         self.action.args.objects.reverse()
#         self.action.args.n_objects = len(sources)
#         if self.action.args.n_objects == 0:
#             self.log.warning('No stars found')
#             self.action.args.fwhm = np.nan
#             self.action.args.ellipticity = np.nan
#         else:
#             FWHM_pix = np.median(sources['fwhm'])
#             roundness = np.median(sources['roundness'])
#             self.log.info(f'  Median FWHM = {FWHM_pix:.1f} pix ({FWHM_pix*pixel_scale:.2f} arcsec)')
#             self.log.info(f'  roundness = {roundness:.2f}')
#             self.action.args.fwhm = FWHM_pix
#             self.action.args.ellipticity = np.nan

        # Source Extractor
        pixel_scale = self.cfg['Telescope'].getfloat('pixel_scale', 1)
        thresh = self.cfg['Extract'].getint('extract_threshold', 9)
        minarea = self.cfg['Extract'].getint('extract_minarea', 7)
        mina = self.cfg['Extract'].getfloat('fwhm_mina', 1)
        minb = self.cfg['Extract'].getfloat('fwhm_minb', 1)
        faint_limit_pct = self.cfg['Extract'].getfloat('faint_limit_percentile', 0)
        bright_limit_pct = self.cfg['Extract'].getfloat('bright_limit_percentile', 100)
        radius_limit = self.cfg['Extract'].getfloat('radius_limit_pix', 4000)

        bsub = self.action.args.kd.pixeldata[0].data - self.action.args.background[0].background
        seperr = self.action.args.kd.pixeldata[0].uncertainty.array
        sepmask = self.action.args.kd.pixeldata[0].mask

        def run_sep(bsub, seperr, sepmask, thresh, minarea):
            try:
                objects = sep.extract(bsub, err=seperr, mask=sepmask,
                                      thresh=float(thresh), minarea=minarea)
                return objects
            except Exception as e:
                if str(e)[:27] == 'internal pixel buffer full:':
                    return None
                else:
                    raise SEPError(str(e))

        objects = None
        while objects is None:
            try:
                self.log.info(f'Invoking SEP with threshold: {thresh}')
                objects = run_sep(bsub, seperr, sepmask, thresh, minarea)
                thresh += 9
            except SEPError as e:
                self.log.error('Source extractor failed:')
                self.log.error(e)
                return self.action.args

        t = Table(objects)
        t['flux'] /= exptime

        ny, nx = bsub.shape
        r = np.sqrt((t['x']-nx/2.)**2 + (t['y']-ny/2.)**2)
        t.add_column(Column(data=r.data, name='r', dtype=np.float))

        coef = 2*np.sqrt(2*np.log(2))
        fwhm = np.sqrt((coef*t['a'])**2 + (coef*t['b'])**2)
        t.add_column(Column(data=fwhm.data, name='FWHM', dtype=np.float))

        ellipticities = t['a']/t['b']
        t.add_column(Column(data=ellipticities.data, name='ellipticity', dtype=np.float))

        faint_limit = np.percentile(t['flux'], faint_limit_pct)
        bright_limit = np.percentile(t['flux'], bright_limit_pct)
        self.log.info(f'  Faintest {faint_limit_pct:.1f}% flux {faint_limit:f}')
        self.log.info(f'  Brightest {bright_limit_pct:.1f}% flux {bright_limit:f}')

        filtered = (t['a'] < mina) | (t['b'] < minb) | (t['flag'] > 0) | (t['flux'] > bright_limit) | (t['flux'] < faint_limit) | (t['r'] > radius_limit)
        self.log.debug(f'  Removing {np.sum(filtered):d}/{len(filtered):d}'\
                       f' extractions from FWHM calculation')
        self.log.debug(f"    {np.sum( (t['a'] < mina) )} removed for fwhm_mina limit")
        self.log.debug(f"    {np.sum( (t['b'] < minb) )} removed for fwhm_minb limit")
        self.log.debug(f"    {np.sum( (t['flag'] > 0) )} removed for source extractor flags")
        self.log.debug(f"    {np.sum( (t['flux'] < faint_limit) )} removed for faint limit")
        self.log.debug(f"    {np.sum( (t['flux'] > bright_limit) )} removed for bright limit")

        self.action.args.n_objects = len(t[~filtered])
        self.log.info(f'  Found {self.action.args.n_objects:d} stars')

        self.action.args.objects = t[~filtered]
        self.action.args.objects.sort('flux')
        self.action.args.objects.reverse()

        if self.action.args.n_objects == 0:
            self.log.warning('No stars found')
            return self.action.args
        else:
            FWHM_pix = np.median(t['FWHM'][~filtered])
            FWHM_mode_bin = pixel_scale*0.25
            FWHM_pix_mode = mode(t['FWHM'][~filtered]/FWHM_mode_bin)*FWHM_mode_bin
            self.log.info(f'  Median FWHM = {FWHM_pix:.1f} pix ({FWHM_pix*pixel_scale:.2f} arcsec)')
            self.log.info(f'  Mode FWHM = {FWHM_pix_mode:.1f} pix ({FWHM_pix_mode*pixel_scale:.2f} arcsec)')
            ellipticity = np.median(t['ellipticity'][~filtered])
            ellipticity_mode_bin = 0.05
            ellipticity_mode = mode(t['ellipticity'][~filtered]/ellipticity_mode_bin)*ellipticity_mode_bin
            self.log.info(f'  Median ellipticity = {ellipticity:.2f}')
            self.log.info(f'  Mode ellipticity = {ellipticity_mode:.2f}')
            self.action.args.fwhm = FWHM_pix_mode
            self.action.args.ellipticity = ellipticity_mode

        ## Do photutils photometry measurement
        positions = [(det['x'], det['y']) for det in self.action.args.objects]
        ap_radius = self.cfg['Photometry'].getfloat('aperture_radius', 2)*FWHM_pix
        star_apertures = photutils.CircularAperture(positions, ap_radius)
        sky_apertures = photutils.CircularAnnulus(positions,
                                                  r_in=int(np.ceil(1.5*ap_radius)),
                                                  r_out=int(np.ceil(2.0*ap_radius)))
        phot_table = photutils.aperture_photometry(
                               self.action.args.kd.pixeldata[0],
                               [star_apertures, sky_apertures])
        phot_table['sky'] = phot_table['aperture_sum_1'] / sky_apertures.area()
        med_sky = np.median(phot_table['sky'])
        self.log.info(f'  Median Sky = {med_sky.value:.0f} e-/pix')
        self.action.args.sky_background = med_sky.value
        self.action.args.objects.add_column(phot_table['sky'])
        bkg_sum = phot_table['aperture_sum_1'] / sky_apertures.area() * star_apertures.area()
        final_sum = (phot_table['aperture_sum_0'] - bkg_sum)
        final_uncert = (bkg_sum + final_sum)**0.5 * u.electron**0.5
        phot_table['apflux'] = final_sum/exptime
        self.action.args.objects.add_column(phot_table['apflux'])
        phot_table['apuncert'] = final_uncert/exptime
        self.action.args.objects.add_column(phot_table['apuncert'])
        phot_table['snr'] = final_sum/final_uncert
        self.action.args.objects.add_column(phot_table['snr'])

        where_bad = (final_sum <= 0)
        self.log.info(f'  {np.sum(where_bad)} stars rejected for flux < 0')
        self.action.args.objects = self.action.args.objects[~where_bad]

#         where_low_snr = (self.action.args.objects['snr'] <= 5)
#         self.log.info(f'  {np.sum(where_low_snr)} stars rejected for SNR < 5')
#         self.action.args.objects = self.action.args.objects[~where_low_snr]

        self.log.info(f'  Fluxes for {self.action.args.n_objects:d} stars')

        return self.action.args


##-----------------------------------------------------------------------------
## Primitive: AssociateCalibratorStars
##-----------------------------------------------------------------------------
class AssociateCalibratorStars(BasePrimitive):
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
                  pre_condition(self, 'Have extracted objects',
                                self.action.args.objects is not None),
                  pre_condition(self, 'Have catalog objects',
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

        pixel_scale = self.cfg['Telescope'].getfloat('pixel_scale', 1) * u.arcsec/u.pix
        assoc_radius = self.cfg['Photometry'].getfloat('accoc_radius', 1)\
                       * self.action.args.fwhm*u.pix\
                       * pixel_scale
        associated = Table(names=('RA', 'DEC', 'x', 'y', 'assoc_distance', 'mag', 'catflux', 'flux', 'apflux', 'instmag', 'snr', 'FWHM'),
                           dtype=('f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8') )

        catalog_coords = c.SkyCoord(self.action.args.calibration_catalog['raMean'],
                                    self.action.args.calibration_catalog['decMean'],
                                    unit=(u.deg, u.deg))

        d_telescope = self.cfg['Telescope'].getfloat('d_primary_mm', 508)
        d_obstruction = self.cfg['Telescope'].getfloat('d_obstruction_mm', 127)
        A = 3.14*(d_telescope/2/1000)**2 - 3.14*(d_obstruction/2/1000)**2 # m^2
        self.action.args.f0 = estimate_f0(A, band=self.action.args.band) # photons / sec

        for detected in self.action.args.objects:
            ra_deg, dec_deg = self.action.args.wcs.all_pix2world(detected['x'], detected['y'], 1)
            detected_coord = c.SkyCoord(ra_deg, dec_deg, frame='fk5', unit=(u.deg, u.deg))
            idx, d2d, d3d = detected_coord.match_to_catalog_sky(catalog_coords)
            if d2d[0].to(u.arcsec) < assoc_radius.to(u.arcsec):
#                 instmag = -2.5*np.log10(detected['apflux'].value)
                instmag = -2.5*np.log10(detected['flux'])
                catmag = self.action.args.calibration_catalog[idx][f'{self.action.args.band}MeanApMag']
                zp_for_star = catmag - instmag
                f0_for_star = 10**(zp_for_star/2.5)
                throughput_for_star = f0_for_star/self.action.args.f0
                if throughput_for_star < 1:
                    associated.add_row( {'RA': self.action.args.calibration_catalog[idx]['raMean'],
                                         'DEC': self.action.args.calibration_catalog[idx]['decMean'],
                                         'x': detected['x'],
                                         'y': detected['y'],
                                         'assoc_distance': d2d[0].to(u.arcsec).value, 
                                         'mag': catmag,
                                         'catflux': self.action.args.f0 * 10**(-self.action.args.calibration_catalog[idx][f'{self.action.args.band}MeanApMag']/2.5), # phot/sec
                                         'flux': detected['flux'],
                                         'instmag': instmag,
                                         'apflux': detected['apflux'].value,
                                         'snr': detected['snr'].value,
                                         'FWHM': detected['FWHM'],
                                         } )
        if len(associated) < 2:
            self.action.args.associated_calibrators = None
            return self.action.args

        self.log.info(f'  Associated {len(associated)} catalogs stars')
        self.action.args.associated_calibrators = associated
        self.action.args.associated_calibrators.sort('catflux')

        magdiffs = associated['mag'] - associated['instmag']
        mean, med, std = stats.sigma_clipped_stats(magdiffs,
                               sigma_lower=4, sigma_upper=2, maxiters=5)
        self.action.args.zero_point = med
        self.action.args.zero_point_f0 = 10**(self.action.args.zero_point/2.5)
        self.action.args.throughput = self.action.args.zero_point_f0/self.action.args.f0
        self.log.info(f'  Zero Point = {self.action.args.zero_point:.2f}')
        self.log.debug(f'  Zero Point F0 = {self.action.args.zero_point_f0:.3g}')
        self.log.debug(f'  Estimated F0 = {self.action.args.f0:.3g}')
        self.log.info(f'  Througput = {self.action.args.throughput:.3f}')

        return self.action.args


##-----------------------------------------------------------------------------
## Primitive: AssociateTargetStars
##-----------------------------------------------------------------------------
class AssociateTargetStars(BasePrimitive):
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
                  pre_condition(self, 'Have extracted objects',
                                self.action.args.objects is not None),
                  pre_condition(self, 'Have target catalog objects',
                                self.action.args.target_catalog is not None),
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

        pixel_scale = self.cfg['Telescope'].getfloat('pixel_scale', 1) * u.arcsec/u.pix
        assoc_radius = self.cfg['Photometry'].getfloat('accoc_radius', 1)\
                       * self.action.args.fwhm*u.pix\
                       * pixel_scale
        associated = Table(names=('RA', 'DEC', 'x', 'y', 'assoc_distance', 'mag', 'catflux', 'flux', 'apflux', 'instmag', 'FWHM'),
                           dtype=('f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8') )

        catalog_coords = c.SkyCoord(self.action.args.target_catalog['RA'],
                                    self.action.args.target_catalog['DEC'])

        band = 4760 # Jy (for i band)
        dl = 0.16 # dl/l (for i band)
        d_telescope = self.cfg['Telescope'].getfloat('d_primary_mm', 508)
        d_obstruction = self.cfg['Telescope'].getfloat('d_obstruction_mm', 127)
        A = 3.14*(d_telescope/2/1000)**2 - 3.14*(d_obstruction/2/1000)**2 # m^2
        # 1 Jy = 1.51e7 photons sec^-1 m^-2 (dlambda/lambda)^-1
        # https://archive.is/20121204144725/http://www.astro.utoronto.ca/~patton/astro/mags.html#selection-587.2-587.19
        self.action.args.f0 = band * 1.51e7 * A * dl # photons / sec

        for detected in self.action.args.objects:
            ra_deg, dec_deg = self.action.args.wcs.all_pix2world(detected['x'], detected['y'], 1)
            detected_coord = c.SkyCoord(ra_deg, dec_deg, frame='fk5', unit=(u.deg, u.deg))
            idx, d2d, d3d = detected_coord.match_to_catalog_sky(catalog_coords)
            if d2d[0].to(u.arcsec) < assoc_radius.to(u.arcsec):
                associated.add_row( {'RA': self.action.args.target_catalog[idx]['RA'],
                                     'DEC': self.action.args.target_catalog[idx]['DEC'],
                                     'x': detected['x'],
                                     'y': detected['y'],
                                     'assoc_distance': d2d[0].to(u.arcsec).value, 
                                     'mag': self.action.args.target_catalog[idx]['mag'],
                                     'catflux': self.action.args.f0 * 10**(-self.action.args.target_catalog[idx]['mag']/2.5), # phot/sec
                                     'flux': detected['flux'],
                                     'instmag': -2.5*np.log10(detected['apflux']),
                                     'apflux': detected['apflux'],
                                     'FWHM': detected['FWHM'],
                                     } )
        if len(associated) < 2:
            self.action.args.associated = None
            return self.action.args

        self.log.info(f'  Measured {len(associated)} target catalog stars')
        self.action.args.measured = associated
        self.action.args.measured.sort('flux')

#         nclip = int(np.floor(0.05*len(associated['catflux'])))
#         fitted_line = sigma_clipping_line_fit(associated['catflux'][nclip:-nclip],
#                                               associated['apflux'][nclip:-nclip],
#                                               intercept_fixed=True)
#         self.log.info(f"  Slope (e-/photon) = {fitted_line.slope.value:.3g}")
#         self.action.args.zero_point_fit = fitted_line
#         deltas = associated['apflux'] - fitted_line(associated['catflux'])
#         mean, med, std = stats.sigma_clipped_stats(deltas)
#         self.log.info(f"  Fit StdDev = {std:.2g}")

        return self.action.args


if __name__ == '__main__':
    pass
