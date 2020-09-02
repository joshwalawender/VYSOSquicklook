from pathlib import Path
from datetime import datetime, timedelta
import sys
import re
import subprocess
from matplotlib import pyplot as plt

import numpy as np
from astropy.io import fits
from astropy import units as u
from astropy import stats
from astropy.time import Time
import astropy.coordinates as c
from astropy.wcs import WCS
from astropy.table import Table, Column
from astropy.modeling import models, fitting
from astroquery.vizier import Vizier
import ccdproc
import photutils
import sep

from keckdata import fits_reader, VYSOS20

from keckdrpframework.primitives.base_primitive import BasePrimitive
from keckdrpframework.models.arguments import Arguments

from .utils import pre_condition, post_condition


##-----------------------------------------------------------------------------
## Function: get_catalog
##-----------------------------------------------------------------------------
def get_catalog(pointing, radius, catalog='UCAC4', maglimit=None):

    catalogs = {'UCAC4': 'I/322A', 'Gaia': 'I/345/gaia2'}
    if catalog not in catalogs.keys():
        print(f'{catalog} not in {catalogs.keys()}')
        raise NotImplementedError

    columns = {'UCAC4': ['_RAJ2000', '_DEJ2000', 'rmag', 'imag'],
               'Gaia': ['RA_ICRS', 'DE_ICRS', 'Gmag', 'RPmag']}
    ra_colname = {'UCAC4': '_RAJ2000',
                  'Gaia': 'RA_ICRS'}
    dec_colname = {'UCAC4': '_DEJ2000',
                   'Gaia': 'DE_ICRS'}
    mag_colname = {'UCAC4': 'imag',
                   'Gaia': 'RPmag'}
    filter_string = '>0' if maglimit is None else f"<{maglimit}"
    column_filter = {mag_colname[catalog]: filter_string}
#     column_filters = {'UCAC4': {"imag": filter_string},
#                       'Gaia': {"RPmag": filter_string} }

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
## Function: sigma_clipping_line_fit
##-----------------------------------------------------------------------------
def sigma_clipping_line_fit(xdata, ydata, nsigma=5, maxiter=3, maxcleanfrac=0.2,
                            intercept_fixed=False, intercept0=0, slope0=1):
        npoints = len(xdata)
        fit = fitting.LinearLSQFitter()
        line_init = models.Linear1D(slope=slope0, intercept=intercept0)
        line_init.intercept.fixed = intercept_fixed
        fitted_line = fit(line_init, xdata, ydata)
        deltas = ydata - fitted_line(xdata)
        mean, median, std = stats.sigma_clipped_stats(deltas)
        cleaned = np.array(abs(deltas) < nsigma*std)
        for iteration in range(1, maxiter):
            last_std = std
            cleaned = cleaned | np.array(abs(deltas) < nsigma*std)
            if np.sum(cleaned)/npoints > maxcleanfrac:
                return fitted_line
            new_fit = fit(line_init, xdata[cleaned], ydata[cleaned])
            deltas = ydata - new_fit(xdata)
            mean, median, std = stats.sigma_clipped_stats(deltas)
            if std >= last_std:
                return new_fit
            else:
                fitted_line = new_fit

        return fitted_line


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

        # Moon illumination formula from Meeus, “Astronomical 
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

        bsub = self.action.args.kd.pixeldata[0].data - self.action.args.background[0].background
        try:
            objects = sep.extract(bsub,
                                  err=self.action.args.kd.pixeldata[0].uncertainty.array,
                                  mask=self.action.args.kd.pixeldata[0].mask,
                                  thresh=float(thresh), minarea=minarea)
        except Exception as e:
            self.log.error('Source extractor failed')
            self.log.error(e)
            self.action.args.fwhm = np.nan
            self.action.args.ellipticity = np.nan
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

        filtered = (t['a'] < mina) | (t['b'] < minb) | (t['flag'] > 0)
        self.log.debug(f'  Removing {np.sum(filtered):d}/{len(filtered):d}'\
                       f' extractions from FWHM calculation')
        self.action.args.n_objects = len(t[~filtered])
        self.log.info(f'  Found {self.action.args.n_objects:d} stars')

        self.action.args.objects = t[~filtered]
        self.action.args.objects.sort('flux')
        self.action.args.objects.reverse()

        if self.action.args.n_objects == 0:
            self.log.warning('No stars found')
            self.action.args.fwhm = np.nan
            self.action.args.ellipticity = np.nan
        else:
            FWHM_pix = np.median(t['FWHM'][~filtered])
            ellipticity = np.median(t['ellipticity'][~filtered])
            self.log.info(f'  Median FWHM = {FWHM_pix:.1f} pix ({FWHM_pix*pixel_scale:.2f} arcsec)')
            self.log.info(f'  ellipticity = {ellipticity:.2f}')
            self.action.args.fwhm = FWHM_pix
            self.action.args.ellipticity = ellipticity

        ## Do second photometry measurement
        positions = [(det['x'], det['y']) for det in self.action.args.objects]
        ap_radius = int(2.0*FWHM_pix)
        star_apertures = photutils.CircularAperture(positions, ap_radius)
        sky_apertures = photutils.CircularAnnulus(positions,
                                                  r_in=ap_radius+2,
                                                  r_out=ap_radius+6)
        phot_table = photutils.aperture_photometry(
                               self.action.args.kd.pixeldata[0],
                               [star_apertures, sky_apertures])
        phot_table['sky'] = phot_table['aperture_sum_1'] / sky_apertures.area()
        med_sky = np.median(phot_table['sky'])
        self.log.info(f'  Median Sky = {med_sky.value:.0f} e-/pix')
        self.action.args.sky_background = med_sky.value
        self.action.args.objects.add_column(phot_table['sky'])
        bkg_sum = phot_table['aperture_sum_1'] / sky_apertures.area() * star_apertures.area()
        final_sum = (phot_table['aperture_sum_0'] - bkg_sum)/exptime
        where_bad = (final_sum <= 0)
        phot_table['flux2'] = final_sum
        self.action.args.objects.add_column(phot_table['flux2'])
        self.action.args.objects = self.action.args.objects[~where_bad]
        self.log.info(f'  Fluxes for {self.action.args.n_objects:d} stars')

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
        force_solve = self.cfg['Telescope'].getboolean('force_solve', False)
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
        ast.api_key = self.cfg['astrometry'].get('api_key', None)
        solve_timeout = self.cfg['astrometry'].getint('solve_timeout', 90)
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
## Primitive: GetCatalogStars
##-----------------------------------------------------------------------------
class GetCatalogStars(BasePrimitive):
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
        catalog = self.cfg['jpeg'].get('catalog', None)
        known_catalogs = ['Gaia', 'UCAC4']
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

        catalogname = self.cfg['jpeg'].get('catalog')
        maglimit = self.cfg['jpeg'].get('catalog_maglimit')

        fp = self.action.args.wcs.calc_footprint(axes=self.action.args.kd.pixeldata[0].data.shape)
        dra = fp[:,0].max() - fp[:,0].min()
        ddec = fp[:,1].max() - fp[:,1].min()
        radius = np.sqrt((dra*np.cos(fp[:,1].mean()*np.pi/180.))**2 + ddec**2)/2.

        if self.action.args.wcs_pointing is not None:
            self.log.debug('Using WCS pointing for catalog query')
            pointing = self.action.args.wcs_pointing
        else:
            self.log.warning('Using header pointing for catalog query')
            pointing = self.action.args.header_pointing

        self.log.info(f"Retrieving {catalogname} entries (magnitude < {maglimit})")
        self.action.args.catalog = get_catalog(pointing, radius, catalog=catalogname, maglimit=maglimit)
        ncat = len(self.action.args.catalog) if self.action.args.catalog is not None else 0
        self.log.info(f"  Found {ncat} catalog entries")

        return self.action.args


##-----------------------------------------------------------------------------
## Primitive: AssociateCatalogStars
##-----------------------------------------------------------------------------
class AssociateCatalogStars(BasePrimitive):
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
                                self.action.args.catalog is not None),
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
        assoc_radius = self.cfg['Extract'].getfloat('accoc_radius', 1)\
                       * self.action.args.fwhm*u.pix\
                       * pixel_scale
        associated = Table(names=('RA', 'DEC', 'x', 'y', 'assoc_distance', 'mag', 'catflux', 'flux', 'flux2', 'instmag', 'FWHM'),
                           dtype=('f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8') )

        catalog_coords = c.SkyCoord(self.action.args.catalog['RA'], self.action.args.catalog['DEC'])

        band = 4760 # Jy (for i band)
        dl = 0.16 # dl/l (for i band)
        d_telescope = self.cfg['Telescope'].getfloat('d_primary_mm', 508)
        d_obstruction = self.cfg['Telescope'].getfloat('d_obstruction_mm', 127)
        A = 3.14*(d_telescope/2/1000)**2 - 3.14*(d_obstruction/2/1000)**2 # m^2
        # 1 Jy = 1.51e7 photons sec^-1 m^-2 (dlambda/lambda)^-1
        # https://archive.is/20121204144725/http://www.astro.utoronto.ca/~patton/astro/mags.html#selection-587.2-587.19
        self.action.args.f0 = band * 1.51e7 * A * dl # photons / sec

#         self.log.info(self.action.args.objects.keys())
        for detected in self.action.args.objects:
            ra_deg, dec_deg = self.action.args.wcs.all_pix2world(detected['x'], detected['y'], 1)
            detected_coord = c.SkyCoord(ra_deg, dec_deg, frame='fk5', unit=(u.deg, u.deg))
            idx, d2d, d3d = detected_coord.match_to_catalog_sky(catalog_coords)
            if d2d[0].to(u.arcsec) < assoc_radius.to(u.arcsec):
                associated.add_row( {'RA': self.action.args.catalog[idx]['RA'],
                                     'DEC': self.action.args.catalog[idx]['DEC'],
                                     'x': detected['x'],
                                     'y': detected['y'],
                                     'assoc_distance': d2d[0].to(u.arcsec).value, 
                                     'mag': self.action.args.catalog[idx]['mag'],
                                     'catflux': self.action.args.f0 * 10**(-self.action.args.catalog[idx]['mag']/2.512), # phot/sec
                                     'flux': detected['flux'],
                                     'instmag': -2.512*np.log(detected['flux']),
                                     'flux2': detected['flux2'],
                                     'FWHM': detected['FWHM'],
                                     } )
        if len(associated) == 0:
            self.action.args.associated = None
            return self.action.args

        self.log.info(f'  Associated {len(associated)} catalogs stars')
        self.action.args.associated = associated
        self.action.args.associated.sort('catflux')
        nclip = int(np.floor(0.05*len(associated['catflux'])))
        fitted_line = sigma_clipping_line_fit(associated['catflux'][nclip:-nclip],
                                              associated['flux2'][nclip:-nclip],
                                              intercept_fixed=True)
        self.log.info(f"  Slope (e-/photon) = {fitted_line.slope.value:.3g}")
        self.action.args.zero_point_fit = fitted_line
        deltas = associated['flux2'] - fitted_line(associated['catflux'])
        mean, med, std = stats.sigma_clipped_stats(deltas)
        self.log.info(f"  Fit StdDev = {std:.2g}")

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

