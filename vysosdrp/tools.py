import numpy as np
from astropy import units as u
from astropy.time import Time
from astropy.table import Table, Column
import astropy.coordinates as c
from astroquery.vizier import Vizier

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
    filter_string = '>0' if maglimit is None else f"<{maglimit}"
    column_filters = {'UCAC4': {"imag": filter_string},
                      'Gaia': {"RPmag": filter_string} }

    v = Vizier(columns=columns[catalog],
               column_filters=column_filters[catalog])
    v.ROW_LIMIT = 1e4

    stars = Table(v.query_region(pointing, catalog=catalogs[catalog],
                                 radius=c.Angle(radius, "deg"))[0])
    stars.add_column( Column(data=stars[ra_colname[catalog]], name='RA') )
    stars.add_column( Column(data=stars[dec_colname[catalog]], name='DEC') )
    return stars


def get_moon_info(lat=None, lon=None, height=None, temperature=None,
                  pressure=None, time=None, pointing=None):
    loc = c.EarthLocation(lon, lat, height)
    altazframe = c.AltAz(location=loc, obstime=time,
                         temperature=temperature,
                         pressure=pressure)
    moon = c.get_moon(Time(time), location=loc)
    sun = c.get_sun(Time(time))

    moon_alt = ((moon.transform_to(altazframe).alt).to(u.degree)).value
    moon_separation = (moon.separation(pointing).to(u.degree)).value if pointing is not None else None

    # Moon illumination formula from Meeus, “Astronomical 
    # Algorithms". Formulae 46.1 and 46.2 in the 1991 edition, 
    # using the approximation cos(psi) \approx -cos(i). Error 
    # should be no more than 0.0014 (p. 316). 
    moon_illum = 50*(1 - np.sin(sun.dec.radian)*np.sin(moon.dec.radian)\
                 - np.cos(sun.dec.radian)*np.cos(moon.dec.radian)\
                 * np.cos(sun.ra.radian-moon.ra.radian))
    return moon_alt, moon_separation, moon_illum
