from astropy import units as u
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


def get_moon_info(lat, lon, height, temperature, pressure, time):
    loc = None