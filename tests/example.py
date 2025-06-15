import astropy.table
from astroquery.mast import MastMissionsClass
import astropy
import requests
from collections import Counter
mission = MastMissionsClass(mission="jwst")

response: astropy.table.Table = mission.query_criteria( # type: ignore
    program='01895',
    productLevel='1b'
)

"""
products = mission.get_unique_product_list(
    response['fileSetName'][0],
    )
selected = mission.filter_products(
    products,
    category=['1b'],
)
"""
response.write('test.txt', format='ascii.fixed_width', overwrite=True)
