import pytest

@pytest.fixture(params=
                [
                 ('wfss', 'test/data/jw01895001002_04101_00001_nrcalong_cal.fits'), 
                 ('image', 'test/data/jw01895001002_04101_00001_nrca1_cal.fits')
                 ]
                )
def example_file_path(request):
    return request.param
