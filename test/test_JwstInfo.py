import pytest
import os
import sys
import logging
from jwstnoobfriend.box import FileBox, JwstInfo
#from jwstnoobfriend import box

def test_filebox_read_and_load(example_file_path, tmpdir):
    FileBox.set_logger(log_level=logging.CRITICAL)
    image_type, file_name = example_file_path
    if image_type == 'wfss':

        print('########'*2)
        print([j for j in sys.modules[FileBox.__module__].__dict__.keys() if 'CustomJwstInfo' in j])
        print('########'*2)
        #assert 'CustomJwstInfo1' not in sys.modules[FileBox.__module__].__dict__.keys()
        file_box = FileBox(container={'detector': str, 'filepaths': dict})
        print('########'*2)
        print([j for j in sys.modules[FileBox.__module__].__dict__.keys() if 'CustomJwstInfo' in j])
        print('########'*2)
        file_box.set_logger(log_level=logging.DEBUG)
        file_box.from_filename(file_name)
        file_box.add_file_path('cal', file_name)
        assert image_type in ['wfss', 'image']
        file_box.save_to_file(tmpdir / 'test_filebox.pkl')
        
        assert 'CustomJwstInfo3' in sys.modules[FileBox.__module__].__dict__.keys()
        del sys.modules[FileBox.__module__].__dict__['CustomJwstInfo3']
        print('########'*2)
        print([j for j in sys.modules[FileBox.__module__].__dict__.keys() if 'CustomJwstInfo' in j])
        print('########'*2)
        
        #assert 'CustomJwstInfo1' in sys.modules[FileBox.__module__].__dict__.keys()

        new_file_box = FileBox(container={'detector': str, 'pupil': str})
        print('########'*2)
        print([j for j in sys.modules[FileBox.__module__].__dict__.keys() if 'CustomJwstInfo' in j])
        print('########'*2)

        new_file_box.load_from_file(tmpdir / 'test_filebox.pkl')


    #assert new_file_box.container == 2
    
    
    
    
