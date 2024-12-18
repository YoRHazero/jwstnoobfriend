import pytest
from jwstnoobfriend import box
import sys

fb1 = box.FileBox(container=1)
fb2 = box.FileBox(container=2)
fb3 = box.FileBox(container={'a': 1, 'b': 2})

print(sys.modules[__name__].__dict__.keys())

print(sys.modules[box.__name__].__dict__.keys()) 