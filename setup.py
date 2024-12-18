from setuptools import setup, find_packages

setup(
    name='noobfriend',
    version='0.1.0',
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'nbfriend-noobook = jwstnoobfriend.cli:main'
        ]
    },
)