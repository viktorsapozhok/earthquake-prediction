from setuptools import setup

SHORT_DESCRIPTION = ''

DEPENDENCIES = [
    "joblib",
    "numpy",
    "pandas==0.23.4",
    "scipy",
    "scikit-learn==0.21.2",
    "tqdm",
    "rfpimp==1.3.4",
    "catboost==0.15",
    "deap==1.2.2",
    "librosa==0.6.3",
    "numba==0.48",
#    "tables==3.5.1",
    "tsfresh==0.11.2",
]

VERSION = '0.0.1'
URL = 'https://github.com/viktorsapozhok/earthquake-prediction'

setup(
    name='earthquake',
    version=VERSION,
    description=SHORT_DESCRIPTION,
    long_description=SHORT_DESCRIPTION,
    url=URL,
    author='viktorsapozhok',
    license='MIT License',
    packages=['earthquake'],
    install_requires=DEPENDENCIES,
    python_requires="==3.7.9"
)