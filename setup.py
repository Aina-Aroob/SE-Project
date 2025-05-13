from setuptools import setup, find_packages

setup(
    name='stream_overlay_renderer',
    version='0.1',
    description='Overlay ball trajectory and decisions on cricket videos',
    author='Kainat Nasir, Laiba Naseer, Arham Jahangir, Muhammed Umer',
    author_email='kainatnasir1@gmail.com',
    packages=find_packages(),
    install_requires=[
        'opencv-python',
        'numpy',
        'flask'
    ],
)