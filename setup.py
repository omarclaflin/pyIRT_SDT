from setuptools import setup, find_packages

setup(
    name='PyIRT_SDT',
    version='0.1.5',
    packages=find_packages(),
    install_requires=[
        # List your package dependencies here
    ],
    url='https://github.com/omarclaflin/pyIRT_SDT',
    author='Omar Claflin',
    author_email='your.email@example.com',
    description='Python IRT calculator with SDT outputs, too, handling continuous and sparse data.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
