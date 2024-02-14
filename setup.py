from setuptools import setup, find_packages

setup(
    name='qiskit_experiments',
    version='1.0.0',
    author='IM',
    author_email='',
    description='A collection of Qiskit Pulse codes',
    long_description='A collection of Qiskit Pulse codes that were used to produce three separate scientific works during my PhD.',
    long_description_content_type='text/markdown',
    url='https://github.com/ivo53/qiskit_experiments',
    packages=find_packages(),  # Automatically find packages in the 'src' directory
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    python_requires='>=3.6',
    install_requires=[
        'numpy>=1.0.0',
        'qiskit>=1.0.0',
        'matplotlib>=3.0.0'
    ],
)