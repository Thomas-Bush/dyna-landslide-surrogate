from setuptools import setup, find_packages

# Read the contents of your README file
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# Read the contents of the requirements file
requirements = (this_directory / "requirements.txt").read_text().splitlines()

setup(
    name='project_name',  # Replace with your project's name
    version='0.1.0',  # Replace with your project's version
    author='Your Name',  # Replace with your name
    author_email='your.email@example.com',  # Replace with your email address
    description='A brief description of your project',  # Provide a short description
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='http://github.com/yourusername/project_name',  # Replace with the URL of your project
    packages=find_packages(),
    install_requires=requirements,
    entry_points={
        'console_scripts': [
            'project_name=main:main',  # This assumes your main.py has a main function
        ],
    },
    python_requires='>=3.11.7',
    classifiers=[
        'Development Status :: 3 - Alpha',  # Change as appropriate for your project's maturity
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',  # Replace with the appropriate license
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.11',
        'Operating System :: OS Independent',
    ],
    keywords='machine learning, debris flow modeling, CNN-LSTM',  # Add keywords relevant to your project
    # If your package includes data files, they can be included like this:
    # package_data={
    #     'sample': ['package_data.dat'],
    # },
    # You could also include extra requirements depending on the environment like this:
    # extras_require={
    #     'dev': ['check-manifest'],
    #     'test': ['coverage'],
    # },
)