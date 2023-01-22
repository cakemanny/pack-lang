from setuptools import setup, find_packages

# Updated by a script (in the future maybe)
VERSION = "0.1.0-dev"

with open('README.md', 'r', encoding='utf-8') as f:
    readme = f.read()

setup(
    name='pack-lang',
    version=VERSION,
    packages=find_packages(exclude=('tests*',)),

    # https://stackoverflow.com/a/1857436
    package_data={'pack': ['core.pack', 'repl.pack']},
    include_package_data=True,

    author='Daniel Golding',
    description='Quickly serialize dataclasses to and from JSON',
    long_description=readme,
    long_description_content_type='text/markdown',
    url='https://github.com/cakemanny/pack-lang',
    project_urls={
        'Bug Tracker': 'https://github.com/cakemanny/pack-lang/issues'
    },
    license='MIT',
    keywords='immutable data structures lisp',
    python_requires='>=3.10',
    extras_require={
        'dev': [
            'pytest',
            'flake8',
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Development Status :: 3 - Alpha",
    ],
    scripts=['.bin/pack']
)
