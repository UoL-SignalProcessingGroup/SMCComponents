import setuptools

setup(
    name="smccomponents",
    version="0.1.0",
    description="A comprehensive toolbox for executing Sequential Monte Carlo (SMC) methods.",
    author=["Alessandro Varsi", "Matthew Carter"],
    author_email=["a.varsi@liverpool.ac.uk", "m.j.carter2@liverpool.ac.uk"],
    packages=setuptools.find_packages(),
    python_requires='>=3.6',
    install_requires=[
        "numpy>=1.19",
        "mpi4py>=3.1",
    ],
)
