from setuptools import setup, find_packages
from pathlib import Path

this_dir = Path(__file__).resolve().parent
with (this_dir / "requirements.txt").open() as rf: 
    install_requires = [ 
        req.strip()
        for req in rf.readlines()
        if req.strip() and not req.startswith("#")
    ]   

setup(
    name="BinFit",
    version="0.0.0",
    author="Maximilian Welsch, William Sutcliffe, Felix Metner",
    url="https://gitlab.ekp.kit.edu/sutclw/BinFit",
    packages=find_packages(),
    description=""" Perform extended binnend log-likelhood fits using histogram templates as pdfs.
                 Rewrite of the original TemplateFitter of Maximilian Welsch""",
    install_requires=install_requires
)
