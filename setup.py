from setuptools import setup

setup(
    name='optimalportfolio',
    version='0.0.3',
    packages=['optimalportfolio'],
    url='https://github.com/VivekPa/OptimalPortfolio',
    license='MIT',
    author='Vivek Palaniappan and Sven Serneels',
    author_email='vivekpalaniappan69@gmail.com',
    description='Portfolio Optimization libary in Python',
    install_requires = ["numpy", "pandas", "scikit-learn", "scipy", "rpy2"],
    python_requires = ">=3",
    project_urls = {
    "Documentation": "",
    "Issues": "https://github.com/VivekPa/OptimalPortfolio/issues",
    "Personal website": "https://medium.com/engineer-quant"},
)
