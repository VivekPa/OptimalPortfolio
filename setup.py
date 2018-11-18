from setuptools import setup

setup(
    name='PortfolioAnalytics',
    version='0.0.1',
    packages=['portfolioopt'],
    url='https://github.com/VivekPa/PortfolioAnalytics',
    license='MIT',
    author='Vivek Palaniappan',
    author_email='vivekpalaniappan69@gmail.com',
    description='Robust Portfolio Optimisation and Analytics',
    install_requires = ["numpy", "pandas", "scikit-learn", "scipy"],
    python_requires = ">=3",
    project_urls = {
    "Documentation": "https://portfolioanalytics.readthedocs.io/en/latest/",
    "Issues": "https://github.com/VivekPa/PortfolioAnalytics/issues",
    "Personal website": "https://medium.com/engineer-quant"},
)
