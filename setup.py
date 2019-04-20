from setuptools import setup

setup(
    name='portfolioopt',
    version='0.0.2',
    packages=['portfolioopt'],
    url='https://github.com/VivekPa/PortfolioAnalytics',
    license='MIT',
    author='Vivek Palaniappan and Sven Serneels',
    author_email='vivekpalaniappan69@gmail.com',
    description='Portfolio Optimisation and Analytics',
    install_requires = ["numpy", "pandas", "scikit-learn", "scipy"],
    python_requires = ">=3",
    project_urls = {
    "Documentation": "https://portfolioanalytics.readthedocs.io/en/latest/",
    "Issues": "https://github.com/VivekPa/PortfolioAnalytics/issues",
    "Personal website": "https://medium.com/engineer-quant"},
)
