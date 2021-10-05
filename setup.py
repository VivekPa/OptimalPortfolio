from setuptools import setup, find_packages

setup(
    name='OptimalPortfolio',
    version='0.0.1',
    packages=find_packages(),
    url='https://github.com/VivekPa/OptimalPortfolio',
    license='MIT',
    author='Vivek Palaniappan',
    author_email='vivekpalaniappan69@gmail.com',
    description='Portfolio Optimisation and Analytics Library',
    # install_requires = ["numpy", "pandas", "scikit-learn", "scipy", "cvxpy"],
    python_requires = ">=3.5",
    project_urls = {
    "Issues": "https://github.com/VivekPa/OptimalPortfolio/issues",
    "Personal website": "https://medium.com/engineer-quant"}
)
