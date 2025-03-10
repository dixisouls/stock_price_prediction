from setuptools import setup, find_packages

setup(
    name="stock_price_prediction",
    version="0.1.0",
    author="divyapanchal",
    description="Stock price prediction using PyTorch Transformer",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
        "yfinance>=0.1.70",
        "scikit-learn>=1.0.0",
        "torch>=1.10.0",
        "python-dotenv>=0.19.0",
    ],
)