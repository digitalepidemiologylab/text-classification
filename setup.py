import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="txcl",
    version="0.0.1",
    author="Crowdbreaks",
    author_email="info@crowdbreaks.org",
    description="Reproducible text classification",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/crowdbreaks/text-classification",
    packages=setuptools.find_packages(),
    # Updated for fasttext model
    install_requires=[
        'dacite', 'tqdm', 'pandas', 'swifter', 'numpy', 'scikit-learn',
        'hyperopt', 'matplotlib', 'seaborn', 'visdom', 'fasttext',
        'boto3', 'munch', 'docker', 'joblib',
        'twiprocess @ git+https://github.com/crowdbreaks/twiprocess.git'],
    entry_points={'console_scripts': [
        'txcl=txcl.cli.cli:entry_point']},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent"],
    python_requires='>=3.6')
