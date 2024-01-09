import setuptools

setuptools.setup(
    name="multiverse_pytorch",  # Replace with your own package name
    version="0.1.0",  # The initial release version
    author="Yutong Wang",  # Replace with your name
    author_email="yutongw@umich.edu",  # Replace with your email
    description="A collection of PyTorch loss functions for various tasks",  # A short description
    long_description=open('README.md').read(),  # Long description read from the the readme file
    long_description_content_type="text/markdown",
    url="https://github.com/YutongWangUMich/multiverse_pytorch",  # Replace with the URL to your repository
    packages=setuptools.find_packages(),
    install_requires=[
        "torch",  # Assuming PyTorch is a dependency; specify the version if necessary
        # Add any other dependencies required by your package
    ],
    classifiers=[
        # Classifiers help users find your project by categorizing it
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",  # Replace with your chosen license
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',  # Minimum version requirement of the Python for your package
)
