import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="distilled_ctr",
    version="0.0.1",
    author="Aljo",
    author_email="aljo.inbox@gmail.com",
    description="Distilled CTR Prediction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/aljo-jose/DistilledCTR.git",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU GENERAL PUBLIC LICENSE",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)