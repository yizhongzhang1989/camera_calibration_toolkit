"""
Setup script for Camera Calibration Toolkit
"""

from setuptools import setup, find_packages

with open("camera_calibration_toolkit/README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("camera_calibration_toolkit/requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="camera-calibration-toolkit",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A comprehensive camera calibration toolkit with web interface",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/camera-calibration-toolkit",
    packages=find_packages(where="camera_calibration_toolkit"),
    package_dir={"": "camera_calibration_toolkit"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.9",
        ],
        "viz": [
            "matplotlib>=3.3.0",
            "plotly>=5.0.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "camera-calibrate=main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "web": ["static/*/*", "templates/*"],
    },
)
