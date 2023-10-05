(installation)=

# Installation Guide for Snake fMRI

Snake fMRI is a Python package designed for simulating functional Magnetic Resonance Imaging (fMRI) data. This guide will walk you through the installation process to get Snake fMRI up and running on your system.

## Prerequisites

Before installing Snake fMRI, make sure you have the following prerequisites installed on your system:

- `Python` (version **3.10** or higher)
- `pip` (Python package manager)

## Installation Steps

Follow these steps to install Snake fMRI:

### 1. Create a Virtual Environment (Optional but Recommended)

Creating a virtual environment is recommended to isolate Snake fMRI and its dependencies from your system-wide Python installation. You can skip this step if you prefer to install packages globally.

```bash
# Create a virtual environment (optional)
python -m venv snake_fmri_env

# Activate the virtual environment
# On Windows:
snake_fmri_env\Scripts\activate
# On macOS and Linux:
source snake_fmri_env/bin/activate
```

### 2. Install Snake fMRI

Use pip to install the Snake fMRI package from the Python Package Index (PyPI). Run the following command:

```bash
pip install snake-fmri
```

### 3. Verify the Installation

After installation, you can verify that Snake fMRI is correctly installed by checking its version:

```bash
snake-fmri --version
```

You should see the installed version of Snake fMRI displayed in the terminal.

### 4. Usage

You can now start using Snake fMRI to simulate fMRI data. Make sure to refer to the package's documentation or examples provided by the authors to learn how to use it effectively.

## Updating Snake fMRI

To update Snake fMRI to the latest version, you can use the following command:

```bash
pip install --upgrade snake-fmri
```

## Uninstallation

If you ever need to uninstall Snake fMRI, you can use the following command:

```bash
pip uninstall snake-fmri
```

If you created a virtual environment, you can deactivate it by running:

```bash
deactivate
```

Now you have successfully installed Snake fMRI and can start using it to simulate fMRI data for your research or projects. Be sure to consult the package's documentation for detailed usage instructions and examples.
