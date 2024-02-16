# Downscaling Near-Surface Air Temperature


## Description
This project aims to downscale near-surface air temperature (Ta) predictions from global climate models to finer spatial resolutions using machine learning techniques. By leveraging a diverse set of features, including satellite data, climate model outputs, and digital elevation models, we develop accurate machine learning models to provide high-resolution Ta data for climate studies.

## Project Structure
**data:** Contains the datasets required for training and validation.

**load_and_process_file:** Scripts for loading and preprocessing data files.

**models:** Implementation of machine learning models for training.

**validation:** Code for validating the models.

**output:** Stores the results and plots generated by the models.

**WeatherDataProcessor:** Modules for processing weather data.

**main_script:** Main script to execute the downscaling process.

## Requirements
Python 3.10

Required Python packages: The required Python packages and their version numbers are listed in the `requirements.txt` file.

## Setup Instructions
1. Clone the repository:
```bash
git clone https://github.com/estipollak/Downscaling-Near-Surface-Air-Temperature.git
```

2. Navigate to the project directory:
```bash
cd Downscaling-Near-Surface-Air-Temperature.git
```

3.Install required Python packages:
```bash
pip install -r requirements.txt
```

4. Copy data [from Google Drive](https://drive.google.com/drive/folders/1fJq7GHRrrf9pk9E-lTuWwubY_5LrWO2m?usp=sharing) to the data\ERA5 folder.


## Usage
1. Execute the main script to start the downscaling process:
```bash
python main.py
```
2. Monitor the progress and check the output folder and console for results.

## Results
Our models achieved impressive accuracy, with RMSE value as follows:

ERA5 hourly Ta: 1.38°C

## Example Output Results
In the `Output` folder of this repository, you can find example results generated by running the main script. These results serve as an illustration of the output format and the performance of the downscaling process. Feel free to explore the plots and data files to gain insights into the downscaling results.

## References
For more details, please refer to the article:

[Add Reference Article Link Here]
