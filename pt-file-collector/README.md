# pt-file-collector

## Overview
The pt-file-collector is a Python project designed to search for all `.pt` files in a specified directory and move them to a designated folder. This tool is useful for organizing model files, checkpoints, or any other data stored in the `.pt` format.

## Features
- Recursively searches through directories for `.pt` files.
- Moves found files to a specified output directory.
- Simple command-line interface for ease of use.

## Installation
1. Clone the repository:
   ```
   git clone <repository-url>
   ```
2. Navigate to the project directory:
   ```
   cd pt-file-collector
   ```
3. (Optional) Create a virtual environment and activate it:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```
4. Install any required dependencies (if applicable).

## Usage
To use the pt-file-collector, run the following command in your terminal:

```
python src/collect_pt_files.py <source_directory> <destination_directory>
```

Replace `<source_directory>` with the path to the directory you want to search and `<destination_directory>` with the path to the folder where you want to collect the `.pt` files.

## Example
```
python src/collect_pt_files.py /path/to/search /path/to/collect
```

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.