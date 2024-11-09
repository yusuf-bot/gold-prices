# Gold Stock Transition Analysis

This project analyzes historical gold stock price data and visualizes the transition probabilities between different price change states using a directed graph. The analysis helps in understanding the behavior of gold prices over time.

## Table of Contents

- [Usage](#usage)
- [Data Source](#data-source)
- [Visualization](#visualization)
- [Dependencies](#dependencies)
- [License](#license)

## Usage
Place your goldstock.csv file in the project directory. The CSV file should contain the historical gold stock prices with at least two columns: date and price.

Run the analysis script:

```
python gold.py
```
The script will read the data, compute the transition probabilities between different price change states, and display a directed graph representing these transitions.

## Data Source
The data used in this project is stored in a CSV file named goldstock.csv. This file should contain historical gold stock prices with the following format:
```
Date,Close
2023-01-01,1800.00
2023-01-02,1810.00
```
Make sure to adjust the file path in the script if your data file is located elsewhere.

## Visualization

The project uses Matplotlib and NetworkX to visualize the transition probabilities as a directed graph. The nodes represent different price change states, and the edges represent the probabilities of transitioning from one state to another. And the project makes use of Tabulate to visualise the most important bits of data as a table.

Nodes are colored red for decreasing states and green for increasing states.
The width of the edges corresponds to the transition probability.

## Dependencies

This project requires the following Python packages:

numpy
matplotlib
tabulate
networkx
You can install these packages via pip as mentioned in the Installation section.


## Next Steps

The next steps are to utilise the rest of the data available, such as the open, high, low, and volume metrics, to create more states and be able to draw more conclusive results than present.


## License
This project is licensed under the MIT License - see the LICENSE file for details.


### Customization Tips:
- **Project Title**: Ensure the title reflects the actual name of your project.
- **GitHub Link**: Replace `https://github.com/yourusername/gold-stock-transition-analysis.git` with the actual URL of your GitHub repository.
- **Data Format**: Adjust the data format description if your CSV file has a different structure.
- **License**: If you have a specific license, include it or link to the license file.
- **Additional Features**: If your project has more features or details (like how to interpret the graph), feel free to add those sections.

This README file serves as a comprehensive guide for users who wish to understand, install, and use your project effectively.
