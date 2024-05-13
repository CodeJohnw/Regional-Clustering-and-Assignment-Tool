# Regional Clustering and Assignment Tool

This Python script uses several libraries to perform geospatial analysis and clustering of regions based on their centroids. It clusters the regions into specified numbers of clusters and assigns each region to a cluster center based on minimizing the euclidean distance. It also ensures that the areas assigned to each cluster center are balanced.

## Features

- **Geospatial Data Handling**: Load and manipulate geographical data from shapefiles.
- **Clustering**: Cluster geographical regions based on the location of their centroids.
- **Optimal Assignment**: Assign each region to a cluster center while balancing the total area assigned to each center.
- **Visualization**: Visualize the clustering results on a map.

## Installation

To run this script, you need to install the required Python libraries. You can install them using pip:

```bash
pip install scikit-learn pulp scipy geopandas numpy pandas matplotlib
```

## Usage

1. **Data Preparation**:
   - Ensure you have a `.shp` file containing the geographical data of the regions.
   - Update the `shp_path` variable in the script to the path of your `.shp` file.

2. **Configuration**:
   - Set the number of clusters by adjusting the `n` variable.
   - Configure other parameters as needed (e.g., tolerance for area assignment).

3. **Execution**:
   - Run the script to perform clustering and see the results plotted on a map.
   - The script will output details such as the status of the optimization, total distance, and area assignments for each cluster center.

## Code Structure

- **Loading Data**: Geographical data from shapefiles are loaded and processed.
- **Clustering Setup**: Initial centroids are defined, and KMeans clustering is applied.
- **Optimization Problem Setup**: Defines and solves the linear programming problem to optimally assign regions to clusters.
- **Visualization**: Plots the final clustering results on a map.

## Outputs

- The script outputs:
  - Status of the optimization problem (whether it solved successfully).
  - Total distance which represents the sum of distances for all assignments.
  - Area assignments for each cluster center.
  - Visualization of the clustering result on a map.

## Contributing

Feel free to fork the repository and submit pull requests. You can also open issues for bugs or feature requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

This tool utilizes powerful libraries such as `pulp` for optimization, `scikit-learn` for clustering, and `geopandas` for geospatial data handling, providing a comprehensive approach to region clustering and assignment.

---

Ensure all data paths and specific configurations match your setup requirements. Happy clustering!

