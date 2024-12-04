import pandas as pd
import matplotlib.pyplot as plt

labels = ['NOx(GT)', 'NO2(GT)', 'CO(GT)']

for label in labels:
    # Load the data from the uploaded files
    file_paths = {
        "Random Forest": f"./model/long-mix/random_forest_Long-term_{label}_prediction_data.csv",
        "BiGRU": f"./model/long-mix/BIGRU_Long-term_{label}_prediction_data.csv",
        "Linear Regression": f"./model/long-mix/linear_regression_Long-term_{label}_prediction_data.csv",
        "LSTM": f"./model/long-mix/LSTM_Long-term_{label}_prediction_data.csv",
    }

    dataframes = {name: pd.read_csv(path) for name, path in file_paths.items()}

    # Preview one of the datasets to understand its structure
    dataframes["Random Forest"].head()

    # Check the structure of other datasets to ensure consistency
    for name, df in dataframes.items():
        print(f"{name} Dataset:")
        print(df.head(), "\n")


    # Define line styles and colors for clarity
    styles = {
        "Random Forest": "--",
        "BiGRU": "-.",
        "Linear Regression": ":",
        "LSTM": "--"
    }


    # Adjust the plotting code to ensure true values are correctly displayed
    plt.figure(figsize=(12, 6))

    # Define a color palette and marker styles for the algorithms
    colors = {
        "Random Forest": "blue",
        "BiGRU": "green",
        "Linear Regression": "orange",
        "LSTM": "red"
    }

    markers = {
        "Random Forest": "x",
        "BiGRU": "^",
        "Linear Regression": "s",
        "LSTM": "*"
    }

    # Plot the true values for all samples (only once)
    for sample in dataframes["Random Forest"]["Sample"].unique():
        sample_data = dataframes["Random Forest"][dataframes["Random Forest"]["Sample"] == sample]
        plt.plot(
            sample_data["Hour_Ahead"] + (sample - 1) * 10,  # Shift x-axis to separate samples
            sample_data["True_Value"],
            linestyle="-",
            marker="o",
            markersize=5,
            alpha=0.5,  # Set transparency
            color="black",
            label="True Sample" if sample == 1 else None,  # Only label the first sample
        )

    # Iterate through the datasets to plot predicted values for each algorithm
    for name, df in dataframes.items():
        unique_samples = df["Sample"].unique()
        for sample in unique_samples:
            sample_data = df[df["Sample"] == sample]
            plt.plot(
                sample_data["Hour_Ahead"] + (sample - 1) * 10,  # Shift x-axis to separate samples
                sample_data["Predicted_Value"],
                linestyle=styles[name],
                marker=markers[name],
                markersize=6,
                alpha=0.5,  # Set transparency
                color=colors[name],
                label=name if sample == unique_samples[0] else None,  # Only label the first sample
            )

    # Adding labels, title, legend
    plt.xlabel("Samples (Shifted Hour Ahead)")
    plt.ylabel(f"{label} Concentration")
    plt.title("Comparison of Predictions Across Samples and Algorithms")
    plt.legend(ncol=1, loc='upper right', frameon=True)
    plt.grid(True)

    # Display the plot
    plt.tight_layout()

    # Save the plot as an image file
    plt.savefig(f"./model/long-mix/{label}_prediction_comparison.png")
    # plt.show()
    plt.close()
