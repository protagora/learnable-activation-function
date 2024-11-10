import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

def create_random_set(cardinality=10, min=0, max=1, type='int', distribution='norm'):
    """
    Function to create an array of random values of predefined size, range, distribution type, and values type.

    :param cardinality (int): Number of elements in the set.
    :param min (number): Minimum value of the set.
    :param max (number): Maximum value of the set.
    :param type (str): Type of the value of the set (int, float).
    :param distribution (str): Name of the distribution to draw values from.

    :returns: numpy array of generated values.

    :raises ValueError: If parameters are out of logical bounds or unsupported.
    """
    
    # Validate cardinality
    if not isinstance(cardinality, int) or cardinality <= 0:
        raise ValueError("Cardinality must be a positive integer")
    
    # Validate min and max
    if not (isinstance(min, (int, float)) and isinstance(max, (int, float))):
        raise ValueError("min and max must be numbers")
    if min >= max:
        raise ValueError("min must be less than max")
    
    # Validate type
    if type not in ['int', 'float']:
        raise ValueError("type must be 'int' or 'float'")

    # Check if the distribution is available in scipy.stats
    if hasattr(stats, distribution):
        dist = getattr(stats, distribution)
    else:
        raise ValueError(f"Unsupported distribution type: {distribution}")

    # Set parameters for the distribution
    if distribution in ['uniform']:
        data = dist(loc=min, scale=max-min).rvs(size=cardinality)
    else:
        # For most distributions, setting loc and scale this way makes sense.
        data = dist(loc=(max+min)/2, scale=(max-min)/6).rvs(size=cardinality)

    # Cast to int if required
    if type == 'int':
        data = np.rint(data).astype(int)

    return data


def assign_consecutive_integers(data):
    """
    Assigns consecutive integers to sorted unique values in the data based on the last occurrence of each value.

    :param data: numpy array of integer data
    :returns: A tuple of (bin_edges, assigned_integers) where `bin_edges` are the distinct sorted values
              and `assigned_integers` is the list of consecutive integers assigned to the last occurrence of these values.
    """
    # Sort the data and find unique values along with their inverse indices and counts
    data_sorted = np.sort(data)
    unique_values, inverse_indices, counts = np.unique(data_sorted, return_inverse=True, return_counts=True)
    
    # Calculate the last index of each unique value
    last_indices = np.cumsum(counts) - 1  # Subtracting 1 because indices are zero-based

    # Map last_indices to the corresponding consecutive integers
    assigned_integers = {}
    for index, value in enumerate(unique_values):
        assigned_integers[value] = last_indices[index]

    # Extracting consecutive integers as an array
    consecutive_integers_array = np.array([assigned_integers[val] for val in unique_values])

    return unique_values, consecutive_integers_array


def normalize_histogram_values(data):
    """
    Normalize the cumulative counts in histogram data.
    
    :param data: A tuple where the first element is an array of bin edges and the second element is an array of cumulative counts.
    :returns: A tuple where the first element remains the bin edges and the second element is the normalized cumulative counts.
    """
    bin_edges, values = data  # Unpack the tuple
    min_val = min(values)
    max_val = max(values)
    
    if max_val == min_val:
        normalized_values = [0.0] * len(values)  # All values are the same, avoid division by zero
    else:
        normalized_values = [(float(val) - min_val) / (max_val - min_val) for val in values]

    return bin_edges, normalized_values


def create_cumulative_histogram(data, num_bins=None):
    """
    Creates a cumulative histogram where the number of bins equals the difference between the max and min values of the data.
    
    :param data: numpy array of integer data
    :returns: A tuple of (bin_edges, cumulative_histogram) where `bin_edges` are the edges of the bins and 
              `cumulative_histogram` is the cumulative sum of the histogram bins.
    """
    # Calculate the number of bins as the difference between the max and min values plus one for inclusive bin
    if num_bins is None:
        num_bins = np.max(data) - np.min(data) + 1

    # Calculate histogram
    histogram, bin_edges = np.histogram(data, bins=num_bins, range=(np.min(data), np.max(data)+1))
    
    # Calculate cumulative sum of the histogram
    cumulative_histogram = np.cumsum(histogram)
    
    return bin_edges, cumulative_histogram


def plot_cumulative_histogram(bin_edges, cumulative_histogram, title="Cumulative Histogram", filename="cumulative_histogram.png"):
    """
    Plots and saves a cumulative histogram.
    
    :param bin_edges: Edges of the histogram bins.
    :param cumulative_histogram: Cumulative sum of the histogram bins.
    :param title: Title of the histogram plot.
    :param filename: Filename to save the plot as a PNG image.
    """
    # Adjust bin_edges to align with the length of cumulative_histogram, if necessary
    if len(bin_edges) == len(cumulative_histogram) + 1:
        bin_edges = bin_edges[:-1]  # Correct common off-by-one error in histogram plotting

    # Plotting the cumulative histogram
    plt.figure(figsize=(10, 6))
    plt.plot(bin_edges, cumulative_histogram, drawstyle="steps-post")
    plt.title(title)
    plt.xlabel("Bin Edges")
    plt.ylabel("Cumulative Count")
    plt.grid(True)

    # Saving the plot
    plt.savefig(filename)
    plt.close()


def plot_dual_cumulative_histogram(bin_edges1, cumulative_histogram1, bin_edges2, cumulative_histogram2, title="Dual Cumulative Histogram", filename="dual_cumulative_histogram.png"):
    """
    Plots and saves a figure with two cumulative histograms on the same plot.
    
    :param bin_edges1: Edges of the first histogram bins.
    :param cumulative_histogram1: Cumulative sum of the first histogram bins.
    :param bin_edges2: Edges of the second histogram bins.
    :param cumulative_histogram2: Cumulative sum of the second histogram bins.
    :param title: Title of the histogram plot.
    :param filename: Filename to save the plot as a PNG image.
    """
    plt.figure(figsize=(10, 6))

    # Adjust bin_edges to align with the length of cumulative_histogram
    if len(bin_edges1) == len(cumulative_histogram1) + 1:
        bin_edges1 = bin_edges1[:-1]
    if len(bin_edges2) == len(cumulative_histogram2) + 1:
        bin_edges2 = bin_edges2[:-1]

    # Plotting the cumulative histograms
    plt.plot(bin_edges1, cumulative_histogram1, drawstyle="steps-post", label='Ordered Histogram')
    plt.plot(bin_edges2, cumulative_histogram2, drawstyle="steps-post", label='Regular Histogram')
    plt.title(title)
    plt.xlabel("Bin Edges")
    plt.ylabel("Cumulative Count")
    plt.legend(loc="upper left")
    plt.grid(True)

    # Saving the plot
    plt.savefig(filename)
    plt.close()


if "__main__" == __name__:

    debug = True
    random_set = create_random_set(
        cardinality = 100,
        min = -1.0,
        max = 1.0,
        type='int',
        distribution='norm',
    )

    if debug is True:
        print(random_set)

    ordered_histogram = assign_consecutive_integers(random_set)
    cumulative_histogram = create_cumulative_histogram(random_set, num_bins=10)

    ordered_histogram = normalize_histogram_values(ordered_histogram)
    cumulative_histogram = normalize_histogram_values(cumulative_histogram)



    if debug is True:
        from pprint import pprint
        pprint(ordered_histogram)
        pprint(cumulative_histogram)

    # # Plot and save the ordered histogram
    # plot_cumulative_histogram(*ordered_histogram, title="Ordered Cumulative Histogram", filename="ordered_cumulative_histogram.png")

    # # Plot and save the regular cumulative histogram
    # plot_cumulative_histogram(*cumulative_histogram, title="Regular Cumulative Histogram", filename="regular_cumulative_histogram.png")

    plot_dual_cumulative_histogram(*ordered_histogram, *cumulative_histogram, title="Comparison of Cumulative Histograms", filename="comparison_cumulative_histograms.png")
