import os
import csv
import matplotlib.pyplot as plt

# Function to read CSV file
def read_csv(file_path):
    data = {'episode': [], 'moving_average': []}
    with open(file_path, 'r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            data['episode'].append(int(row['episode']))
            data['moving_average'].append(float(row[' moving_average']))
            # data['moving_average'].append(float(row[' score ']))
    return data

# Function to plot data from multiple files with different colors
def plot_multiple_files(file_paths):
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']  # You can add more colors if needed
    for i, file_path in enumerate(file_paths):
        data = read_csv(file_path)
        plt.plot(data['episode'], data['moving_average'], label=f'Worker_{i+1}', color=colors[i % len(colors)])

# Directory containing CSV files (replace 'your_directory_path' with the actual path)
directory_path = r'C:\tmp\a2c'
#directory_path = r'C:\tmp\ac2-almost-finished'
#directory_path = r'C:\tmp\a2c-140000'
# directory_path = r'C:\tmp\a2c – kopio'
file_paths = [os.path.join(directory_path, f'worker_{i}.txt') for i in range(0, 6) if os.path.exists(os.path.join(directory_path, f'worker_{i}.txt'))]  # Adjust the range as needed

# Plot data from multiple files
plot_multiple_files(file_paths)

# Add legend, title, and labels
plt.legend()
plt.title('Moving Average Over Episodes - Multiple Workers')
plt.xlabel('Episode')
plt.ylabel('Moving Average')
plt.grid(True)
plt.show()