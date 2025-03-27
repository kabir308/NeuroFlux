from datasets import Dataset
import pandas as pd

def load_dataset():
    """
    Load and return the dataset.
    """
    # Load CSV data
    df = pd.read_csv('data/models.csv')
    
    # Convert to Dataset
    dataset = Dataset.from_pandas(df)
    
    return dataset

def main():
    """
    Main function to load and display the dataset.
    """
    dataset = load_dataset()
    print("\nDataset Overview:")
    print(dataset)
    
    print("\nDataset Features:")
    print(dataset.features)
    
    print("\nFirst Example:")
    print(dataset[0])

if __name__ == "__main__":
    main()
