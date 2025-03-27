import pandas as pd
import os

def generate_models_csv():
    """
    Generate a CSV file containing model information.
    """
    # Create data
    data = {
        'model_name': ['tinybert', 'mobilenet', 'emotion-detector'],
        'model_type': ['bert', 'mobilenet', 'custom'],
        'pipeline_tag': ['text-classification', 'image-classification', 'text-classification'],
        'model_size': [10, 5, 3],
        'description': [
            'Lightweight version of BERT for microscopic AI applications',
            'Lightweight version of MobileNet for microscopic AI applications',
            'Specialized model for detecting emotions in text and voice'
        ],
        'target_devices': [
            'microcontrollers, IoT devices',
            'microcontrollers, IoT devices',
            'microcontrollers, IoT devices'
        ],
        'inference_time': ['~10ms', '~5ms', '~2ms'],
        'memory_usage': ['~2MB RAM', '~1MB RAM', '~500KB RAM'],
        'accuracy': ['90%', '85%', '88%']
    }
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Save to CSV
    csv_path = os.path.join('data', 'models.csv')
    df.to_csv(csv_path, index=False)
    print(f"Dataset saved to {csv_path}")

if __name__ == "__main__":
    generate_models_csv()
