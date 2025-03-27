import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset

class EmotionDetectorTrainer:
    def __init__(self):
        """
        Initialize the Emotion Detector trainer.
        """
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertForSequenceClassification.from_pretrained(
            'bert-base-uncased',
            num_labels=8,  # Example: 8 emotions (happy, sad, angry, etc.)
            output_attentions=False,
            output_hidden_states=False
        )
        
    def prepare_dataset(self):
        """
        Prepare and preprocess the dataset.
        """
        # Load a sample dataset (replace with your actual emotion dataset)
        dataset = load_dataset('emotion')  # Example dataset
        
        # Tokenize the dataset
        def tokenize_function(examples):
            return self.tokenizer(
                examples['text'],
                padding='max_length',
                truncation=True,
                max_length=128
            )
        
        tokenized_datasets = dataset.map(tokenize_function, batched=True)
        tokenized_datasets = tokenized_datasets.remove_columns(['text'])
        tokenized_datasets = tokenized_datasets.rename_column('label', 'labels')
        
        return tokenized_datasets
        
    def train(self):
        """
        Train the Emotion Detector model.
        """
        # Prepare the dataset
        train_dataset = self.prepare_dataset()['train']
        
        # Define training arguments
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=5,
            per_device_train_batch_size=32,
            per_device_eval_batch_size=32,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=10,
            evaluation_strategy='epoch',
            save_strategy='epoch',
            save_total_limit=2,
            learning_rate=2e-5
        )
        
        # Initialize the Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=train_dataset
        )
        
        # Train the model
        trainer.train()
        
        # Save the model
        self.model.save_pretrained('./emotion-detector')
        self.tokenizer.save_pretrained('./emotion-detector')

if __name__ == "__main__":
    trainer = EmotionDetectorTrainer()
    trainer.train()
