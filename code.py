import json
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, Trainer, TrainingArguments, pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from datasets import Dataset


class FinancialQAModel:
    def __init__(self, model_name="deepset/roberta-base-squad2"):
        """
        Initialize the Hugging Face pipeline for question answering.
        """
        print(f"Initializing model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        self.qa_pipeline = pipeline("question-answering", model=self.model, tokenizer=self.tokenizer)

    def load_dataset(self, filepath, sample_size=None):
        """
        Load the ConvFinQA dataset from a JSON file and optionally subsample it.
        """
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            print(f"Total entries in dataset: {len(data)}")

            # Filter out entries without the 'qa' field
            data = [entry for entry in data if 'qa' in entry]

            # Subsample the dataset if sample_size is provided
            if sample_size:
                from random import sample
                data = sample(data, min(sample_size, len(data)))

            # Extract questions, answers, and contexts
            questions = [entry['qa']['question'] for entry in data]
            answers = [entry['qa']['answer'] for entry in data]
            contexts = [' '.join(entry.get('pre_text', []) + entry.get('post_text', [])) for entry in data]

            print(f"Loaded {len(questions)} valid questions (sampled).")
            return questions, answers, contexts
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return [], [], []

    def preprocess_data(self, questions, answers, contexts):
        """
        Preprocess the data for model input by tokenizing and aligning answers.
        """
        # Tokenize questions and contexts
        encodings = self.tokenizer(
            questions,
            contexts,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt"
        )

        start_positions = []
        end_positions = []

        # Align answers to token positions
        for i in range(len(questions)):
            answer = answers[i]
            context = contexts[i]
            start_idx = context.find(answer)
            end_idx = start_idx + len(answer) - 1

            # If answer is found, compute start and end positions
            if start_idx != -1:
                start_positions.append(start_idx)
                end_positions.append(end_idx)
            else:
                # If not found, assign padding values
                start_positions.append(0)
                end_positions.append(0)

        encodings.update({
            'start_positions': torch.tensor(start_positions),
            'end_positions': torch.tensor(end_positions)
        })
        return encodings

    def fine_tune(self, questions, answers, contexts, batch_size=16, epochs=1):
        """
        Fine-tune the model using the provided dataset.
        """
        # Preprocess data
        encodings = self.preprocess_data(questions, answers, contexts)

        # Convert to Hugging Face Dataset
        train_dataset = Dataset.from_dict(encodings)

        # Set training arguments
        training_args = TrainingArguments(
            output_dir='./results',  # Output directory
            evaluation_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=batch_size,
            num_train_epochs=epochs,
            weight_decay=0.01,
            logging_dir='./logs',  # Directory for storing logs
        )

        # Initialize Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=train_dataset,
        )

        # Train the model
        trainer.train()

        # Save the fine-tuned model
        self.model.save_pretrained('./fine_tuned_model')
        self.tokenizer.save_pretrained('./fine_tuned_model')
        print("Model fine-tuned and saved successfully!")

    def evaluate_model(self, questions, answers, contexts, batch_size=16):
        """
        Evaluate the QA model on the dataset and compute metrics.
        """
        normalized_answers = [self._normalize_answer(ans) for ans in answers]
        X_train, X_test, y_train, y_test = train_test_split(
            list(zip(questions, contexts)),
            normalized_answers,
            test_size=0.2,
            random_state=42
        )

        predictions = []
        for i in range(0, len(X_test), batch_size):
            batch = X_test[i:i+batch_size]
            batch_predictions = []
            for question, context in batch:
                try:
                    result = self.qa_pipeline({"question": question, "context": context})
                    normalized_pred = self._normalize_answer(result.get('answer', ''))
                    batch_predictions.append(normalized_pred)
                except Exception as e:
                    print(f"Prediction error: {e}")
                    batch_predictions.append(None)
            predictions.extend(batch_predictions)

        # Filter out invalid predictions
        valid_indices = [i for i, pred in enumerate(predictions) if pred is not None]
        valid_predictions = [predictions[i] for i in valid_indices]
        valid_true_answers = [y_test[i] for i in valid_indices]

        # Compute metrics
        metrics = self._compute_metrics(valid_true_answers, valid_predictions)
        return metrics

    def _compute_metrics(self, true_answers, predicted_answers):
        """
        Compute evaluation metrics for the QA task.
        """
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_answers, predicted_answers, average='weighted', zero_division=0
        )
        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }

    def _normalize_answer(self, answer):
        """
        Normalize an answer by removing punctuation, whitespace, and articles.
        """
        import re
        import string
        answer = answer.lower()
        answer = re.sub(f"[{string.punctuation}]", "", answer)
        answer = re.sub(r"\s+", " ", answer).strip()
        return answer

    def generate_report(self, metrics):
        """
        Generate a performance report.
        """
        report = "# Financial QA Model Performance Report\n\n"
        report += "## Evaluation Metrics\n\n"
        for metric, value in metrics.items():
            report += f"- **{metric.capitalize()}**: {value:.2%}\n"
        report += "\n## Model report\n"
        report += "- The model's performance is summarized above.\n"
        report += "- Fine-tuning is needed on domain-specific data for better results.\n"
        return report


def main():
    filepath = '/content/train.json' 
    model = FinancialQAModel()

    # Load a smaller dataset with a sample size of 100
    sample_size = 100
    questions, answers, contexts = model.load_dataset(filepath, sample_size=sample_size)

    # Fine-tune the model for 1 epoch
    model.fine_tune(questions, answers, contexts, batch_size=4, epochs=1)

    # Evaluate the fine-tuned model
    metrics = model.evaluate_model(questions, answers, contexts, batch_size=4)

    # Generate and print the report
    report = model.generate_report(metrics)
    print(report)

    # Save the report to a file
    with open('financial_qa_report.md', 'w') as f:
        f.write(report)


if __name__ == "__main__":
    main()
