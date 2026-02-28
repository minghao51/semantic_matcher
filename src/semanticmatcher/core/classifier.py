from typing import Optional, Union, List
import numpy as np
from datasets import Dataset

try:
    from setfit import SetFitModel, Trainer, TrainingArguments

    SETFIT_AVAILABLE = True
except ImportError:
    SETFIT_AVAILABLE = False


class SetFitClassifier:
    """Wrapper for SetFit training and prediction."""

    def __init__(
        self,
        labels: List[str],
        model_name: str = "sentence-transformers/paraphrase-mpnet-base-v2",
        num_epochs: int = 4,
        batch_size: int = 16,
    ):
        if not SETFIT_AVAILABLE:
            raise ImportError("setfit is required. Install with: pip install setfit")

        self.labels = labels
        self.model_name = model_name
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.model: Optional[SetFitModel] = None
        self.is_trained = False

    def train(
        self,
        training_data: List[dict],
        num_epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
    ):
        epochs = num_epochs or self.num_epochs
        batch = batch_size or self.batch_size

        self.model = SetFitModel.from_pretrained(self.model_name, labels=self.labels)

        dataset = Dataset.from_list(training_data)

        args = TrainingArguments(
            num_epochs=epochs,
            batch_size=batch,
            body_learning_rate=2e-5,
            head_learning_rate=1e-3,
        )

        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=dataset,
        )

        trainer.train()
        self.is_trained = True

    def predict(self, texts: Union[str, List[str]]) -> Union[str, List[str]]:
        if not self.is_trained or self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")

        if isinstance(texts, str):
            texts = [texts]

        predictions = self.model.predict(texts)

        if len(predictions) == 1:
            return predictions[0]
        return predictions.tolist()

    def predict_proba(self, text: str) -> np.ndarray:
        if not self.is_trained or self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")
        probs = self.model.predict_proba([text])
        return np.asarray(probs)[0]

    def save(self, path: str):
        if not self.is_trained or self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")
        self.model.save_pretrained(path)

    @classmethod
    def load(cls, path: str) -> "SetFitClassifier":
        model = SetFitModel.from_pretrained(path)
        clf = cls(labels=model.labels)
        clf.model = model
        clf.is_trained = True
        return clf
