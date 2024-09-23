from transformers import AutoImageProcessor, ResNetForImageClassification
import torch
from datasets import load_dataset
from transformers import Trainer, TrainingArguments
import evaluate
import numpy as np

# load dataset
dataset = load_dataset("mnist", trust_remote_code=True)

# load model and preprocessor
# reshaping of the images will be handled by `image_processor`
image_processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")


def preprocess_function(examples):
    images = [img.convert('RGB') for img in examples['image']]
    return image_processor(images=images, return_tensors="pt")
# process the dataset
encoded_dataset = dataset['test'].map(preprocess_function, batched=True)

# define the eval metric
metric = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# define the TrainingArguments for class `Trainer`
training_args = TrainingArguments(
    output_dir="./results",
    per_device_eval_batch_size=64,
    num_train_epochs=0,
    evaluation_strategy="epoch",
)

# define trainer with no training process
trainer = Trainer(
    model=model,
    args=training_args,
    compute_metrics=compute_metrics, 
    eval_dataset=encoded_dataset,
)

# eval and print the result!
eval_results = trainer.evaluate()
print(eval_results)