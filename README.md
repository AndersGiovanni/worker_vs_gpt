# Worker vs. GPT

## Features

The project contain code and functionality to do the following experiments.

- **Zero-shot classification** using LLMs.
- **Data augmentation** using LLMs.
- **Datasize experiment** using progressively larger samplesize in training.
- **Traditional LM training**.
- **Few-shot learning** with contrastive pre-training using the [SetFit](https://github.com/huggingface/setfit) framework.

We use the OpenAI API to interact with LLMs. We use ChatGPT and GPT-4. 

## Requirements

- Latest versions of [poetry](https://python-poetry.org/) installed.
- OpenAI API key. Make sure to put it in `.env` file.
- A [Weights & Bias](https://wandb.ai/) account for performance reporting.

## Installation

You can install the environment using:

```bash
# Create the environment
$ poetry shell

# Update dependencies
$ poetry update

# Install project
$ poetry install
```

## Weights & Bias Activation

- Login to your W&B account: `$ wandb loging`
- Enable tracking of experiments: `$ wandb enabled`
- Disable tracking of experiments (for debugging): `$ wandb disabled`

## Usage

### Data Augmentation

In `src/worker_vs_gpt/conf/config_prompt_augmentation.yaml` you find the configuration file with variables to change:

```yaml
model: vicuna # can be gpt-3.5-turbo or gpt-4
dataset: analyse-tal # can be hate-speech, sentiment, ten-dim
sampling: balanced # can be proportional or balanced
```

Next, execute the script: `python -m src.worker_vs_gpt.prompt_augmentation`

### Zero-shot Classification

In `src/worker_vs_gpt/conf/config_prompt_classification.yaml` you find the configuration file with variables to change:

```yaml
model: gpt-4 # can be gpt-3.5-turbo or gpt-4
dataset: sentiment # can be hate-speech, sentiment, ten-dim
wandb_project: W&B_project_name
wandb_entity: W&B_account_name
```

Next, execute the script: `python -m src.worker_vs_gpt.prompt_classification`

### Datasize Experiment

In `src/worker_vs_gpt/conf/config_datasize.yaml` you find the configuration file with variables to change:

```yaml
ckpt: intfloat/e5-base # The model you want to use from the Hugginface model hub
dataset: ten-dim # can be 'hate-speech', 'sentiment', 'ten-dim'
use_augmented_data: True # Whether or not to use augmented data
sampling: proportional # can be proportional or balanced
augmentation_model: gpt-3.5-turbo # can be gpt-3.5-turbo or gpt-4
wandb_project: W&B_project_name
wandb_entity: W&B_account
batch_size: 32 # batch size
lr: 2e-5 # learning rate
num_epochs: 10 # number of epochs
weight_decay: 0 # weight decay
```

Next, execute the script: `python -m src.worker_vs_gpt.datasize_experiment`

### Normal LM Training

In `src/worker_vs_gpt/conf/config_trainer.yaml` you find the configuration file with variables to change:

```yaml
ckpt: intfloat/e5-base # The model you want to use from the Hugginface model hub
dataset: ten-dim # can be 'hate-speech', 'sentiment', 'ten-dim'
use_augmented_data: True # Whether or not to use augmented data
sampling: proportional # can be proportional or balanced
augmentation_model: gpt-3.5-turbo # can be gpt-3.5-turbo or gpt-4
experiment_type: both # can be crowdsourced (only crowdsourced), aug (only augmented data), both (crowdsourced and augmented data concatenated)
wandb_project: W&B_project_name
wandb_entity: W&B_account
batch_size: 32 # batch size
lr: 2e-5 # learning rate
num_epochs: 10 # number of epochs
weight_decay: 0 # weight decay
```

Next, execute the script: `python -m src.worker_vs_gpt.__main__`

### Few-shot with contrastive pre-training using SetFit

In `src/worker_vs_gpt/conf/setfit.yaml` you find the configuration file with variables to change:

```yaml
ckpt: intfloat/e5-base # The model you want to use from the Hugginface model hub
text_selection: h_text # this is for social-dim dataset, don't change
experiment_type: aug # can be 'crowdsourced', 'aug', 'both'
sampling: balanced # can be proportional or balanced
augmentation_model: gpt-3.5-turbo # can be gpt-3.5-turbo or gpt-4
dataset: hate-speech # can be 'hate-speech', 'sentiment', 'ten-dim'
batch_size: 8 # Batch size
lr_body: 1e-5 # Learning rate for the contrastive pre-training of the model body.
lr_head: 1e-5 # Learning rate for the classification head
num_iterations: 20 # Parameter to construct pais in pre-training. They use in the paper (20)
num_epochs_body: 1 # How many epochs to do contrastive pre-training. 1 is used in paper.
num_epochs_head: 20 # How many iterations to train the head for. In their tutorial they use 50. Not clear how many they use in training.
weight_decay: 0 # weight decay
wandb_project: W&B_project_name
wandb_entity: W&B_account

```

Next, execute the script: `python -m src.worker_vs_gpt.setfit_classfication`

## License

Distributed under the terms of the [MIT license][license],
_Worker vs. GPT_ is free and open source software.

## Credits

This project was generated from [@cjolowicz]'s [Hypermodern Python Cookiecutter] template.

[@cjolowicz]: https://github.com/cjolowicz
[pypi]: https://pypi.org/
[hypermodern python cookiecutter]: https://github.com/cjolowicz/cookiecutter-hypermodern-python
[license]: https://github.com/AGMoller/worker_vs_gpt/blob/main/LICENSE