# LLM Self Evaluation

This folder contains the code to evaluate the LLM model on the self-evaluation task. The code is part of my master level research project with Luca Maria Aiello, "Anders Giovanni Møller" <agmo@itu.dk>; "Arianna Pera" <arpe@itu.dk>; "Viktor Due Pedersen" <vipe@itu.dk> as supervisors.

- [LLM Self Evaluation](#llm-self-evaluation)
  - [The task at hand](#the-task-at-hand)
    - [Description of the ten emotions](#description-of-the-ten-emotions)
  - [Evaluation subset](#evaluation-subset)
    - [Label to other label](#label-to-other-label)
    - [Within label aumentation not from original](#within-label-aumentation-not-from-original)
    - [Within label Original to Original](#within-label-original-to-original)
  - [To do](#to-do)

## The task at hand

Through out the rest of this repository you can find the code to create the dataset that is the foundation of this task. The dataset `data/ten-dim/balanced_gpt-4_augmented_full.json` contain curated data of pairs of texts. Each entry contain a:

- h_text: The *source* text belonging to the `target`.
- `target`: The emotion label of the `h_text`.
- augmented_h_text: A LLM received `h_text` and the given `target` as prompt and was asked to generate texts containing the same emotion.

### Description of the ten emotions

- knowledge: Exchange of ideas or information,
- power: Having power over the behavior and outcomes of another,
- respect: Conferring status, appreciation, gratitude, or admiration upon another,
- trust: Will of relying on the actions or judgments of another,
- social_support: Giving emotional or practical aid and companionship,
- similarity_identity: Shared interests, motivations, outlooks or Shared sense of belonging to the same community or group,
- fun: Experiencing leisure, laughter, and joy,
- conflict: Contrast or diverging views,
- neutral: neutral communication

## Evaluation subset

The counts of each `target` in `data/ten-dim/balanced_gpt-4_augmented_full.json` are as so:

| target              | Unique source texts | Total Augmented texts | Augmentations per source text |
| ------------------- | ------------------- | --------------------- | ----------------------------- |
| similarity_identity | 17                  | 550                   | 32,35                         |
| neutral             | 59                  | 590                   | 10                            |
| conflict            | 56                  | 560                   | 10                            |
| social_support      | 55                  | 550                   | 10                            |
| respect             | 30                  | 550                   | 18,33                         |
| knowledge           | 55                  | 550                   | 10                            |
| fun                 | 13                  | 550                   | 42,3                          |
| power               | 4                   | 550                   | 137,5                         |
| trust               | 13                  | 550                   | 42,3                          |

Clearly, some of the emotions are underrepresented, but have the same amount of augmented texts.

This is too many for an efficient evaluation of each emotion since I want to evaluate the texts in different ways. Therefore, I have created a subset of the data, in  `src/worker_vs_gpt/evaluation/subsets/` containing three different subsets. Explanation of each subset is as follows:

Creating the subsets is done by running

```bash
python src/worker_vs_gpt/evaluation/generate_datasets.py
```

### Label to other label

### Within label aumentation not from original

### Within label Original to Original

## To do
