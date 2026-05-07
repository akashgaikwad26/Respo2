---
license: other
license_name: health-ai-developer-foundations
license_link: https://developers.google.com/health-ai-developer-foundations/terms
language:
  - en
tags:
  - medical
  - medical-embeddings
  - audio
  - health-acoustic
extra_gated_heading: Access HeAR on Hugging Face
extra_gated_prompt: >-
  To access HeAR on Hugging Face, you're required to review and
  agree to [Health AI Developer Foundation's terms of use](https://developers.google.com/health-ai-developer-foundations/terms).
  To do this, please ensure you're logged in to Hugging Face and click below.
  Requests are processed immediately.
extra_gated_button_content: Acknowledge license
---
# HeAR model card

**Model documentation:** [HeAR](https://developers.google.com/health-ai-developer-foundations/hear)

**Resources**:

*   Model on Google Cloud Model Garden: [HeAR](https://console.cloud.google.com/vertex-ai/publishers/google/model-garden/hear)

*   Model on Hugging Face: [google/hear](https://huggingface.co/google/hear)

*   GitHub repository (supporting code, Colab notebooks, discussions, and
    issues): [HeAR](https://github.com/google-health/hear)

*   Quick start notebook: [notebooks/quick\_start](https://github.com/google-health/hear/blob/master/notebooks/quick_start_with_hugging_face.ipynb)

*   Support: See
    [Contact](https://developers.google.com/health-ai-developer-foundations/hear/get-started.md#contact).

Terms of use: [Health AI Developer Foundations terms of
use](https://developers.google.com/health-ai-developer-foundations/terms)

**Author**: Google

## Model information

This section describes the HeAR model and how to use it.

### Description

Health-related acoustic cues, originating from the respiratory system's airflow,
including sounds like coughs and breathing patterns can be harnessed for health
monitoring purposes. Such health sounds can also be collected via ambient
sensing technologies on ubiquitous devices such as mobile phones, which may
augment screening capabilities and inform clinical decision making. Health
acoustics, specifically non-semantic respiratory sounds, also have potential as
biomarkers to detect and monitor various health conditions, for example,
identifying disease status from cough sounds, or measuring lung function using
exhalation sounds made during spirometry.

Health Acoustic Representations, or HeAR, is a health acoustic foundation model
that is pre trained to efficiently represent these non-semantic respiratory
sounds to accelerate research and development of AI models that use these inputs
to make predictions. HeAR is trained unsupervised on a large and diverse
unlabelled corpus, which may generalize better than non-pretrained models to
unseen distributions and new tasks.

Key Features

*   Generates health-optimized embeddings for biological sounds such as coughs
    and breathes

*   Versatility: Exhibits strong performance across diverse health acoustic
    tasks.

*   Data Efficiency: Demonstrates high performance even with limited labeled
    training data for downstream tasks.

*   Microphone robustness: Downstream models trained using HeAR generalize
    well to sounds recorded from unseen devices.

Potential Applications

HeAR can be a useful tool for AI research geared towards
discovery of novel acoustic biomarkers in the following areas:

*   Aid screening & monitoring for respiratory diseases like COVID-19,
    tuberculosis, and COPD from cough and breath sounds.

*   Low-resource settings: Can potentially augment healthcare services in
    settings with limited resources by offering accessible screening and
    monitoring tools.

### How to use

Below are some example code snippets to help you quickly get started running the
model locally. If you want to use the model to run inference on a large amount
of audio, we recommend that you create a production version using [the Vertex
Model
Garden](https://console.cloud.google.com/vertex-ai/publishers/google/model-garden/hear).

```python
import numpy as np
from huggingface_hub.utils import HfFolder
from huggingface_hub import notebook_login, from_pretrained_keras, notebook_login
if HfFolder.get_token() is None:
   notebook_login()


# Load the model from Hugging Face
model = from_pretrained_keras("google/hear",)
serving_signature = model.signatures['serving_default']


# Generate 4 Examples of two-second random audio clips
raw_audio_batch = np.random.normal(size=(4, 32000))


# Perform Inference to obtain HeAR embeddings
# There are 4 embeddings each with length 512 corresponding to the 4 inputs
embedding_batch = serving_signature(x=raw_audio_batch)['output_0'].numpy()
```

### Examples

See the following Colab notebooks for examples of how to use HeAR:

*   To give the model a quick try, running it locally with weights from Hugging
    Face, see [Quick start notebook in
    Colab](https://colab.research.google.com/github/google-health/hear/blob/master/notebooks/quick_start_with_hugging_face.ipynb).

*   For an example of how to use the model to train a linear classifier, see
    [Linear classifier notebook in
    Colab](https://colab.research.google.com/github/google-health/hear/blob/master/notebooks/train_data_efficient_classifier.ipynb).

### Model architecture overview

HeAR is a [Masked Auto Encoder](https://arxiv.org/abs/2111.06377), a
[transformer-based](https://arxiv.org/abs/1706.03762) neural
network.

*   It was trained using masked auto-encoding on a large corpus of
    health-related sounds, with a self-supervised learning objective on a
    massive dataset (\~174k hours) of two-second audio clips. At training time,
    it tries to reconstruct masked spectrogram patches from the visible patches.

*   After it is trained, its encoder can generate low-dimensional
    representations of two-second audio clips, optimized for capturing and
    containing the most salient parts of health-related information from
    sounds like coughs and breathes.

*   These representations, or embeddings, can be used as inputs to other
    models trained for a variety of supervised tasks related to health.

*   The HeAR model was developed based on a [ViT-L architecture](https://arxiv.org/abs/2010.11929)

*   Instead of relying on CNNs, a pure transformer applied directly to
    sequences of image patches is the idea behind the model architecture,
    and it resulted in good performance in image classification tasks. This
    approach of using the Vision Transformer (ViT) attains excellent results
    compared to state-of-the-art convolutional networks while requiring
    substantially fewer computational resources to train.

*   The training process for HeAR comprised of three main components
  *   A data curation step (including a health acoustic event detector);
  *   A general purpose training step to develop an audio encoder (embedding
      model), and
  *   A task-specific evaluation step that adopts the trained embedding model
      for various downstream tasks.

*   The system is designed to encode two-second long audio clips and
      generate audio embeddings for use in downstream tasks.

### Technical Specifications

*   Model type: [ViT (vision transformer)](https://arxiv.org/abs/2010.11929)

*   Key publication: [https://arxiv.org/abs/2403.02522](https://arxiv.org/abs/2403.02522)

*   Model created: 2023-12-04

*   Model Version: 1.0.0

### Performance & Validation

HeAR's performance has been validated via linear probing the frozen embeddings
on a benchmark of 33 health acoustic tasks across 6 datasets.

HeAR is benchmarked on a diverse set of health acoustic tasks spanning 13 health
acoustic event detection tasks, 14 cough inference tasks, and 6 spirometry
inference tasks, across 6 datasets, and it demonstrated that simple linear
classifiers trained on top of our representations can perform as good or better
than many similar leading models.

### Key performance metrics

*   HeAR achieved high performance on **diverse health-relevant tasks**:
    inference of medical conditions (TB, COVID) and medically-relevant
    quantities (lung function, smoking status) from recordings of coughs or
    exhalations, including a task on predicting chest X-ray findings (pleural
    effusion, opacities etc.).

*   HeAR had **superior device generalizability** compared to other models
    (MRR=0.745 versus second-best being CLAP with MRR=0.497), which is
    crucially important for real-world applications.

*   HeAR is more **data efficient** than baseline models, sometimes reaching
    the same level of performance when trained on as little as 6.25% of the
    amount of training data.

### Inputs and outputs

**Input:** Two-second long 16 kHz mono audio clip. Inputs can be batched so you
can pass in n=10 as (10,32k) or n=1 as (1,32k)

**Output:** Embedding vector of floating point values in (n, 512) for n
two-second clips in the vector, or an embedding of length 512 for each
two-second input clip.

### Dataset details

### Training dataset

For training, a dataset of YT-NS (YouTube Non-Semantic) was curated, and it
consisted of two-second long audio clips extracted from three billion public
non-copyrighted YouTube videos using a health acoustic event detector, totalling
313.3 million two-second clips or roughly 174k hours of audio. We chose a
two-second window since most events we cared about were shorter than that. The
HeAR audio encoder is trained solely on this dataset.

### Evaluation dataset

Six datasets were used for evaluation:

* [FSD50K](https://zenodo.org/records/4060432)
* [Flusense](https://github.com/Forsad/FluSense-data)
* [CoughVID](https://zenodo.org/records/4048312)
* [Coswara](https://zenodo.org/records/7188627)
* [CIDRZ](https://www.kaggle.com/datasets/googlehealthai/google-health-ai)
* [SpiroSmart](https://dl.acm.org/doi/10.1145/2370216.2370261)

## License

The use of the HeAR is governed by the [Health AI Developer Foundations terms of
use](https://developers.google.com/health-ai-developer-foundations/terms).

### Implementation information

Details about the model internals.

### Software

Training was done using [JAX](https://github.com/jax-ml/jax)

JAX allows researchers to take advantage of the latest generation of hardware,
including TPUs, for faster and more efficient training of large models.

## Use and limitations

### Intended use

*   Research and development of health-related acoustic biomarkers.

*   Exploration of novel applications in disease detection and health
    monitoring.

### Benefits

HeAR embeddings can be used for efficient training of AI models for
health acoustics tasks with significantly less data and compute than training
neural networks initialised randomly or from checkpoints trained on generic
datasets. This allows quick prototyping to see if health acoustics signals can
be used by themselves or combined with other signals to make predictions of
interest.

### Limitations

*   Limited Sequence Length: Primarily trained on 2-second audio clips.

*   Model Size: Current model size is too large for on-device deployment.

*   Bias Considerations: Potential for biases based on demographics and
    recording device quality, necessitating further investigation and
    mitigation strategies.

*   HeAR was trained using two-second audio clips of health-related sounds from
    a public non-copyrighted subset of Youtube. These clips come from a
    variety of sources but may be noisy or low-quality.

*   The model is only used to generate embeddings of the user-owned dataset.
    It does not generate any predictions or diagnosis on its own.

*   As with any research, developers should ensure that any downstream
    application is validated to understand performance using data that is
    appropriately representative of the intended use setting for the
    specific application (e.g., age, sex, gender, recording device,
    background noise, etc.).
