# Health Sound Event Detector

This repository provides three TensorFlow SavedModels as supplementary assets
for the HeAR model, designed for detecting health-related sounds. These models,
with parameter counts significantly smaller than the hundreds of millions in
models like ViT-Large used in HeAR, offer efficient alternatives for various
deployment scenarios or dataset filtering tasks. They can also be converted to
`.tflite` format for even more efficient on-device or low-power, low-latency
applications.

## Models

The following event detector models use the same frontend and have the same
input and output specifications. The only difference is the MobileNet backend.

1.  **event\_detector\_small**:
    *   Architecture: [MobileNetV3Small](https://www.tensorflow.org/api_docs/python/tf/keras/applications/MobileNetV3Small)
    *   Input: 2 second single channel 16kHz audio clip
    *   Output: 8 detection probability scores for the following labels:
        *   `['Cough', 'Snore', 'Baby Cough', 'Breathe', 'Sneeze',
              'Throat Clear', 'Laugh', 'Speech']`
    *   Description: A compact model trained for health sound event detection,
        integrated with our optimized spectrogram frontend. This model, with
        approximately **1M parameters (3.60 MB)**, is particularly suitable for
        resource-constrained environments.

2.  **event\_detector\_large**:
    *   Architecture: [MobileNetV3Large](https://www.tensorflow.org/api_docs/python/tf/keras/applications/MobileNetV3Large)
    *   Input: 2 second single channel 16kHz audio clip
    *   Output: 8 detection probability scores for the following labels:
        *   `['Cough', 'Snore', 'Baby Cough', 'Breathe', 'Sneeze',
              'Throat Clear', 'Laugh', 'Speech']`
    *   Description: A larger and potentially more accurate model trained for
        health sound event detection, also integrated with our optimized
        spectrogram frontend. This model, with approximately **3M (11.46 MB)**,
        can capture more complex patterns at the cost of slower, more
        compute-heavy inference.

3.  **spectrogram\_frontend**:
    *   Input: 2 second single channel 16kHz audio clip
    *   Output: A spectrogram with 200 time steps (for 2s input) and 48
        Mel-frequency bins
    *   Description: Our custom, on-device optimized spectrogram frontend. It
        efficiently converts raw audio into
        [PCEN](https://research.google/pubs/trainable-frontend-for-robust-and-far-field-keyword-spotting)
        scaled
        [Mel-spectrogram](https://huggingface.co/learn/audio-course/en/chapter1/audio_data#mel-spectrogram)
        features. This frontend has approximately **5k parameters (18.56 KB)**,
        none of which are trainable.
    *   Usage:
        *   Already fused with the `event_detector_small` and
            `event_detector_large` models.
        *   Can be used standalone for:
            *   Generating and visualizing audio features.
            *   Creating custom audio datasets.
            *   Integration with other machine learning models.

## License

license: other license_name: health-ai-developer-foundations license_link:
https://developers.google.com/health-ai-developer-foundations/terms
