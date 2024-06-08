# Medical AI/ML Models Repository
This repository curates AI/ML models (open-source & closed-source) for medical applications, providing a centralized resource for collaboration and innovation in healthcare. Here are additional repositories you may find useful: [Datasets Repository](https://github.com/GlobalHealthAI/DataHub) | [Standards Repository](https://github.com/GlobalHealthAI/StandardsAndPractices)

## Open Source Medical Models

### LLMs (Large Language Models)

| Model Name     | Description                                                     | Link                                               | Use Cases                           |
| -------------- | --------------------------------------------------------------- | -------------------------------------------------- | ----------------------------------- |
| ClinicalBERT | A Pretrained Model on a large corpus of 1.2B words of diverse diseases. | [ClinicalBERT](https://huggingface.co/medicalai/ClinicalBERT) | Medical question-answering (QA) tasks |
| LLaVA-Med | A large language and vision model trained using a curriculum learning method for adapting LLaVA to the biomedical domain. | [microsoft/llava-med-v1.5-mistral-7b](https://huggingface.co/microsoft/llava-med-v1.5-mistral-7b) | Medical question-answering (QA) tasks |
| BioMistral-7B | A Collection of Open-Source Pretrained Large Language Models for Medical Domains. | [BioMistral](https://huggingface.co/BioMistral/BioMistral-7B) | Medical question-answering (QA) tasks |
| BioGPT         | A generative pre-trained transformer model for biomedical text generation. | [BioGPT](https://github.com/microsoft/BioGPT) | Biomedical text generation, NLP     |
| BioBERT        | A pre-trained biomedical language representation model.         | [BioBERT](https://github.com/dmis-lab/biobert)     | Biomedical text mining, NLP tasks   |
| ClinicalBERT   | BERT model fine-tuned on clinical notes.                        | [ClinicalBERT](https://github.com/EmilyAlsentzer/clinicalBERT) | Clinical text analysis, EHR data    |
| SciBERT        | A BERT-based model trained on scientific text.                  | [SciBERT](https://github.com/allenai/scibert)      | Scientific literature analysis      |
| BlueBERT       | BERT-based model trained on PubMed abstracts and MIMIC-III clinical notes. | [BlueBERT](https://github.com/ncbi-nlp/bluebert) | Biomedical research, clinical notes |


### Vision Models

| Model Name          | Description                                                         | Link                                               | Use Cases                        |
| ------------------- | ------------------------------------------------------------------- | -------------------------------------------------- | -------------------------------- |
| Vision Transformer             | A deep learning model for classification & segmentation on imaging data .    | [Vision Transformer](https://github.com/google-research/vision_transformer) | Radiology, Pathology  |
| CheXbert             | Automatic Labelers and Expert Annotations.    | [CheXbert](https://github.com/stanfordmlgroup/CheXbert) |  radiology   |
| UNet                | A convolutional neural network for biomedical image segmentation.   | [UNet](https://github.com/zhixuhao/unet)           | Biomedical image segmentation    |
| nnU-Net             | A self-configuring method for biomedical image segmentation.        | [nnU-Net](https://github.com/MIC-DKFZ/nnUNet)      | Biomedical image segmentation    |

### Multimodal Models

| Model Name | Description                                          | Link                                               | Use Cases                        |
| ---------- | ---------------------------------------------------- | -------------------------------------------------- | -------------------------------- |
| MedNLI     | Natural language inference dataset for the clinical domain. | [MedNLI](https://github.com/jgc128/mednli) | Clinical decision support, NLP   |
| Med3D      | Pre-trained 3D medical image analysis model.         | [Med3D](https://github.com/Tencent/MedicalNet)     | 3D medical image analysis        |

### Other Specialized Models

| Model Name | Description                                          | Link                                               | Use Cases                        |
| ---------- | ---------------------------------------------------- | -------------------------------------------------- | -------------------------------- |
| COVID-Net  | A deep learning model for detecting COVID-19 from chest X-ray images. | [COVID-Net](https://github.com/lindawangg/COVID-Net) | COVID-19 detection, radiology    |

### Frameworks
|  Name          | Description                                                         | Link                                               |
| ------------------- | ------------------------------------------------------------------- | -------------------------------------------------- |
| MONAI  | An open-source, PyTorch-based framework with multiple models for end-to-end workflow. | [MONAI](https://monai.io/) | 

## Closed Source Medical Models

### LLMs (Large Language Models)

| Model Name              | Description                                                   | Link                                                                                       | Use Cases                                           |
| ----------------------- | ------------------------------------------------------------- | ------------------------------------------------------------------------------------------ | --------------------------------------------------- |
| IBM Watson for Oncology | AI model for assisting oncologists in treatment decision-making | [IBM Watson for Oncology](https://www.ibm.com/watson-health/solutions/oncology)            | Oncology decision support, treatment planning       |

### Vision Models

| Model Name              | Description                                                     | Link                                                                                       | Use Cases                                           |
| ----------------------- | --------------------------------------------------------------- | ------------------------------------------------------------------------------------------ | --------------------------------------------------- |
| AI for Breast Cancer    | AI model for improving breast cancer screening accuracy         | [Google AI for Breast Cancer](https://health.google/caregivers/mammography/)                          | Breast cancer screening, diagnostics                |
| AI for Diabetic Retinopathy | Detects diabetic retinopathy and diabetic macular edema       | [Google AI for Diabetic Retinopathy](https://health.google/caregivers/arda/)                   | Diabetic retinopathy screening, ophthalmology       |
| Sepsis Prediction Algorithm | Early warning system for sepsis in ICU patients               | [Johns Hopkins Sepsis Prediction](https://www.hopkinsmedicine.org/)                        | Sepsis detection, ICU management                    |
| COVID-19 Severity Prediction | AI models for predicting COVID-19 severity                  | [Johns Hopkins COVID-19 Prediction](https://www.hopkinsmedicine.org/)                      | COVID-19 patient management, severity prediction    |
| InnerEye                | AI for medical imaging, including radiotherapy and image analysis | [Microsoft InnerEye](https://www.microsoft.com/en-us/research/project/medical-image-analysis/) | Radiotherapy planning, medical imaging analysis     |
| PathAI                 | Models for pathology image analysis, including cancer detection  | [PathAI](https://www.pathai.com/)                                                          | Pathology diagnostics, cancer detection             |
| RAD AI       | AI-powered assistants for radiologists                           | [RAD AI](https://www.radai.com/) | Radiology assistance, diagnostics                   |
| Butterfly iQ+           | Handheld ultrasound device with integrated AI for diagnostics    | [Butterfly iQ+](https://www.butterflynetwork.com/)                                         | Point-of-care diagnostics, ultrasound imaging       |

### Frameworks
|  Name          | Description                                                         | Link                                               |
| ------------------- | ------------------------------------------------------------------- | -------------------------------------------------- |
| NVIDIA Clara | NVIDIA Clara for healthcare and life sciences, from imaging and instruments to genomics and drug discovery. | [NVIDIA Clara](https://www.nvidia.com/en-sg/clara/) | 

## References: 
1. [Artificial Intelligence and Machine Learning (AI/ML)-Enabled Medical Devices](https://www.fda.gov/medical-devices/software-medical-device-samd/artificial-intelligence-and-machine-learning-aiml-enabled-medical-devices)
2. [Open Medical-LLM Leaderboard](https://huggingface.co/spaces/openlifescienceai/open_medical_llm_leaderboard)

## Getting Help
If you need help or have any questions, view [contributing guide](CONTRIBUTING.md) or feel free to reach out by opening an issue or joining our [Discord Community](https://discord.gg/KXG8V5ZSpy).

Your contributions are invaluable, and together, we can build a healthier future through innovation and excellence in medical AI/ML!
