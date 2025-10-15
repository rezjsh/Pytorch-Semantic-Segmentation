# 🏙️ Modular Semantic Segmentation Project

**An End-to-End MLOps Pipeline for Semantic Segmentation using PyTorch and Flask.**

This project implements a complete, modular, and stage-based pipeline for training a state-of-the-art **Attention UNet** model as the default model and collection of other models to choose for semantic segmentation (e.g., Cityscapes). It includes structured data handling, model training with checkpointing, metric logging, and deployment as a class-based Flask app.
## 🎯 Key Features

* **Modular MLOps Pipeline:** Training structured into distinct stages (`main.py`) for clarity and extensibility.
* **Attention UNet:** Implements the Attention UNet architecture using custom `AttentionGate` modules.
* **Model Zoo:** Collection of pre-trained models (e.g., DeepLabV3+, SimpleFCN) available for inference.
* **Custom Models:** Easily add and switch between different model architectures.
* **Configuration Management:** Uses `config.yaml` and `params.yaml` for environment-agnostic configuration and parameter tuning.
* **Logging and Monitoring:** Integrated logging and monitoring for training and inference.
* **Class-Based Flask API:** Robust, class-based deployment using a **Singleton Pattern** (`PredictionService`) to load the heavy model only once upon startup.
* **Reproducibility:** Dedicated `environment.yaml` file for easy environment setup via Conda.

---

## 📁 Project Structure

The project is organized to separate configuration, pipeline stages, core components, and the final application deployment.
```
Pytorch-Semantic-Segmentation
    ├── LICENSE
    ├── README.md
    ├── setup.py
    ├── template.py
    ├── run.py
    ├── requirements.txt
    └── Semantic_Segmentation
        ├── config/
        │   └── config.yaml
        ├── data/
        │   ├── cityscapes/
        │   ├── train/
        │   └── val/
        ├── logs/
        ├── docs/
        ├── reports/
        ├── models/
        ├── results/
        ├── src/
        │   ├── components/
        │   │   ├── data_ingestion.py
        │   │   ├── data_transformation.py
        │   │   ├── dataset.py
        │   │   ├── data_loader.py
        │   │   ├── model.py
        │   │   └── model_trainer.py
        │   ├── entity/
        │   │   └── config_entity.py
        │   ├── config/
        │   │   └── configuration.py
        │   ├── constants/
        │   │   └── constants.py
        │   ├── core/
        │   │   └── singletone.py
        │   ├── factory/
        │   │   └── model_factory.py
        │   ├── models/
        │   │   ├── AttentionGate.py
        │   │   ├── AttentionUnet.py
        │   │   ├── ConvBlock.py
        │   │   ├── DeepLabV3P.py
        │   │   └── SimpleFCN.py
        │   ├── modules/
        │   │   └── MetricLogger.py
        │   ├── pipeline/
        │   │   ├── stage_01_data_ingestion.py
        │   │   ├── stage_02_data_transformation.py
        │   │   ├── stage_03_dataset.py
        │   │   ├── stage_04_data_loader.py
        │   │   ├── stage_05_model.py
        │   │   └── stage_06_model_trainer.py
        │   ├── utils/
        │   │   ├── helpers.py
        │   │   ├── logging_setup.py
        │   │   └── device.py
        │   └── app/
        │       ├── prediction_service.py
        │       ├── static/
        │       │   └── style.css
        │       ├── templates/
        │       │   └── index.html
        │       └── routes.py
        ├── main.py
        └── params.py
```

---

## 🛠️ Setup and Installation

### Prerequisites

* **Conda / Miniconda** (Recommended for managing virtual environments).
* **A trained model checkpoint.** (The Flask app expects a file like `AttentionUNet_best.pth` to be available in the configured checkpoint directory).

### Environment Setup

1.  **Clone the repository:**
    ```bash
    git clone [<your_repo_url>](https://github.com/rezjsh/Pytorch-Semantic-Segmentation.git)
    cd Pytorch-Semantic-Segmentation
    ```

2.  **Create and activate the Conda environment:**
    ```bash
    conda env create -f environment.yaml
    conda activate semantic-segmentation-app
    ```

3.  **Place Data:** Ensure your segmentation dataset (e.g., Cityscapes images and labels) is placed in the location specified in `params.yaml` (default: `data/cityscapes`).

---

## 💻 Usage: Training Pipeline

The `main.py` file orchestrates the entire training pipeline.

### Stages Executed

1.  **Data Ingestion:** Downloads/prepares the raw data.
2.  **Data Transformation:** Defines and applies image/label transformations (resizing, normalization).
3.  **Dataset Preparation:** Creates custom PyTorch `Dataset` objects.
4.  **DataLoader Creation:** Creates PyTorch `DataLoader` objects for batching.
5.  **Model Building:** Instantiates the **Attention UNet** model using the `ModelFactory`.
6.  **Model Training & Evaluation:** Runs the training loops, logs metrics, and saves checkpoints.

### Running Training

To retrain the model or reproduce results:

```bash
python main.py
```

Check the artifacts/logs/running_logs.log for detailed output and the artifacts/model_checkpoints/ directory for the saved .pth files.

## 🚀 Usage: Flask Deployment
The Flask application serves the trained model via a simple web interface.

### Running the Web Service

1. Ensure you are in the Conda environment.
   ```bash
   conda activate semantic-segmentation-app
2. Run the Flask entry point:
   ```bash
   python app.py
   ```
   The server will start, and the PredictionService will load the heavy PyTorch model into memory immediately.
3. Access the Application:
   * Open your web browser to: http://127.0.0.1:5000 (or the host/port shown in your terminal).


### API Functionality
The application exposes a single functional route via the main blueprint:
```
Endpoint  |  Method  |  Description                                                             
----------+----------+--------------------------------------------------------------------------+
/         |  GET     |  Renders the main image upload page (index.html).                        
/predict  |  POST    |  Accepts a multipart image file, runs segmentation, returns image + mask.
```
## ⚙️ Configuration & Customization
The project is highly configurable through the following files:
1. **params.yaml**

    This file controls the core operational parameters:

    ```
    Section        |  Parameter      |  Description                               
    ---------------+-----------------+--------------------------------------------
    transforms     |  size           |  Model input size                       
    transforms     |  num_classes    |  Number of semantic mask classes (e.g., 21)
    model          |  model_name     |  Model to use (default: AttentionUNet)     
    model_trainer  |  learning_rate  |  Optimizer learning rate                   
    data_loader    |  batch_size     |  Training/validation batch size            
    ```
2. **config/config.yaml**

    This file handles artifact and path management:

    ```
    Section        |  Parameter      |  Description                               
    ---------------+-----------------+--------------------------------------------
    data_ingestion |  dest_dir       |  Local path for raw data.
    model_trainer  |  checkpoint_dir  |  Directory where model checkpoints (_best.pth, _last.pth) are saved. Crucial for Flask deployment.

    ```

## ⚖️ License
This project is licensed under the MIT License - see the LICENSE file for details. (If you have one, otherwise remove this section).

## 🙏 Acknowledgments
* Built with PyTorch.

* Inspired by modular MLOps design patterns.

* The Attention Gate implementation follows the structure proposed by Oktay et al. (Attention U-Net: Learning Where to Look for the Pancreas).