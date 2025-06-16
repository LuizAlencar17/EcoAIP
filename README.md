# EcoAIP 🌿🦒

EcoAIP is a cutting-edge machine learning project for **animal classification** using the **AIPResNet50 model**. It skillfully leverages a ResNet50-based architecture to process **Serengeti dataset partitions** for training, validation, and testing. 🚀

---

## Features ✨
* **Customizable training parameters**: Tailor your training to perfection. 🛠️
* **Early stopping mechanism**: Prevent overfitting and save time. ⏱️
* **Configurable dataset paths**: Easily switch between datasets. 📂
* **Model checkpointing and evaluation**: Track progress and ensure optimal performance. 💾

---

## Why EcoAIP Surpasses the Original AIP Approach 🚀

The original AIP approach focused on adaptive image processing embedding to enhance deep learning for ecological tasks on camera trap images. EcoAIP takes this a significant step further!

#### Original AIP Approach: Adaptive image processing embedding to make the ecological tasks of deep learning more robust on camera traps images.
Access the original paper here: [sciencedirect.com/science/article/pii/S1574954124002474](https://www.sciencedirect.com/science/article/pii/S1574954124002474)

1.  **Comprehensive DIP Enhancements**:
    * EcoAIP's improved **Digital Image Processing (DIP)** module integrates **all five crucial image processing functions** (gamma, contrast, white balance, tone, sharpening) with a **differentiable, learnable Gaussian sharpening kernel** and **advanced piecewise linear tone mapping**. 🎨
    * The original model had a simplified tone adjustment and a static sharpening kernel, limiting its adaptability to diverse scenes.

2.  **Enhanced Non-Local Parameter Predictor (NLPP)**:
    * The new **NLPP** incorporates **Multi-Head Self-Attention (MHSA)** to significantly improve global feature representation and environmental understanding. 🧠
    * The original used a simpler non-local block, which constrained its ability to capture complex environmental interactions in camera trap images.

3.  **Integration of CBAM in Backbone**:
    * The incorporation of **Convolutional Block Attention Module (CBAM)** into ResNet50 empowers the model to dynamically focus on the most informative spatial and channel features. 🎯
    * The original architecture lacked such attention mechanisms within ResNet, making it less effective in the cluttered, low-quality scenes typical of camera trap data.

4.  **Adaptive Soft Blending**:
    * EcoAIP employs a **soft gating mechanism** to smoothly combine original and enhanced images, allowing the model to dynamically determine the optimal degree of enhancement during training. 🖼️➡️✨
    * The original model used a hard threshold, which could lead to abrupt and potentially suboptimal decisions.

5.  **Joint End-to-End Training**:
    * Our improved architecture is **fully differentiable** from input through DIP and NLPP to the ResNet backbone, ensuring optimal co-adaptation of all modules during training. 🔗
    * The original implementation didn't fully leverage this end-to-end optimization potential, potentially leading to suboptimal parameter learning.

6.  **Robust Data Augmentation Strategy**:
    * EcoAIP encourages the use of **hybrid data augmentation** (synthetic exposure variations, blurs, etc.) to train the model under various real-world-like degradations. 🔄
    * While the original article introduced synthetic data, our approach explicitly integrates a flexible augmentation pipeline adaptable to different ecosystems beyond camera traps.

7.  **Scalability and Flexibility**:
    * This architecture is **highly modular**, allowing for easy extension with more sophisticated attention mechanisms (e.g., Transformer blocks) or advanced data augmentation strategies for broader ecological applications. 📈
    * The original design was more rigid, limiting potential enhancements and domain transferability.

**In Conclusion**: EcoAIP's enhanced architecture boosts **robustness, adaptability, and task performance** in ecological image classification by integrating more sophisticated image processing, attention, and parameter prediction mechanisms. It's a game-changer for wildlife monitoring! 🦉🦅🦊

---

## Installation 💻

1.  **Clone the repository**:
    ```bash
    git clone /home/luiz/experiments/my-repositories/EcoAIP
    cd EcoAIP
    ```
2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

---

## Usage ▶️

Run the training script:
```bash
python train.py --config config.aip_resnet50
```

---

## Configuration Parameters ⚙️

Here are the key parameters you can configure:

* **BATCH\_SIZE**: `16`
* **DATA\_TEST\_CSV\_PATH**: `/data/luiz/dataset/partitions/animal-classifier/serengeti/test.csv`
* **DATA\_TRAIN\_CSV\_PATH**: `/data/luiz/dataset/partitions/animal-classifier/serengeti/train.csv`
* **DATA\_VAL\_CSV\_PATH**: `/data/luiz/dataset/partitions/animal-classifier/serengeti/val.csv`
* **EPOCHS**: `100`
* **IMAGE\_SIZE**: `(300, 300)`
* **LEARNING\_RATE\_MODEL**: `0.001`
* **model\_name**: `normal`
* **MODEL**: `AIPResNet50`
* **OUTPUT\_DIR**: `/data/luiz/dataset/EcoAIP/`
* **PATIENCE**: `10`
* **SEED**: `42`
* **WEIGHTS\_PATH**: `/data/luiz/dataset/EcoAIP/AIPResNet50/model_best.pth`

---

## License 📄

This project is licensed under the **MIT License**. See `LICENSE` for more details. ✅