
# EcoAIP

EcoAIP is a machine learning project designed for animal classification using the AIPResNet50 model. It leverages a ResNet50-based architecture to process Serengeti dataset partitions for training, validation, and testing.

## Features
- Customizable training parameters
- Early stopping mechanism
- Configurable dataset paths
- Model checkpointing and evaluation

## Why the EcoAIP is Superior to the Original AIP Approach

#### -  Original AIP Approach: Adaptive image processing embedding to make the ecological tasks of deep learning more robust on camera traps images. 
Access: sciencedirect.com/science/article/pii/S1574954124002474

1. **Comprehensive DIP Enhancements**:
   - The improved DIP module incorporates all five crucial image processing functions (gamma, contrast, white balance, tone, sharpening) with a differentiable, learnable Gaussian sharpening kernel and an advanced tone mapping using a piecewise linear function. 
   - The original model had a simplified tone adjustment and static sharpening kernel, reducing its capacity to adapt image processing optimally to different scenes.

2. **Enhanced Non-Local Parameter Predictor (NLPP)**:
   - The new NLPP incorporates Multi-Head Self-Attention (MHSA) to improve global feature representation and environmental understanding.
   - The original used a simple non-local block, limiting its ability to capture complex environmental interactions in camera trap images.

3. **Integration of CBAM in Backbone**:
   - Incorporation of Convolutional Block Attention Module (CBAM) in ResNet50 enables the model to focus on the most informative spatial and channel features dynamically.
   - The original architecture lacked such attention mechanisms within the ResNet, making it less effective in cluttered, low-quality scenes typical of camera trap data.

4. **Adaptive Soft Blending**:
   - The proposed model uses a soft gating mechanism to combine original and enhanced images smoothly, allowing the model to decide the degree of enhancement dynamically during training.
   - The original model employed a hard threshold, introducing abrupt and potentially suboptimal decisions.

5. **Joint End-to-End Training**:
   - The improved architecture is fully differentiable from input through DIP and NLPP to the ResNet backbone, allowing optimal co-adaptation of all modules during training.
   - The original implementation did not fully exploit this end-to-end optimization potential, possibly leading to suboptimal parameter learning.

6. **Robust Data Augmentation Strategy**:
   - Our architecture encourages the use of hybrid data augmentation (synthetic exposure variations, blurs, etc.) to train the model under varied real-world-like degradations.
   - While the article introduced synthetic data, our approach explicitly integrates a flexible augmentation pipeline adaptable to different ecosystems beyond camera traps.

7. **Scalability and Flexibility**:
   - This architecture is modular, allowing easy extension with more sophisticated attention mechanisms (e.g., Transformer blocks) or advanced data augmentation strategies for broader ecological applications.
   - The original design was more rigid, limiting potential enhancements and domain transferability.

**Conclusion**:
The improved architecture enhances robustness, adaptability, and task performance in ecological image classification by integrating more sophisticated image processing, attention, and parameter prediction mechanisms.


## Installation
1. Clone the repository:
   ```bash
   git clone /home/luiz/experiments/my-repositories/EcoAIP
   cd EcoAIP
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
Run the training script:
```bash
python train.py --config config.aip_resnet50
```

## Configuration Parameters
- **BATCH_SIZE**: 16
- **DATA_TEST_CSV_PATH**: `/data/luiz/dataset/partitions/animal-classifier/serengeti/test.csv`
- **DATA_TRAIN_CSV_PATH**: `/data/luiz/dataset/partitions/animal-classifier/serengeti/train.csv`
- **DATA_VAL_CSV_PATH**: `/data/luiz/dataset/partitions/animal-classifier/serengeti/val.csv`
- **EPOCHS**: 100
- **IMAGE_SIZE**: (300, 300)
- **LEARNING_RATE_MODEL**: 0.001
- **LOSS_COMPUTATION**: normal
- **MODEL**: AIPResNet50
- **OUTPUT_DIR**: `/data/luiz/dataset/EcoAIP/`
- **PATIENCE**: 10
- **SEED**: 42
- **WEIGHTS_PATH**: `/data/luiz/dataset/EcoAIP/AIPResNet50/model_best.pth`

## License
This project is licensed under the MIT License. See `LICENSE` for details.


