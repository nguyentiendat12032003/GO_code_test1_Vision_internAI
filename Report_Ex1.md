# Dog vs Cat Image Classification
## 1. Pipeline
Dataset Collection & Preparation:
- Collected a dataset (~6000 labeled images, balanced Dog vs Cat) from Roboflow.
- Checked and split data into training and validation sets.
Model Training:
- Used Teachable Machine to train a classifier.
- Base Model: Transfer learning with a pre-trained CNN backbone (MobileNetV2 / EfficientNetLite) trained on ImageNet.
Training strategy:
- Feature extractor layers of base model frozen.
- Added a new dense classification head (fully connected layer + softmax).
Hyperparameters:
- Epochs: 25 → The model was trained for 25 full passes over the dataset, enough for convergence without heavy overfitting.
- Batch size: 16 → Each gradient update used 16 images at once, balancing between stability and speed on limited hardware.
- Learning rate: 0.001 → A small learning rate ensured stable training, preventing the optimizer from overshooting minima.
- Optimizer: Adam (adaptive learning rate optimization, widely used for deep learning).
- Loss Function: Binary cross-entropy (since there are 2 classes: Dog vs Cat).
Model Export
- Exported model in TensorFlow SavedModel format, compatible with Python / TensorFlow.js / TensorFlow Lite.
Saved in savedmode/.
Model Evaluation: 
- Wrote evaluate.py to test model on validation dataset.
- Printed accuracy and classification results.
Model Inference (Testing)
- Wrote load_savemode.py to classify new/random input images.
## 2. Algorithms Used
Transfer Learning CNN (MobileNetV2 / EfficientNetLite backbone).
Frozen feature extractor + new dense classification head.
Adam optimizer + Binary cross-entropy loss for binary classification.
## 3. Remaining Problems
Since training was done on Teachable Machine, hyperparameter control is limited:
- Cannot fine-tune backbone layers.
- Cannot try advanced optimizers, schedulers, or early stopping.
- Cannot handle model drift or customize augmentation strategies.
- Hard to upgrade or scale model to multi-class tasks beyond dog/cat.
- Limited ability to debug or adapt to edge cases in private test data.
## 4. Ideas for Improvements
- Train model directly in TensorFlow/PyTorch for full control of hyperparameters.
- Apply data augmentation (rotation, flipping, brightness shifts) to improve generalization.
- Use transfer learning from stronger backbones (ResNet, EfficientNetB3).
- Implement fine-tuning on some layers instead of freezing the whole backbone.
- Export optimized versions (TensorFlow Lite, ONNX) for faster inference.
- Add logging & monitoring to handle errors and detect misclassifications.







