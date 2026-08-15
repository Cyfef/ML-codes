# ML-Codes
This repo contains **PyTorch** implementations of many **traditional machine learning** algorithms, **deep learning** algorithms, and **reinforcement learning** algorithms.  

For some algorithms based on strict mathematical formulas, it is advisable to maintain the **consistency between the code parameters and the mathematical symbols**.

## Structure

```txt
ML-codes/
├── Classification
│   ├── Adaboost
│   ├── Forests
│   ├── KNN
│   ├── Logistic
│   ├── Softmax
│   ├── SVM
│   └── Tree
├── Data
│   └── phd
├── NN
│   ├── AE
│   ├── CNN
│   ├── Diffusion
│   ├── GAN
│   ├── MLP
│   ├── PointNet
│   ├── RNN
│   ├── Transformer
│   ├── UNet
│   └── VAE
├── Regression
│   ├── GP
│   ├── Lasso
│   ├── Linear
│   └── Ridge
├── RL
│   ├── PPO
│   └── TRPO
└── Unsupervised
    ├── K-means
    ├── MoG
    └── PCA
```

## Environment

```bash
conda create -n ml python=3.11 -y
conda activate ml
pip install -r requirements.txt

# version due to your device
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## References

- https://github.com/MuLabPKU/MachineLearningCourse2025
- https://pku-epic.github.io/Intro2CV_2026/
- https://github.com/luokn/ml
- https://github.com/milesial/Pytorch-UNet