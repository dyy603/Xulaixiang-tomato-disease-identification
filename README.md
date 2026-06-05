# Tomato Leaf Disease Identification System
The tomato leaf disease recognition system based on improved AlexNet achieves automatic classification and recognition of leaf diseases through deep learning technology.
# Project Overview 
This project aims to use deep learning technology to construct an efficient tomato leaf disease recognition model. By utilizing an improved AlexNet network architecture combined with attention mechanisms (multi head self attention mechanism and multi head late attention mechanism), automatic recognition of common tomato leaf diseases is achieved, providing support for early detection and diagnosis of diseases in agricultural production.
# Project Structure
```
MLSA/
├── MLSA.pth                   # weight file
├── MLSA_train.py              # Main script for model training
├── model.py                   # Definition of the improved AlexNet model
├── mla.py                     # Implementation of the MLA attention mechanism
├── msa.py                     # Implementation of the MSA attention mechanism
├── predict.py                 # Model prediction script
├── Preprocess_dehanced.py     # data processing
├── inference.py               # Model inference script
```
# Core Technologies 
__1.Improved AlexNet__：Add a layer of convolution to enhance the depth of feature extraction.  
__2.Integrate Attention Mechanisms：__
  multi head self attention mechanism and multi head late attention mechanism  
  __3.The early stop mechanism__: Terminate training prematurely when validation set performance stops improving
# Recommended Environment
Python 3.12  
PyTorch == 2.3
# Dataset acquisition and structure
The data can be accessed through the Baidu Cloud link:https://pan.baidu.com/s/1Q3rgLQvhX9-p05pSMu9L9g?
The dataset should be organized in the following structure:
```
fanqie
├── fanqie data/ 
│   ├── class1/
│   ├── class2/
│   └── ...
├── train/
│   ├── class1/
│   ├── class2/
│   └── ...
├── val/
├── ├──class1/
│   ├── class2/
│   └── ...
├── test/
├── ├──class1/
│   ├── class2/
│   └── ...
```
Each category folder contains tomato leaf images corresponding to that category.
# Model Training 
Use the `MLSA_train.py` script for model training:
```
python MLSA_train.py
```
# Training Parameter Description
During the training process, model weights will be automatically saved. After training, the model performance will be evaluated on the test set.
| Initial learning rate | Epoch| Batch size | 
|:------|:----:|-------:|
|0.0001 | 100  | 64   |  
# Model Inference
Use the provided inference script to make predictions on new images:  
__Single Image Prediction__  
```
python inference.py --model_path ./MLSA.pth --image_path ./test.jpg
```
__Batch Image Prediction__  
```
python inference.py --model_path ./MLSA.pth --image_dir ./test_images
```
# Performance Evaluation
After the training is completed, the model will be evaluated on the test set, and key metrics such as accuracy will be output. 
# References and contact information
The paper is in the submission stage and will update the BiBTeX citation format after its official publication. Currently, it can be temporarily cited:
```
@article{tssc_pea_disease,  
  title={MLSA: A Multi-Head Latent and Self-Attention Deep Learning Network for Tomato Leaf Disease Identification },  
  author={[Author's name, to be added when published]},  
  journal={[Journal name, to be supplemented after acceptance]},  
  year={2025},  
  note={Manuscript submitted for publication}  
}  
```
# Contact Information
If you encounter code running issues or academic exchange needs, please contact:  
Email:dongyanyanhuuc@yeah.net  
GitHub Issue：Submit an issue directly in this warehouse
