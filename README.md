# Handwritten Digit Recognition using CNN with ONNX & FPGA Deployment Pipeline

## 📌 Project Overview

This project presents a complete end-to-end implementation of **handwritten digit recognition** using a custom-designed **Convolutional Neural Network (CNN)** trained on the widely used **MNIST dataset**.
The work demonstrates the **full deep learning lifecycle**, including dataset preprocessing, neural network design, supervised training, quantitative evaluation, inference performance measurement, visualization of predictions, and conversion of the trained model into a **hardware-deployable ONNX format** suitable for embedded AI systems such as **PYNQ-Z2 FPGA platforms**.

---

## 🎯 Objectives

* Design and train a lightweight yet accurate CNN for digit classification
* Achieve high generalization accuracy on unseen MNIST test samples
* Measure real-time inference latency on CPU environments
* Provide visual verification of predictions for interpretability
* Convert the trained PyTorch model into **ONNX** for embedded deployment
* Demonstrate readiness for **edge AI / FPGA acceleration workflows**

---

## 🧠 Model Architecture

The implemented CNN consists of:

* Two **Convolution → ReLU → MaxPooling** feature extraction stages
* A **Flatten layer** followed by fully connected dense layers
* Final **10-class output layer** representing digits **0–9**

The architecture is intentionally lightweight to ensure:

* Fast inference
* Low memory footprint
* Compatibility with **embedded hardware constraints**

---

## 📊 Performance Results

* **Test Accuracy:** ~99% on real MNIST test images
* **Inference Time (CPU):** ~40–60 ms per image
* **Visual Prediction Validation:** Confirmed correct classification across sampled test set

These results demonstrate **strong generalization** and **efficient computation**, making the model suitable for **real-time embedded inference scenarios**.

---

## ⚙️ Key Features Implemented

* Custom CNN training pipeline in **PyTorch**
* Automated **accuracy evaluation over multiple test samples**
* **Inference benchmarking** for performance analysis
* **Visual prediction interface** using OpenCV
* **Model export to ONNX** for cross-platform deployment
* Structured project layout for **reproducibility and scalability**

---

## 🧩 Embedded AI Readiness

The trained CNN model is exported to **ONNX**, enabling:

* Framework-independent inference
* Deployment on **ARM processors, edge devices, and FPGA toolchains**
* Future integration with **Xilinx PYNQ-Z2 FPGA acceleration workflows**

This bridges the gap between **deep learning research** and **real-world embedded AI deployment**.

---

## 🗂 Project Structure

```
cnn_model.pth          # Trained PyTorch CNN weights  
cnn_model.onnx         # Hardware-ready ONNX model  
cnn_infer.py           # Single-image inference script  
evaluate_accuracy.py   # Accuracy evaluation on MNIST test set  
visual_test.py         # Visual prediction verification  
export_onnx.py         # PyTorch → ONNX conversion script  
images/                # Sample MNIST test images  
labels.csv             # Ground-truth labels  
requirements.txt       # Python dependencies  
```

---

## 🚀 Future Enhancements

* Real-time **camera-based digit recognition**
* **Quantization & optimization** for faster embedded inference
* Deployment on **PYNQ-Z2 FPGA with hardware acceleration**
* Integration into **edge AI applications** such as smart meters or digit readers

---

## 👩‍💻 Author

**Sreenija Akarapu**

Embedded AI & Deep Learning Project
