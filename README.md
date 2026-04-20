# 🌌 Planetary Crater Detection using Semantic Segmentation

## 📌 Overview
This project focuses on **automatic detection and segmentation of impact craters** on planetary surfaces (e.g., Mars) using deep learning techniques.

Manual crater detection is:
- Time-consuming  
- Error-prone  

Our system enables:
- ⚡ Scalable crater detection  
- 🤖 Automated segmentation  
- 📊 Post-processing for crater analysis  

---

## 🚀 Features
- 🔍 Semantic segmentation of craters from planetary images  
- 🧠 Multiple deep learning models (UNet variants, Transformer-based models)  
- 📈 Performance evaluation (IoU, Dice, Precision, Recall, F1-score)  
- 🧪 Advanced data augmentation (Copy-Paste Augmentation)  
- 📍 Post-processing to extract crater properties:
  - Center coordinates  
  - Radius  
- 🌐 Interactive web-based demo  

---

## 🗂️ Dataset
- **Dataset Used:** THEMIS Day IR Global Mosaic (Mars)  
- Includes:
  - Raw images  
  - Edge-enhanced images  
  - Binary masks
    
Link : https://www.kaggle.com/datasets/sahibjparmar/mars-crater-dataset-for-semantic-segmentation
---

## ⚙️ Methodology

### 1. Data Preprocessing
- Copy-Paste Augmentation (handles class imbalance)
- Random flips & rotations
- Illumination variation
- Noise injection

### 2. Model Development
We experimented with:
- UNet (Small & Large)
- Dense-UNet
- Attention-UNet
- Trans-UNet
- RDT-UNet & RDT-UNet++
- Ghost-based architectures
- Swin Transformer based model (**SwinCraterNet**)
- MIA-UNet (Multi-scale Inception Attention UNet)

### 3. Training Pipeline
- Data Loading  
- Augmentation  
- Model Training  
- Validation  
- Best Model Selection  

### 4. Post-processing
- Extract individual craters from segmentation mask  
- Compute:
  - Center coordinates  
  - Radius  
- Export results:
  - Annotated image  
  - CSV file  

---

## 📊 Results

| Model | Parameters | IoU | Dice | Precision | Recall | F1 |
|------|------------|-----|------|----------|--------|----|
| UNet (Small) | 7,849,509 | 0.5447 | 0.7052 | 0.7236 | 0.6878 | 0.7052 |
| UNet (Large) | 31,383,681 | 0.5826 | 0.7363 | 0.6738 | 0.8116 | 0.7363 |
| Dense-UNet | 721,060 | 0.4901 | 0.6578 | 0.5932 | 0.7382 | 0.6578 |
| Trans-UNet | 101,032,033 | 0.5160 | 0.6807 | 0.6081 | 0.7730 | 0.6807 |
| RDT-UNet | 13,253,097 | 0.4762 | 0.6452 | 0.5690 | 0.7449 | 0.6452 |
| RDT-UNet++ | 4,617,243 | 0.5477 | 0.7077 | 0.6642 | 0.7574 | 0.7077 |
| Ghost-RDT-UNet++ | 6,397,335 | 0.5678 | 0.7243 | 0.6617 | 0.8000 | 0.7243 |
| Ghost-ECA-MLP | 9,467,757 | **0.5984** | **0.7487** | 0.6991 | 0.8059 | **0.7487** |
| MIA-UNet | 28,139,885 | 0.5835 | 0.7370 | 0.6660 | 0.8248 | 0.7370 |
| Attention-UNet | 31,427,593 | 0.5827 | 0.7364 | 0.6687 | 0.8193 | 0.7364 |
| SwinCraterNet | 36,142,656 | 0.5647 | 0.7218 | 0.6473 | 0.8156 | 0.7218 |
| MSA-UNet | 45,017,509 | 0.4134 | 0.5850 | 0.4398 | **0.8734** | 0.5850 |
---

## Best Models 
Link : https://drive.google.com/drive/folders/1UzXdJjP1AjvKSgjsykHp3x3IZKwHo4Wq?usp=drive_link

---

## 🖥️ Demo Workflow
1. Upload planetary surface image  
2. Select model  
3. Run segmentation  
4. View:
   - Segmented mask  
   - Overlay image  
5. Run post-processing  
6. Download:
   - Annotated image  
   - CSV with crater details  

---

## 🧠 Challenges
- High variation in crater sizes  
- Overlapping and clustered craters  
- Low contrast & noisy regions  
- Irregular shapes due to erosion  
- Limited labeled data  

---

## 🔮 Future Work
- Improve post-processing (replace template matching)  
- Better separation of overlapping craters  
- Faster and more accurate crater extraction  

---

## 🛠️ Tech Stack
- **Language:** Python  
- **Frameworks:** PyTorch  
- **Libraries:** OpenCV, NumPy, Pandas  
- **Frontend:** Web-based interface  

---

## 👥 Team Members
- Amanpreet Singh  
- Dhruv Roy Talukdar  
- Harshit Srivastava  
- Meetkumar Boda  
- Sahib Parmar  
- V Arvind  
- Vinayak Bhosle  

---

## 📌 Conclusion
- Explored multiple segmentation architectures  
- Designed novel hybrid models  
- Achieved strong performance with efficient models  
- Built a complete end-to-end crater detection pipeline  

---

## ⭐ Acknowledgements
- Dataset: THEMIS Mars Data  
- Inspired by research in semantic segmentation and planetary science  

---

## 📬 Contact
For queries or collaboration, feel free to reach out!
