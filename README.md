# 🎮 Game Genre Classification from Screenshots

Deep learning project for automatic classification of video game genres from gameplay screenshots using transfer learning.

---

## 🧠 Problem Description

The goal of this project is to develop a deep learning model capable of predicting the **genre of a video game** based solely on a screenshot.

The task is formulated as a **multi-class image classification problem**:

**Input:** RGB image (game screenshot)  
**Output:** One of the following genres:

- 🔫 Shooter
- 🧙 RPG / MMORPG
- ⚽ Sports
- 🏟️ MOBA
- 🌍 Sandbox / Survival

To ensure the model learns **genre characteristics rather than specific games**, the dataset is split using a **game-level split**.  
This means some games are completely excluded from training and used only for testing.

---

## 🎯 Motivation

Automatic video game genre recognition has applications in:

- 📺 Content recommendation systems (Steam, YouTube, Twitch)
- 🗂️ Automatic media organization and filtering
- 📊 Video game market analysis

The project focuses on generalization to unseen games — a realistic scenario in industry systems.

---

## 🗃️ Dataset

We use the public Kaggle dataset:

👉 https://www.kaggle.com/datasets/juanmartinzabala/videogame-image-classification

It contains screenshots from 21 games:

ApexLegends, CSGO, ClashRoyale, DeathByDaylight, Dota2, EscapeFromTarkov, FIFA21, Fortnite, FreeFire, GTAV, LeagueOfLegends, Minecraft, Overwatch, PUBG_Battleground, RainbowSix, RocketLeague, Rust, SeaOfThieves, Valorant, Warzone, WorldOfWarcraft

Each game is mapped into one of the 5 defined genres.

> Dataset is NOT included in the repository due to size.

---

## ⚙️ Methodology

### Preprocessing
- Resize images to **224×224**
- Normalization
- Data augmentation (flip, crop, rotation, color jitter)

### Model
- 🧠 **ResNet50 (pretrained on ImageNet)**
- Transfer learning
- Fine-tuning last layers

### Training
- Loss: Cross-Entropy
- Optimizer: Adam
- Framework: PyTorch

---

## 📊 Evaluation

### Data Split (Game-Level)
- Test set: at least 1 game per genre
- Training set: remaining games
- Validation set: 20% of training images

### Metrics
- Accuracy
- Precision
- Recall
- F1-Score

### Analysis
- Confusion Matrix
- Misclassified samples inspection

---

## 🚀 Setup

```bash
git clone https://github.com/dmitar-strbac/game-genre-classification.git
cd game-genre-classification
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Download dataset manually and place it inside:
```
data/raw/
```

---

## 🏁 Training
```
python src/train.py
```

---

## 👤 Author
Dmitar Štrbac  
Faculty of Technical Sciences, University of Novi Sad

---

## 📜 License
MIT License
