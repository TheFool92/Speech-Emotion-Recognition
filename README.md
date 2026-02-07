# 🗣️ Speech Emotion Recognition (SER) - AI CLUB Task ✅

**Status**: Complete | **Deadline**: Feb 7, 2025 | **Test Accuracy**: **59.0%**

## 📊 Final Results [file:93]
| Metric       | Score    |
|--------------|----------|
| **Test Accuracy** | **59.0%** |
| **Macro F1-score** | **58.2%** |
| **Male F1**  | **59.7%** |
| **Female F1**| **53.3%** |
| **Pitch Bias**| **±3.2%** (Minimal) |

## 🛠️ Technical Pipeline [file:93]
1. **Dataset**: RAVDESS (1440 clips, 8 emotions)
2. **Features**: Log-Mel spectrograms (128×87, silence-trimmed)
3. **Model**: CNN (111K params, Conv+BatchNorm+Dropout)
4. **Training**: Adam, EarlyStopping, 100 epochs
5. **Test**: 55 clips → **59.0% acc** [file:93]

## 📈 Visuals [file:93]
![Training Curves](trainingcurves.png)
![Confusion Matrix](confusionmatrix.png)
![Angry vs Sad EDA](edacomparison.png)

## 🚀 Quick Inference [file:93]
```python
model = tf.keras.models.load_model('best_ser.keras')
pred = model.predict(mel_spec)
emotion = le.classes_[pred.argmax()]  # "angry"
