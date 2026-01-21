# Understanding SVM Margins

This module explores the concept of **Support Vector Machine (SVM) margins** and how they are affected by data separability and regularization.

## Concepts Covered
- Linearly separable data and hard margin SVM
- Overlapping data and limitations of hard margin
- Soft margin SVM and regularization
- Effect of parameter `C` on margin width and support vectors

## File Structure
- `data.py` – Dataset generation utilities
- `visual.py` – Visualization helpers for SVM decision boundaries
- `hard_margin.py` – Hard margin SVM on separable data
- `overlapping_data.py` – Hard margin failure on overlapping data
- `soft_margin.py` – Soft margin SVM demonstration
- `margin_vs_C.py` – Effect of different `C` values
- `requirements.txt` – Required Python dependencies

## How to Run
Install dependencies:
```bash
pip install -r requirements.txt
