# Flag Classification of South American Countries 

This project performs **image classification of South American country flags** using a **finetuned VGG16 deep learning model**.  
A custom dataset of over **10,000 flag images** was collected and used for training.

---

## Results

| Metric   | Value                                           |
|----------|------------------------------------------------|
| Accuracy | 97.0%                                           |
| Dataset  | ~10,000 self-collected flag images              |
| Classes  | South American countries (e.g., Brazil, Argentina, Chile, etc.) |

---

## How to Run It

### 1. Install dependencies

```bash
pip install -r requirements.txt
```
```bash
You can use the inference.py script to predict on your own dataset or individual images.
dataset_folder/
    Brazil/
        img1.jpg
        img2.jpg
    Argentina/
        img3.jpg
    Chile/
        ...
```
```bash
**Then run** : python inference.py --dataset path/to/dataset_folder
```
```bash
The model file (flag_classifier_vgg16.h5) will be downloaded automatically from Hugging Face just make sure u have an active internet connection while running inference.py
```

```bash
Notes
The model is finetuned specifically for South American flags and may not perform well on other datasets and would forcefully classify non South American flags.

Images will be resized automatically during inference.

Internet connection is required on the first run to download the model.
```

```bash
Contact
If you have any questions or issues, please open an issue or contact me at radiumiilee1729@gmail.com.
```

