# Nvidia_AISpecialist_Certification
# YOLOv5-Based Animal Detection for Safer Autonomous Driving

---

## Overview

> **Project Background**
> 

> **Project Description**
> 

> **Data Acquisition Methods**
> 

> **Annotation process**
> 

> **YOLOv5 Training Process**
> 

> **Training Results**
> 

> **Significance of this Project**
> 

> **Faced Limitations**
> 

> **Discussion**
> 

---

## Project Background

<aside>
💬

**The autonomous vehicle market is rapidly growing, with increasing consumer demand and an expanding market size. While the advancement of autonomous driving technology is exciting, as a cat owner, I believe it’s equally important to prevent collisions with animals like cats and dogs during autonomous driving. Ensuring ethical development alongside technological progress is a crucial challenge.**

</aside>

## **Project Description**

<aside>
💬

**This project utilizes YOLOv5 to develop a real-time animal detection system for autonomous vehicles. This system aims to detect animals such as dogs and cats, assisting the vehicle in adjusting its actions to avoid potential collisions. The process involves collecting data, labeling it, training a model, and testing its performance for accurate detection. Ultimately, the goal is to improve road safety by enabling autonomous vehicles to respond effectively to animals on the road.**

</aside>

## **Data Acquisition Methods**

<aside>
📷

**I recorded my pet with my phone and downloaded videos of dogs and cats from [Videvo](https://www.videvo.net/).**

[https://drive.google.com/drive/folders/1Dh0xd0Z7DzT_wSvgdRksRBRvK9Sk87Qp?usp=drive_link](https://drive.google.com/drive/folders/1Dh0xd0Z7DzT_wSvgdRksRBRvK9Sk87Qp?usp=drive_link)

</aside>

<aside>
💾

**I downloaded images of various breeds of cats and dogs from [Roboflow](https://roboflow.com/)**

[https://drive.google.com/drive/folders/1vd2TEx0XRxx6_SK336H9BihINHWTir3v?usp=sharing](https://drive.google.com/drive/folders/1vd2TEx0XRxx6_SK336H9BihINHWTir3v?usp=sharing)

</aside>

## Annotation process

<aside>
📋

### Creating a Project

![390937532-13ccf404-2d11-4b1a-a0a7-a0a2788f56a1](https://github.com/user-attachments/assets/e2f0c212-1622-46a8-a9c2-6fc861d40725)

**I appropriately named the project and set its type**

</aside>

<aside>
📋

### Labeling

![390937535-8a82bb0b-162c-4e71-b805-132b4604584e](https://github.com/user-attachments/assets/57760a56-60a1-457f-b6e3-c3be3e022c81)

**I used Roboflow to convert the video into images. Then, to train the model more accurately, I annotated the images with polygons instead of rectangles and set the appropriate classes for dogs and cats**

</aside>

<aside>
📋

### Check Files

![390937520-075648ce-9e63-49d2-aba5-eb4b54e23073](https://github.com/user-attachments/assets/66934d51-4bf4-4b74-b860-2df32ab7dfa7)

![390937526-7d1c9b8a-410e-4506-af52-aa8953d550e7](https://github.com/user-attachments/assets/ad81261d-4019-489d-b73f-a5abd9225f9a)
**After completing the annotations, I verified that the images and text files were saved correctly without any missing files**

</aside>

## YOLOv5 Training Process

<aside>
📋

```python
**!git clone [https://github.com/ultralytics/yolov5](https://github.com/ultralytics/yolov5)
%cd yolov5
%pip install -qr requirements.txt**
```

**Clones the YOLOv5 repository from GitHub to your local machine and installs all the dependencies listed in the requirements.txt file.**

---

```python
**!pip install Pillow==10.3**
```

**Installs a specific version (10.3) of the Python library Pillow, which is used for image processing tasks**

---

```python
**!mkdir -p Train/labels
!mkdir -p Train/images
!mkdir -p Val/labels
!mkdir -p Val/images**
```

**Creates the `Train` and `Val` folders, along with their subfolders `images` and `labels`, providing storage space for images and text files.**

---

```python
**import os
import shutil
from sklearn.model_selection import train_test_split

def create_validation_set(train_path, val_path, split_ratio=0.3):
    os.makedirs(os.path.join(val_path, 'images'), exist_ok=True)
    os.makedirs(os.path.join(val_path, 'labels'), exist_ok=True)

    train_images = os.listdir(os.path.join(train_path, 'images'))
    train_images = [f for f in train_images if f.endswith(('.jpg', '.jpeg', '.png'))]

    _, val_images = train_test_split(train_images,
                                   test_size=split_ratio,
                                   random_state=42)

    for image_file in val_images:
        src_image = os.path.join(train_path, 'images', image_file)
        dst_image = os.path.join(val_path, 'images', image_file)
        shutil.copy2(src_image, dst_image)

        label_file = os.path.splitext(image_file)[0] + '.txt'
        src_label = os.path.join(train_path, 'labels', label_file)
        dst_label = os.path.join(val_path, 'labels', label_file)
        if os.path.exists(src_label):
            shutil.copy2(src_label, dst_label)

    print(f"Created validation set with {len(val_images)} images")

train_path = '/content/drive/MyDrive/yolov5/Train'
val_path = '/content/drive/MyDrive/yolov5/Val'

create_validation_set(train_path, val_path)**
```

**Use the `train_test_split()` function to split 30% of the data from the `Train`  into `Val`, and then print the number of image and label files in both the `Train`  and `Val` directories after the split.**

---

```python
**def check_dataset():
    train_path = '/content/drive/MyDrive/yolov5/Train'
    val_path = '/content/drive/MyDrive/yolov5/Val'

    train_images = len(os.listdir(os.path.join(train_path, 'images')))
    train_labels = len(os.listdir(os.path.join(train_path, 'labels')))

    val_images = len(os.listdir(os.path.join(val_path, 'images')))
    val_labels = len(oxs.listdir(os.path.join(val_path, 'labels')))

    print("Dataset status:")
    print(f"Train - Images: {train_images}, Labels: {train_labels}")
    print(f"Val - Images: {val_images}, Labels: {val_labels}")

check_dataset()**
```

**The `check_dataset()` function checks the number of image and label files in the `Train` and `Val` directories**

---

```python
**import torch
import os
from IPython.display 
import Image, clear_output**
```

**`import torch`: For tensor operations and deep learning with PyTorch.**

**`import os`: For managing files and directories.
`from IPython.display import Image, clear_output`: To display images and clear outputs in Colab.**

---

```python
**import numpy as np
import tensorflow as tf
import os
from PIL import Image
from tensorflow.python.eager.context import eager_mode

def _preproc(image, output_height=512, output_width=512, resize_side=512):
    with eager_mode():
        h, w = image.shape[0], image.shape[1]
        scale = tf.cond(tf.less(h, w), lambda: resize_side / h, lambda: resize_side / w)
        resized_image = tf.compat.v1.image.resize_bilinear(tf.expand_dims(image, 0), [int(h*scale), int(w*scale)])
        cropped_image = tf.compat.v1.image.resize_with_crop_or_pad(resized_image, output_height, output_width)
        return tf.squeeze(cropped_image)

def Create_npy(imagespath, imgsize, ext) :
    images_list = [img_name for img_name in os.listdir(imagespath) if
                os.path.splitext(img_name)[1].lower() == '.'+ext.lower()]
    calib_dataset = np.zeros((len(images_list), imgsize, imgsize, 3), dtype=np.float32)

    for idx, img_name in enumerate(sorted(images_list)):
        img_path = os.path.join(imagespath, img_name)
        try:
            if os.path.getsize(img_path) == 0:
                print(f"Error: {img_path} is empty.")
                continue

            img = Image.open(img_path)
            img = img.convert("RGB") 
            img_np = np.array(img)

            img_preproc = _preproc(img_np, imgsize, imgsize, imgsize)
            calib_dataset[idx,:,:,:] = img_preproc.numpy().astype(np.uint8)
            print(f"Processed image {img_path}")

        except Exception as e:
            print(f"Error processing image {img_path}: {e}")

    np.save('calib_set.npy', calib_dataset)**
```

**Preprocesses images by resizing and central-cropping them, then saves the processed images as a NumPy array in the `calib_set.npy` file.**

---

```python
**Create_npy('/content/drive/MyDrive/yolov5/Train/images', 512, 'jpg')**
```

**Loads and processes images from `/content/drive/MyDrive/yolov5/Train/images` into 512x512 size, saving them as a NumPy array in `calib_set.npy`.**

---

```python
**!python train.py --img 512 --batch 16 --epochs 300 --data /content/drive/MyDrive/yolov5/data.yaml --weights yolov5n.pt --cache**
```

**Trains a YOLOv5 model with the following parameters:**

- **`--img 512`: Input image size of 512x512 pixels.**
- **`--batch 16`: Batch size of 16 images per training step.**
- **`--epochs 300`: Train for 300 epochs.**
- **`--data /content/drive/Mydrive/yolov5/data.yaml`: Specifies the dataset configuration file.**
- **`--weights yolov5n.pt`: Starts training with pre-trained weights from `yolov5n.pt`.**
- **`--cache`: Caches images for faster data loading during training.**

---

```python
**%load_ext tensorboard
%tensorboard --logdir runs**
```

**`%load_ext tensorboard`: Loads TensorBoard extension.
`%tensorboard --logdir runs`: Launches TensorBoard to view logs from the runs directory**

---

```python
**!python detect.py --weights /content/drive/MyDrive/yolov5/runs/train/exp/weights/best.pt --img 512 --conf 0.1 --source /content/drive/MyDrive/dog.mp4**
```

**Uses the trained model (`best.pt`) to perform object detection on the video `dog.mp4`, with a resolution of 512 and a confidence threshold of 0.1.**

---

</aside>

## **Training Results**

<aside>
📋

[https://drive.google.com/drive/folders/1EXorwSwcuytMRyNO9RDN6n_Bs4zD4ZcA?usp=drive_link](https://drive.google.com/drive/folders/1EXorwSwcuytMRyNO9RDN6n_Bs4zD4ZcA?usp=drive_link)

### 📊C**onfusion_matrix**

![390937528-9231834d-dc53-46e4-8e35-69079c185199](https://github.com/user-attachments/assets/10d6560a-1e0e-42b2-8273-78cc255fa6d0)

### 📊**F1_curve**

![390937536-33118b4f-c185-4fd1-aec6-0b33ee5a8243](https://github.com/user-attachments/assets/b098aa2f-1757-44b1-ad6c-69a3df0ec1a0)

### 📊**PR_curve**

![390937545-6a3369a8-6569-4314-94a9-8f52489b0c84](https://github.com/user-attachments/assets/ba1984e6-57f4-4b8d-9c57-b365a449b968)

### 📊L**abels**

![390937539-09db7b26-55da-47c6-8047-1dec9f64e4e6](https://github.com/user-attachments/assets/25e3a8d4-e670-4c64-b093-90b51650b791)

### 📊**P_curve**

![390937542-97869cb5-a783-4aa7-92ec-bd42ce40a149](https://github.com/user-attachments/assets/c75d3695-ab3f-48b5-be1e-06fe48d12f73)

### 📊**R_curve**

![390937547-95f6273c-dcd6-49ba-8156-af2d444b7bde](https://github.com/user-attachments/assets/da74791a-d1d2-4467-b3b4-ebbc7c08fba3)

### 📊R**esult**

![390937550-6c8eb81e-4710-46fa-8bc0-39569e220e69](https://github.com/user-attachments/assets/b85e7654-460b-4368-9c5a-2f258de8214d)

### 📊**Train_batch**

![390937551-84b672bd-c5ec-425e-a1e0-5aa8e8c411c3](https://github.com/user-attachments/assets/31773db3-2a51-48ae-b062-ff694cfd3db3)

### 📋**detect.py execution videos**

[https://drive.google.com/drive/folders/1pzZ6P4oq56Pmb_9vfQQ04OOpQF0ULPxG?usp=drive_link](https://drive.google.com/drive/folders/1pzZ6P4oq56Pmb_9vfQQ04OOpQF0ULPxG?usp=drive_link)

### 📊**Val_batch**

![390937555-0206723d-f28a-495b-8064-cfaacc406b69](https://github.com/user-attachments/assets/8da67ebc-6315-4d77-adc0-30ee04df9bd7)

### 📋**detect.py execution images**

[https://drive.google.com/drive/folders/1VQJ1qoCkoHAyNpvtDKmNL90mSEa3USHk?usp=sharing](https://drive.google.com/drive/folders/1VQJ1qoCkoHAyNpvtDKmNL90mSEa3USHk?usp=sharing)

</aside>

## **Significance of this Project**

<aside>
📋

**Collisions with animals pose risks to both humans and wildlife. This project uses YOLOv5 for real-time animal detection, improving autonomous driving safety and reducing accidents. By addressing this critical need, the project supports safer roads and sustainable human-wildlife coexistence.**

</aside>

## **Faced Limitations**

<aside>
📋

**Key limitations included Inadequate animal datasets, high computational demands for training, and detection difficulties in low-light or occluded environments.**

</aside>

## Discussion

<aside>
📋

**Future research directions include: expanding the dataset with diverse animal images from various environments to improve model generalization; integrating thermal cameras and infrared imaging for better detection in low-light and occluded conditions; optimizing the model for improved computational efficiency and real-time performance; and combining data from cameras, LiDAR, and radar for more accurate animal detection.**

</aside>
## report_images : https://github.com/chaesh1201/Nvidia_AISpecialist_Certification/issues/2#issue-2703603563
## Original Link : https://pickle-eagle-bbb.notion.site/YOLOv5-Based-Animal-Detection-for-Safer-Autonomous-Driving-145cb791756080b5a14fffce04965e99?pvs=4
