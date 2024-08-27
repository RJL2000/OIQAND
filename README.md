# Subjective and Objective Quality Assessment of Non-Uniformly Distorted Omnidirectional Images
[Jiebin Yan], [Jiale Rao], [Xuelin Liu] [Yuming Fang] [Yifan Zuo]  [Weide Liu]

# JUFE-10K


## :book:Model Architecture
![image.png](image/model.jpg)


## :hammer_and_wrench: Usage

If you want to train the code on your database:

First, prepare the database
```sh

```
Then
```sh

```
If you want to test the trained model on the test set:

```sh

```

### Training OIQAND
- Modify "dataset_name" to choose which datasets you want to train in config
- Modify training and validation dataset path

model_name and type_name is the file path for saving checkpoint and log file
```
python train_oiqa.py
```


