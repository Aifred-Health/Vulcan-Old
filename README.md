# aifred
Clinical Decision Aid for Increased Treatment Efficacy

Mental health is an integral part of our well-being and how we function within society. aifred aims to bring personalized treatments at scale by combining best practices from psychiatry and deep learning.

## Dependencies
 Simply run:
 ```
 pip install -r requirements.txt
 ```

## Running the scripts:
 To train the studies on gender and predict the class run:
 ```
 python train_citalopram_gender.py
 ```
 or
  ```
 python train_duloxetine_gender.py
 ```

 To train the studies on remission and predict the class run:
  ```
 python train_citalopram_remission.py
 ```
 or
  ```
 python train_duloxetine_remission.py
 ```