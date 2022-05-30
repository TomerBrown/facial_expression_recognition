# Psychology Seminar at Galit Yovel's Lab

### Setup:
* **Download stuff**
    * all of our resources can be found in the [following link](https://drive.google.com/drive/folders/1B6PIZww-UtlqpLhoG2euaS-T9qQMgQX9?usp=sharing)
* **Model**:
    * The model can be download via the  link and should be saved in models directory with name VGGNet , e.g. ```/models/VGGNet```.
    * The model was taken [from here](https://github.com/usef-kh/fer)
* **Dataset**:
     * The database and csv file should be downloaded from link above.
     * The folders structure should be:
        * CSV file - (outside the KDEF folder but inside datasets)
            * `/datasets/KDEF.CSV `

        * All images (KDEF folder inside it all the images). for example:
            * `/datasets/KDEF/AF01`    

### Important Files:
*  #### expermints.py: 
    * A file to define each experiment we want to run (currently only split by identity exists)
    * Every experiment should be defined as function that returns 4 things:
        1) **train_df** (What the model should be trained on)
        2) **val_df** (Validation Set)
        3) **Test_df** (What the model should be tested on).
        4) **name** - should be unique and descriptive of what we do.
        note: the returned items should be pandas.DataFrame (for easy building of experiments)
    
*   #### finetune.py
    * the training loop, taking the pre-trained VGG-net model and fine tune it for n_epochs(parameters) epochs.
    * after ~70 epochs it doesn't improve sagnificantly.
    * Achieves 80% accuracy on test set

*  #### evaluate.py
    * The evaluation on the test set while keeping the predicitions made by the model for later analysis.