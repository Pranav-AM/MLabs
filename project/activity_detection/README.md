# Activity Detection module 

* To train the model, execute this command
  ```bash
   $ python main.py
   ```
   The trained model will be saved as MLabs/data/act_rec_lstm.h5
   
* To run the pre-trained model,
  1) Change **TRAINED_MODEL = TRUE** in **config.ini** to **TRAINED_MODEL = FALSE**
  2) Run the commmand
      ```bash
      $ python main.py
      ```
* To run the activity detection module on a new activity sequence,
  1) Change **line 12** in **demo.py** to another sequence
  2) Run the command
      ```bash
      $ python demo.py
      ```
  
  
  

