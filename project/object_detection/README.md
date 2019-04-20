# Object Detection module 

* **NOTE**: Please run setup.py and add the project directory to PYTHONPATH (as explained in [MLabs/project/README.md](../README.md)) before commencing.

# Run Pre-trained Model:

* To run the pre-trained model, run main.py as:
  ```bash
   $ python main.py
   ```

# Train and Run New Model:

* To convert the annotated data from .txt format to .csv, run txt2csv.py as:
  ```bash
   $ python txt2csv.py
   ```
* To generate train and test records from the csv files, run generate_tfrecord.py as:
  ```bash
   $ python generate_tfrecord.py
   ```
* To train the model, run legacy/train.py as:
  ```bash
   $ python legacy/train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/ssd_mobilenet_v1_pets.config
   ```
  The training checkpoints will be saved at [MLabs/project/object_detection/training](./training).

* To generate the model file at output directory [MLabs/project/object_detection/inference_graph](./inference_graph), run export_inference_graph.py as:
  ```bash
   $ python export_inference_graph.py     
     --input_type image_tensor     
     --pipeline_config_path training/ssd_mobilenet_v1_pets.config     
     --trained_checkpoint_prefix training/model.ckpt-xx     
     --output_directory inference_graph
   ```
   (Here, xx in --trained_checkpoint_prefix training/model.ckpt-xx stands for the number of the latest training checkpoint in the [MLabs/project/object_detection/training](./training) folder.)

* To run the newly trained model, change MODEL_NAME in line 64 of main.py to 'inference_graph' and run main.py as:
  ```bash
   $ python main.py
   ```
