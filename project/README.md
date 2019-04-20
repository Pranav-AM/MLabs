* Run the project code
    ```bash
    $ python main.py
    ```
* The output images are saved at MLabs/data/images

* NOTE: It is necessary to perform the following before running object detection code:
1. Run setup.py before running object detection code as:
    ```bash
    $ python setup.py build
    $ python setup.py install #Optional    
    ```
2. Add the relevant directories to PYTHONPATH when running locally as:
    ```bash
    $ export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
    ```

