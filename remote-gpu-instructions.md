# Remote GPU instructions

### HPC

* The steps to run on HPC are largely the same, only the environment setup and resource selection is different. See [this repo](https://github.com/cvalenzuela/hpc) to get started with HPC

* Log in and clone repo

  ```
  git clone https://github.com/ml5js/training-dcgan
  cd training-dcgan
  ```

* Transfer your dataset to the `images` folder

* Make an environment

  ```
  module load python3/intel/3.6.3
  virtualenv -p python3 $HOME/dcgan-env
  source $HOME/dcgan-env/bin/activate
  pip install tensorflowjs pillow chainer absl-py cupy-cuda100
  module unload python3/intel/3.6.3
  ```

* The steps below demonstrate interactive runs for 64px images, but **these jobs should be queued if your dataset is large and/or you're using 128px or 256px images**

  - Pre-process data

    ```
    srun -c2 -t 1:00:00 --pty /bin/bash 		
    source $HOME/dcgan-env/bin/activate
    cd PATH/TO/REPO/training-dcgan/
    ./datatool.py --size 64
    ```

  * Train

    ```
    srun -c2 -t 5:00:00 --gres=gpu:1 --mem=64GB --pty /bin/bash
    module load cuda/10.0.130  cudnn/10.0v7.4.2.24
    source $HOME/dcgan-env/bin/activate
    cd PATH/TO/REPO/training-dcgan/
    ./chainer_dcgan.py --arch dcgan64 --image_size 64
    ```

  * Convert Model

    ```
    srun -c2 -t 1:00:00 --pty /bin/bash 		
    source $HOME/dcgan-env/bin/activate
    cd PATH/TO/REPO/training-dcgan/
    ./dcgan_chainer_to_keras.py --arch dcgan64 --chainer_model_path result/SmoothedGenerator_XXXX.npz
    ```

* Depending on the size of your dataset and the `image_size` and architecture you decide to use, more time and resources could be required. Pre-processing could take a few hours, and training could take a couple of days. You might need to increase the memory for training (`—mem`) and the number of cpus (`—c`)



### COLAB

* The steps for Google Colab differ mainly in how data is uploaded to Colab

* Make a new Python 3 notebook

* On the top menu, go to Runtime > Change Runtime Type  and select GPU from the Hardware accelerator menu

* Clone the repo (Colab provides you with a Python notebook - to execute command line commands, put an `!` before the command. )

  ```
  !git clone https://github.com/ml5js/training-dcgan
  ```

* To upload image to colab, create an archive on your machine first (assuming images are in a folder called `dataset`)

  ```
  tar -cvzf dataset.tar.gz dataset
  ```

* On Colab, expand the menu on the left and go the the Files tab and hit upload, and upload your `.tar.gz` file

* Extract the contents on Colab

  ```
  !tar -C training-dcgan/images/ -xf dataset.tar.gz --strip-components=1
  ```

* Change directory

  ```
  %cd training-dcgan
  ```

* The rest of the steps are the same as the main training instructions — just remember to prepend a `!`  to the commands





### SPELL

* [Set up an account on Spell](https://spell.run/docs/quickstart)

* Upload data (can take a little bit of time)

  ```
  spell upload -c ./images/
  ```

* The remaining steps are similar. Make note of the run number of each step, it will be used in the step that follows.

* Clone the repo

  ```
  git clone https://github.com/ml5js/training-dcgan
  cd training-dcgan
  ```

* Train. We only need to use a GPU for the training step.

* ```
  spell run -m uploads/images:images "./datatool.py --size 64"       
  ```

  Take note of the run number — we'll need that to use the output from this stage in the training stage. The `-t` flag is used to specify the GPU type — while the K80 is cheaper, it is slower as well.

  ```
  spell run -m runs/RUN_NUMBER/dataset.npz:dataset.npz --pip-req requirements.txt -t K80 "./chainer_dcgan.py --arch dcgan64 --image_size 64"
  ```

  Take note of this run number as well, and mount the results to convert the model

  ```
  spell run -m runs/RUN_NUMBER/result:result --pip-req requirements.txt "./dcgan_chainer_to_keras.py --arch dcgan64 --chainer_model_path result/SmoothedGenerator_XXXXX.npz"
  ```

  Download the converted model from spell

  ```
  spell cp runs/RUN_NUMBER/result/SmoothedGenerator_XXXXX_tfjs ./my-tfjs-model
  ```

  
