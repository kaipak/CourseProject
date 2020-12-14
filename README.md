# CourseProject - BERT For 100% Accuracy (Sarcasm)
This readme serves as technical documentation and overview of the
project code. Please review the PDF included in this repository
for more detailed information on the BERT model, results, and 
technical details.

## Technical Requirements
The code has been tested on Python 3.8.x and requires the following
core packages.

- torch >= 1.5
- transformers >= 3.5.1
- tensorflow >= 2.3.1
- pandas >= 1.1.4
- numpy >= 1.18.5
- scikit-learn
- matplotlib
- seaborn

It is highly recommended to use virtual environments for running
this code, or on a dedicated cloud environment such as Google
Colab. 

## Running in Google Colab
This is probably the easiest way to get things going. The 
`TeamKayak_SarcasmDetectorDemo.ipynb` notebook located in this repo
gives a good overview of the environment set up steps you'll need to
run this repo. 

## Install required packages
You will want to be able to sync to Google Drive so you can
check out this repo and have it available to Colab. Running the cell
below will install the Google package and mount your GDrive. Then,
run the sample `git clone` command replacing `my_directory` to a 
location of your convenience.

```
from google.colab import drive
drive.mount('/content/drive', force_remount=True)
!git clone https://github.com/kaipak/CourseProject.git my_directory
!pip install transformer
```

## Set up Environment
You will need to set up some directories in your GDrive to hold 
the input datasets and so the code can also output processed data,
model checkpoints, metrics, etc.

You can use the following cell as a template.

```
# Set Paths for Google Drive
source_folder = '/content/drive/My Drive/Data'
destination_folder = '/content/drive/My Drive/Model/Response'
code_folder = '/content/drive/My Drive/CourseProject/src'
train_log_dir = '/content/drive/My Drive/logs/tensorboard/train/'
```

Then, import needed packages to your notebook:

```
import sys, os
from pathlib import Path
import random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(code_folder)
from model import SarcasmDetector
from data_prep import DataPrep
```

## Prepare Data
The input data for the model can be processed using our `data_prep`
class:

```
# Prepare Data
data_prepper = DataPrep(train_path=source_folder + '/train.jsonl',
                       sub_path=source_folder + '/test.jsonl',
                       response_only=False)
data_prepper.train_test_split()
data_prepper.write_data(datapath=source_folder)
```

Most of the parameters have sane default values, but you can see
the docstrings to alter if you like.

## Train a Model
The model should be ready to train now.

```
### Instantiate Model
SarcModel = SarcasmDetector(input_dir=source_folder, 
                            output_dir=destination_folder
                            train_log_dir = train_log_dir
                            
```

### Tokenize text
Then, tokenize the input data:

```
SarcModel.tokenize_data(train_fname = 'train.csv', 
                        validate_fname = 'validate.csv', 
                        test_fname = 'test.csv', 
                        batch_size = 8
                        
```

### Start up Tensorboard
This will give you feedback on how training is going. Run the 
magic command.

```
%tensorboard --logdir
```

### Train!
There's a `tune()` method that does grid search to find best 
set of hyperparameters but just running on lists of singletons is
essesntially as running `train()`.

```
# learning rate list
lr_list = [5e-7]

# number epochs list
num_epochs_list = [15]

# tune hyperparameters
SarcModel.tune(lr_list=lr_list, num_epochs_list=num_epochs_list
```

### Evaluate
You can then run the trained model against the test set to get
some metrics on how well it does on unseen observations.

```
SarcModel.evaluate(model_name = 'lr_5e-07_epochs_15_')
```



