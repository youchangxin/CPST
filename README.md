
## Getting Started

### Prerequisites

Python 3.6 or above.

PyTorch 1.6 or above

For packages, see requirements.txt.

  ```sh
  pip install -r requirements.txt
  ```

<p align="right">(<a href="#top">back to top</a>)</p>

### Installation

   Clone the repo
   ```sh
   git clone https://github.com/zyxElsa/CAST_pytorch.git
   ```

<p align="right">(<a href="#top">back to top</a>)</p>

### Datasets

   Then put your content images in ./datasets/{datasets_name}/testA, and style images in ./datasets/{datasets_name}/testB.
   
   Example directory hierarchy:
   ```sh
      CAST_pytorch
      |--- datasets
             |--- {datasets_name}
                   |--- trainA
                   |--- trainB
                   |--- testA
                   |--- testB
                   
      Then, call --dataroot ./datasets/{datasets_name}
   ```

<p align="right">(<a href="#top">back to top</a>)</p>

### Train

   Train the CAST model:
   ```sh
   python train.py --dataroot ./datasets/{dataset_name} --name {model_name}
   ```
   
   The pretrained style classification model is saved at ./models/style_vgg.pth.
   
   Google Drive: Check [here](https://drive.google.com/file/d/12JKlL6QsVWkz6Dag54K59PAZigFBS6PQ/view?usp=sharing)
   
   The pretrained content encoder is saved at ./models/vgg_normalised.pth.
   
   Google Drive: Check [here](https://drive.google.com/file/d/1DKYRWJUKbmrvEba56tuihy1N6VrNZFwl/view?usp=sharing)
   
<p align="right">(<a href="#top">back to top</a>)</p>

### Test

   Test the CAST model:
   
   ```sh
   python test.py --dataroot ./datasets/{dataset_name} --name {model_name}
   ```
   
   The pretrained model is saved at ./checkpoints/CAST_model/*.pth.

