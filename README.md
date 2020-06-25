# BassUNet

Algorithm for bass transcription (joint frame-level pitch and voicing estimation) using U-Net Fully Convolutional Networks

  - J. Abeßer & M. Müller: BassUNet: Jazz Bass Transcription using a U-Net Architecture, ISMIR 2020


We recommend you to install *miniconda* (https://conda.io/miniconda.html).
You can create a suitable environment using
```
conda env create -f conda_env.yml
```
and activated it via
```
source activate bassunet
```

## example

Now you can run the transcription algorithm on a test file by calling
```
python transcriber.py
```

After running the transcriber on the test file ```ArtPepper_Anthropology_Excerpt.wav```, 
the frame-level pitch estimates as well as the estimated notes are stored in two CSV files.
Both can be imported into Sonic Visualiser as time-instance and note layers.
You can open the example SV project ```ArtPepper_Anthropology_Excerpt.sv``` for an example.

Enjoy.

