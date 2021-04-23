# BassUNet

## Reference

Algorithm for bass transcription (joint frame-level pitch and voicing estimation) using U-Net Fully Convolutional Networks

  - J. Abeßer & M. Müller: BassUNet: Jazz Bass Transcription using a U-Net Architecture, 
  Electronics, 10(6), 2021 
    - https://www.mdpi.com/2079-9292/10/6/670

## Run
We recommend you to install *miniconda* (https://conda.io/miniconda.html).
You can create a suitable environment using
```
conda create --name bassunet python="3.6"
conda activate bassunet
pip install librosa tensorflow==1.15 "h5py<3.0.0"
```

You can run the bass transcription algorithm as shown in the ``bassunet.py`` file:

```
bun = BassUNet()
t, f0, onset, duration, pitch = bun.run(wav_file_name)
```

## Example

Now you can run the transcription algorithm on a test file by calling
```
python bassunet.py
```

After running the transcriber on the test file ```ArtPepper_Anthropology_Excerpt.wav```, 
the frame-level pitch estimates as well as the estimated note events are stored in two CSV files.
Both can be imported into Sonic Visualiser as time-instance and note layers, respectively.

You can open the example SV project ```ArtPepper_Anthropology_Excerpt.sv``` for an example.

Enjoy.

