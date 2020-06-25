import os
import librosa
import csv
import numpy as np
from tensorflow.keras.models import load_model
import time
import glob


class BassUNet:
    """ Class implements the inference using the BassUNet model for bass transcription
    Reference:  [1] J. Abesser & M. Mueller: BassUNet: Jazz Bass Transcription using a U-Net
                Architecture, ISMIR 2020
    """
    def __init__(self,
                 fn_model="basssegnet_mixed.h5",
                 verbose=True):
        """ Initialize class
        Args:
            fn_model (string): Model file name
                'basssegnet_mixed.h5': Model "BassUNet^M" (for mixed music genres)
                'basssegnet_jazz.h5': Model "BassUNet^J" (for jazz)
        """
        assert fn_model in ('basssegnet_mixed.h5', 'basssegnet_jazz.h5'), 'Non-valid model file name!'
        # define processing parameters as in [1]
        self.__fs = 22050.
        self.__hop_len = 512
        self.__bpo = 12
        midi_lo, midi_hi = 25, 88
        self.__midi_range = np.arange(midi_lo, midi_hi+1)
        self.__midi_range_plus_unvoiced = np.append(self.__midi_range, -1)
        self.__n_bins = len(self.__midi_range)
        self.__fmin = 440.*2**((self.__midi_range[0]-69)/12.)
        self.__model = load_model(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data', fn_model))
        self.__verbose = verbose

    def run(self, fn_wav, min_note_len=2):
        """ Transcribe WAV file
        Args:
            fn_wav (string): WAV file name
            min_note_len (int): Minimum note duration in frames (one frame = 23.2 ms)
        Returns:
            time_axis_sec (np.ndarray): Time axis in seconds (n_frames)
            est_freq (np.ndarray): Estimated bass frequencies in Hz (0 = unvoiced frames) (n_frames)
            onset (np.ndarray): Note onset times (n_notes)
            duration (np.ndarray): Note durations (n_notes)
            pitch (np.ndarray): Note pitch values (n_notes)
        """
        if self.__verbose:
            t1 = time.time()
        # load WAV file, enforce sample rate
        x, fs = librosa.load(fn_wav, sr=self.__fs, mono=True)
        # extract features
        feat = self.audio_to_feat(x, fs)
        # get model predictions
        pred = self.__model.predict(feat)
        target_pred = pred[0, :, :]
        # get frame-wise f0 estimates
        est_freq = np.zeros(target_pred.shape[0])
        est_pitch = self.__midi_range_plus_unvoiced[np.argmax(target_pred, axis=1)]
        est_freq[est_pitch > 0] = 440. * 2 ** ((est_pitch[est_pitch > 0] - 69) / 12)
        time_axis_sec = np.arange(len(est_freq)) * self.__hop_len/self.__fs
        # note formation
        onset, duration, pitch = self.note_formation(est_pitch, time_axis_sec, min_note_len=min_note_len)
        if self.__verbose:
            print('Transcription of {:2.2f}s audio in {:2.2f}s'.format(len(x)/fs, time.time()-t1))
        return time_axis_sec, est_freq, onset, duration, pitch

    def audio_to_feat(self, x, fs):
        """ Extract spectrogram features for model input from waveform
        Args:
            x (np.ndarray): Audio samples
            fs (float): Sample rate in Hz
        Returns:
            feat (4D np.ndarray): Tensor input for BassSegNet model
        """
        cqt = np.abs(librosa.core.cqt(x, sr=fs, hop_length=self.__hop_len, fmin=self.__fmin,
                                      bins_per_octave=self.__bpo, n_bins=self.__n_bins))
        feat = cqt.astype(np.float16).T
        feat -= np.min(feat)
        feat /= np.max(feat)
        feat = np.expand_dims(feat, 0)
        feat = np.expand_dims(feat, -1)
        return feat

    @staticmethod
    def add_new_note(onset, offset, pitch, curr_frame, curr_pitch):
        """ Add a new note candidate to the note lists
        Args:
            onset (list): List of onset candidates
            offset (list): List of offset candidates
            pitch (list): List of pitch candidates
        Returns:
            curr_frame (int): Current onset frame index
            curr_pitch (int): Current MIDI pitch candidate
        """
        onset.append(curr_frame)
        offset.append(curr_frame)
        pitch.append(curr_pitch)
        return onset, offset, pitch

    def note_formation(self, est_pitch, time_axis, pitch_unvoiced=-1, min_note_len=2):
        """ Simple note formation approach to convert frame-wise f0 estimates with voicing
            to list of note events. Note that this cannot resolve multiple successive notes
            with the same pitch (and no unvoiced frame between).
        Args:
            est_pitch (np.ndarray): Frame-level pitch estimates
            time_axis (np.ndarray): Frame times in seconds
            pitch_unvoiced (int): MIDI pitch value in est_pitch that indicates unvoiced frames
            min_note_len (int): Minimum duration thresholds to consider a note candidate
                                in the final set of notes
        Returns:
            onset (np.ndarray): Note onset times
            duration (np.ndarray): Note durations
            pitch (np.ndarray): Note pitches
        """
        dt = time_axis[1] - time_axis[0]
        onset, offset, pitch = [], [], []
        n_frames = len(est_pitch)
        prev_frame_pitch = pitch_unvoiced
        # iterate over frames
        for i in range(n_frames):
            if est_pitch[i] != pitch_unvoiced:
                # (1) voiced frame
                if prev_frame_pitch == pitch_unvoiced:
                    # (1.1) previous frame was unvoiced -> create a new note
                    onset, offset, pitch = BassUNet.add_new_note(onset, offset, pitch, i, est_pitch[i])
                    prev_frame_pitch = est_pitch[i]
                else:
                    # (1.2.) previous frame was voiced
                    if est_pitch[i] == prev_frame_pitch:
                        # (1.2.1.) continue note (increase offset frame)
                        offset[-1] = i
                    else:
                        # (1.2.2.) create new note
                        onset, offset, pitch = BassUNet.add_new_note(onset, offset, pitch, i, est_pitch[i])
                        prev_frame_pitch = est_pitch[i]
            else:
                # (2) unvoiced frame
                prev_frame_pitch = pitch_unvoiced

        onset = np.array(onset, dtype=np.int32)
        offset = np.array(offset, dtype=np.int32)
        pitch = np.array(pitch, dtype=np.int32)

        duration = offset+1-onset

        # remove too short notes
        filter = duration >= min_note_len
        duration = duration[filter]
        onset = onset[filter]
        pitch = pitch[filter]
        if self.__verbose:
            print(np.sum(filter), '/', len(filter), ' notes kept')

        duration = duration.astype(np.float16) * dt
        onset = time_axis[onset]

        return onset, duration, pitch

    @staticmethod
    def export_pitch_track_as_csv(t, f0, fn_csv):
        """ Export pitch track as two-column CSV file to be importable in Sonic Visualiser
        Args:
            t (np.ndarray): Time values in seconds
            f0 (np.ndarray): Frame-level f0 estimates (0 = unvoiced frame) in Hz
            fn_csv (string): CSV file name
        """
        with open(fn_csv, 'w') as f:
            writer = csv.writer(f, delimiter=',')
            for i in range(len(t)):
                writer.writerow([t[i], f0[i]])

    @staticmethod
    def export_notes_as_csv(onset, duration, pitch, fn_csv):
        """ Export bass notes as three-column CSV file to be imported in SV as note layer
        Args:
            onset (np.ndarray): Note onset times
            duration (np.ndarray): Note durations
            pitch (np.ndarray): Note pitches
            fn_csv (string): CSV file name
            """
        with open(fn_csv, 'w') as f:
            writer = csv.writer(f, delimiter=',')
            for i in range(len(onset)):
                writer.writerow([onset[i], duration[i], pitch[i]])

if __name__ == '__main__':
    fn_wav = os.path.join(os.path.dirname(__file__), 'data','ArtPepper_Anthropology_Excerpt.wav')
    bun = BassUNet()
    # transcription
    t, f0, onset, duration, pitch = bun.run(fn_wav)

    # export pitch track
    fn_csv = fn_wav.replace('.wav', '_bass_f0.csv')
    BassUNet.export_pitch_track_as_csv(t, f0, fn_csv)
    print(fn_csv, ' saved!')

    # export notes
    fn_csv = fn_wav.replace('.wav', '_bass_notes.csv')
    BassUNet.export_notes_as_csv(onset, duration, pitch, fn_csv)
    print(fn_csv, ' saved!')