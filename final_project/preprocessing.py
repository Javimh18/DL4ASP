import os
import shutil
import librosa
import numpy as np
import pickle
import math

class Loader:
    """
    Responsible of loading an audio file.
    """
    
    def __init__(self, sample_rate, duration, mono: bool):
        self.sample_rate = sample_rate
        self.duration = duration
        self.mono = mono
        
    def load(self, file_path):
        # librosa returns a tuple -> signal, sample_rate
        signal = librosa.load(file_path,
                              sr = self.sample_rate,
                              duration = self.duration,
                              mono = self.mono)[0]
        
        return signal

class Padder:
    """
    Responsible to apply padding to an array
    """
    
    def __init__(self, mode="constant") -> None:
        self.mode = mode
        
    def left_pad(self, array, num_missing_items):
        padded_array = np.pad(array,
                              (num_missing_items, 0),
                              mode=self.mode)
        
        return padded_array
    
    def right_pad(self, array, num_missing_items):
        padded_array = np.pad(array,
                              (0, num_missing_items),
                              mode=self.mode)
        
        return padded_array

class LogSpectrogramExtractor:
    """
    Responsible of extracting log-spectrogram in dB from a time-series signal
    """
    
    def __init__(self, frame_size, hop_length):
        self.frame_size = frame_size
        self.hop_length = hop_length

    def extract(self, signal):
        stft = librosa.stft(signal,
                            n_fft=self.frame_size,
                            hop_length=self.hop_length)[:-1] # (1+frame_size / 2, num_frames)
        spectrogram = np.abs(stft)
        log_spectrogram = librosa.amplitude_to_db(spectrogram)
        return log_spectrogram
    
class MinMaxNormaliser:
    """
    Responsible of apply min max normalisation to an array.
    """
    
    def __init__(self, min_val, max_val):
        self.min = min_val
        self.max = max_val
        
    def normalise(self, array):
        norm_array = (array - array.min()) / (array.max() - array.min())
        norm_array = norm_array * (self.max - self.min) + self.min
        return norm_array
    
    def denormalise(self, norm_array, original_min, original_max):
        array = (norm_array - self.min) / (self.max - self.min)
        array = array * (original_max - original_min) + original_min
        return array

class Saver:
    """
    Responsible to save features and the min max values of the log_spectrogram
    """

    def __init__(self, feature_save_dir, save_min_max_values_save_dir):
        self.feature_save_dir = feature_save_dir
        self.save_min_max_values_save_dir = save_min_max_values_save_dir
    
    def save_feature(self, norm_feature, file_path):
        save_path = self._generate_save_path(file_path)
        file_name = os.path.split(save_path)[1]
        np.save(save_path, norm_feature)
        return file_name
        
    def save_min_max_values(self, min_max_values):
        save_path = os.path.join(self.save_min_max_values_save_dir, "min_max_values.pkl")
        self._save(min_max_values, save_path)
        
    def _generate_save_path(self, file_path):
        file_name = os.path.split(file_path)[1]
        save_path = os.path.join(self.feature_save_dir, file_name + ".npy")
        return save_path
    
    @staticmethod
    def _save(data, save_path):
        with open(save_path, "wb") as f:
            pickle.dump(data, f)
        

class PreprocessingPipeline:
    """
    Responsible of processing audio files in a directory, applying the steps:
    1. - load file
    2. - pad the signal
    3. - extract log spectrogram from signal
    4. - normalise spectrogram
    5. - save the normalised spectrogram
    """
    
    def __init__(self):
        self.padder = None
        self.extractor = None
        self.normaliser = None
        self.saver = None
        self.min_max_values = {}
        self._loader = None
        self._num_expected_samples = None
        
    @property
    def loader(self):
        return self._loader
    
    @loader.setter
    def loader(self, loader):
        self._loader = loader
        self._num_expected_samples = int(loader.sample_rate * loader.duration)
        
    def process(self, audio_files_dir):
        for root, _, files in os.walk(audio_files_dir):
            for file in files:
                file_path = os.path.join(root, file)
                self._process_file(file_path)
                print(f"Processed file: {file_path}")
                
        self.saver.save_min_max_values(self.min_max_values)
                
    def _process_file(self, file_path):
        signal = self.loader.load(file_path)
        if self._is_padding_necessary(signal):
            signal = self._apply_padding(signal)
            
        feature = self.extractor.extract(signal)
        norm_feature = self.normaliser.normalise(feature)
        save_path = self.saver.save_feature(norm_feature, file_path)
        self._store_min_max_value(save_path, feature.min(), feature.max())
        
    def _is_padding_necessary(self, signal):
        if len(signal) < self._num_expected_samples:
            return True
        return False
    
    def _apply_padding(self, signal):
        num_missing_samples = self._num_expected_samples - len(signal)
        padded_signal = self.padder.right_pad(signal, num_missing_samples)
        return padded_signal
    
    def _store_min_max_value(self, save_path, min_val, max_val):
        self.min_max_values[save_path] = {
            "min": min_val,
            "max": max_val
        }
        
if __name__ == "__main__":
    FRAME_SIZE = 512
    HOP_LENGTH = 256
    DURATION = 0.74
    SAMPLE_RATE = 22050
    MONO = True
    SPLIT_TEST_TRAIN = 0.8
    TOP_SAMPLE = 49
    
    SPECTROGRAMES_SAVE_DIR = './data/real/spectrograms/'
    MIN_MAX_VALUES_SAVE_DIR = './data/real'
    FILES_DIR = './data/real/audio/'
    
    if not os.path.exists(SPECTROGRAMES_SAVE_DIR):
        os.makedirs(SPECTROGRAMES_SAVE_DIR)
    
    loader = Loader(SAMPLE_RATE, DURATION, MONO)
    padder = Padder()
    log_spectrogram_extractor = LogSpectrogramExtractor(FRAME_SIZE, HOP_LENGTH)
    min_max_normaliser = MinMaxNormaliser(0, 1)
    saver = Saver(SPECTROGRAMES_SAVE_DIR, MIN_MAX_VALUES_SAVE_DIR)
    preprocessing_pipeline = PreprocessingPipeline()
    preprocessing_pipeline.loader = loader
    preprocessing_pipeline.padder = padder
    preprocessing_pipeline.extractor = log_spectrogram_extractor
    preprocessing_pipeline.normaliser = min_max_normaliser
    preprocessing_pipeline.saver = saver
    
    preprocessing_pipeline.process(FILES_DIR)
    
    # TRAIN-TEST split
    list_of_spectrograms = os.listdir(SPECTROGRAMES_SAVE_DIR)
    idx_train_test_split = math.floor(TOP_SAMPLE*SPLIT_TEST_TRAIN)
    for s in list_of_spectrograms:
        path_spectrogram = os.path.join(SPECTROGRAMES_SAVE_DIR, s)
        idx_sample = int(s.split("_")[2].split('.')[0])
        if idx_sample < idx_train_test_split:
            train_path = os.path.join(SPECTROGRAMES_SAVE_DIR, "train")
            if not os.path.exists(train_path):
                os.makedirs(train_path)
            shutil.move(path_spectrogram, train_path)
        else:
            test_path = os.path.join(SPECTROGRAMES_SAVE_DIR, "test")
            if not os.path.exists(test_path):
                os.makedirs(test_path)
            shutil.move(path_spectrogram, test_path)
        
    # TRAIN-VAL split
    train_path = os.path.join(SPECTROGRAMES_SAVE_DIR, "train")
    list_of_spectrograms = os.listdir(train_path)
    idx_train_val_split = math.floor(idx_train_test_split*SPLIT_TEST_TRAIN)
    for s in list_of_spectrograms:
        path_spectrogram = os.path.join(train_path, s)
        idx_sample = int(s.split("_")[2].split('.')[0])
        if idx_sample > idx_train_val_split:
            val_path = os.path.join(SPECTROGRAMES_SAVE_DIR, "val")
            if not os.path.exists(val_path):
                os.makedirs(val_path)
            shutil.move(path_spectrogram, val_path)
            
    
    
    
    