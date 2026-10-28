# ============================================================================
# preprocess.py - EEG Signal Preprocessing Pipeline
# ============================================================================

import numpy as np
import mne
import os
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION (From your Colab notebook)
# ============================================================================

# EEG channel names (19 channels - exact from your notebook)
EEG_CHANNELS = [
    "EEG Fp1-LE",  "EEG F3-LE",  "EEG C3-LE",  "EEG P3-LE",  "EEG O1-LE",
    "EEG F7-LE",   "EEG T3-LE",  "EEG T5-LE",  "EEG Fz-LE",  "EEG Fp2-LE",
    "EEG F4-LE",   "EEG C4-LE",  "EEG P4-LE",  "EEG O2-LE",  "EEG F8-LE",
    "EEG T4-LE",   "EEG T6-LE",  "EEG Cz-LE",  "EEG Pz-LE"
]

# Filtering parameters
LOW_FREQ = 1      # Hz
HIGH_FREQ = 40    # Hz

# Window parameters
WINDOW_SECONDS = 4
OVERLAP = 0.5

# ============================================================================
# CLASS: EEGProcessor
# ============================================================================

class EEGProcessor:
    """
    Complete EEG preprocessing pipeline (from your Colab notebook).
    
    Pipeline:
    1. Load EDF file using MNE
    2. Select 19 EEG channels
    3. Apply band-pass filter (1-40 Hz)
    4. Normalize per channel (z-score)
    5. Create 4-second overlapping windows
    6. Add channel dimension for model input
    
    Returns: numpy array of shape (num_windows, 19, 1024, 1)
    """
    
    def __init__(self, low_freq=LOW_FREQ, high_freq=HIGH_FREQ,
                 window_seconds=WINDOW_SECONDS, overlap=OVERLAP):
        """
        Initialize EEG processor with filtering and windowing parameters.
        
        Args:
            low_freq (int): Low cutoff frequency in Hz
            high_freq (int): High cutoff frequency in Hz
            window_seconds (int): Window duration in seconds
            overlap (float): Overlap ratio between windows (0-1)
        """
        self.low_freq = low_freq
        self.high_freq = high_freq
        self.window_seconds = window_seconds
        self.overlap = overlap
        
        logger.info(f"EEGProcessor initialized:")
        logger.info(f"  Filter: {low_freq}-{high_freq} Hz")
        logger.info(f"  Window: {window_seconds}s, Overlap: {overlap*100}%")
    
    def load_edf(self, file_path):
        """
        Load EDF file using MNE.
        
        Args:
            file_path (str): Path to EDF file
            
        Returns:
            mne.io.Raw: Raw EEG data object
        """
        try:
            logger.info(f"Loading EDF file: {file_path}")
            
            raw = mne.io.read_raw_edf(
                file_path,
                preload=True,
                verbose=False
            )
            
            logger.info(f"  ✓ Loaded successfully")
            logger.info(f"  Channels: {len(raw.ch_names)}")
            logger.info(f"  Sampling rate: {raw.info['sfreq']} Hz")
            logger.info(f"  Duration: {raw.times[-1]:.1f} seconds")
            
            return raw
        
        except FileNotFoundError:
            logger.error(f"❌ File not found: {file_path}")
            raise
        except Exception as e:
            logger.error(f"❌ Error loading EDF: {e}")
            raise
    
    def select_channels(self, raw):
        """
        Select 19 standard EEG channels.
        
        Args:
            raw (mne.io.Raw): Raw EEG data
            
        Returns:
            mne.io.Raw: Data with selected channels only
        """
        # Find available channels that match our list
        available_channels = [ch for ch in EEG_CHANNELS if ch in raw.ch_names]
        
        logger.info(f"Channel selection:")
        logger.info(f"  Available: {len(available_channels)}/19")
        
        if len(available_channels) < 15:
            logger.warning(f"⚠️ Only {len(available_channels)} channels found (need at least 15)")
        
        # Select channels
        raw.pick(available_channels)
        
        logger.info(f"  ✓ Selected {len(raw.ch_names)} channels")
        
        return raw
    
    def apply_filter(self, raw):
        """
        Apply band-pass filter (1-40 Hz).
        
        Args:
            raw (mne.io.Raw): Raw EEG data
            
        Returns:
            mne.io.Raw: Filtered EEG data
        """
        try:
            logger.info(f"Applying band-pass filter: {self.low_freq}-{self.high_freq} Hz")
            
            raw.filter(
                l_freq=self.low_freq,
                h_freq=self.high_freq,
                verbose=False
            )
            
            logger.info(f"  ✓ Filter applied successfully")
            return raw
        
        except Exception as e:
            logger.error(f"❌ Error applying filter: {e}")
            raise
    
    def normalize_signal(self, data):
        """
        Normalize EEG signal using z-score per channel.
        
        Args:
            data (np.ndarray): Raw EEG data (num_channels, num_samples)
            
        Returns:
            np.ndarray: Normalized data (same shape)
        """
        # Calculate mean and std per channel
        mean = np.mean(data, axis=1, keepdims=True)
        std = np.std(data, axis=1, keepdims=True)
        
        # Z-score normalization (avoid division by zero)
        normalized = (data - mean) / (std + 1e-8)
        
        logger.info(f"Signal normalized (z-score per channel)")
        
        return normalized
    
    def create_windows(self, data, sfreq):
        """
        Create overlapping 4-second windows from continuous EEG signal.
        
        Window size: 4 seconds × 256 Hz = 1024 samples
        Overlap: 50% = 512 sample step
        
        Args:
            data (np.ndarray): EEG data (num_channels, num_samples)
            sfreq (int): Sampling frequency in Hz
            
        Returns:
            np.ndarray: Windows of shape (num_windows, num_channels, 1024)
        """
        window_size = self.window_seconds * sfreq
        step_size = int(window_size * (1 - self.overlap))
        
        logger.info(f"Creating windows:")
        logger.info(f"  Window size: {window_size} samples ({self.window_seconds}s @ {sfreq}Hz)")
        logger.info(f"  Step size: {step_size} samples ({self.overlap*100}% overlap)")
        
        windows = []
        num_samples = data.shape[1]
        
        # Slide window across signal
        for start in range(0, num_samples - window_size + 1, step_size):
            end = start + window_size
            window = data[:, start:end]  # Shape: (num_channels, 1024)
            windows.append(window)
        
        windows = np.array(windows)  # Shape: (num_windows, num_channels, 1024)
        
        # Keep as (num_windows, num_channels, 1024) -- this matches the
        # model's expected input order (channels, time_samples), so no
        # transpose here. (Previously this was transposed to
        # (num_windows, 1024, num_channels), which was the wrong order
        # for this model and caused a shape mismatch at prediction time.)
        
        logger.info(f"  ✓ Created {len(windows)} windows")
        
        return windows
    
    def add_channel_dimension(self, data):
        """
        Add channel dimension for EEGNet model input.
        
        Input shape:  (num_windows, 19, 1024)
        Output shape: (num_windows, 19, 1024, 1)
        
        Args:
            data (np.ndarray): Windows (num_windows, num_channels, 1024)
            
        Returns:
            np.ndarray: Data with channel dimension
        """
        data = np.expand_dims(data, axis=-1)
        
        logger.info(f"Added channel dimension")
        logger.info(f"  Final shape: {data.shape}")
        
        return data
    
    def process_file(self, file_path):
        """
        Complete preprocessing pipeline for a single EDF file.
        
        Args:
            file_path (str): Path to EDF file
            
        Returns:
            tuple: (processed_data, sampling_rate)
                - processed_data: shape (num_windows, 19, 1024, 1)
                - sampling_rate: sampling frequency in Hz
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing: {os.path.basename(file_path)}")
        logger.info(f"{'='*60}")
        
        try:
            # Step 1: Load EDF
            raw = self.load_edf(file_path)
            sfreq = int(raw.info['sfreq'])
            
            # Step 2: Select channels
            raw = self.select_channels(raw)
            
            # Step 3: Apply filter
            raw = self.apply_filter(raw)
            
            # Step 4: Get data and normalize
            data = raw.get_data()  # Shape: (num_channels, num_samples)
            data = self.normalize_signal(data)
            
            # Step 5: Create windows
            windows = self.create_windows(data, sfreq)
            
            # Step 6: Add channel dimension
            windows = self.add_channel_dimension(windows)
            
            logger.info(f"✓ Processing complete!")
            logger.info(f"  Output shape: {windows.shape}")
            logger.info(f"{'='*60}\n")
            
            return windows, sfreq
        
        except Exception as e:
            logger.error(f"❌ Processing failed: {e}")
            raise


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def validate_edf_file(file_path):
    """
    Validate if an EDF file is valid and contains required channels.
    
    Args:
        file_path (str): Path to EDF file
        
    Returns:
        tuple: (is_valid, message)
    """
    try:
        raw = mne.io.read_raw_edf(file_path, preload=False, verbose=False)
        
        # Check for required channels
        available_channels = [ch for ch in EEG_CHANNELS if ch in raw.ch_names]
        
        if len(available_channels) < 15:
            return False, f"Insufficient channels: {len(available_channels)}/19"
        
        # Check duration (at least 4 seconds for one window)
        duration = raw.times[-1]
        if duration < WINDOW_SECONDS:
            return False, f"File too short: {duration:.1f}s (need at least {WINDOW_SECONDS}s)"
        
        return True, "File is valid"
    
    except Exception as e:
        return False, f"Invalid EDF file: {str(e)}"


# ============================================================================
# MAIN (For testing)
# ============================================================================

if __name__ == "__main__":
    """
    Test the EEG processor with a sample file.
    """
    
    print("\n" + "="*60)
    print("EEG PROCESSOR TEST")
    print("="*60)
    
    # Example usage:
    # processor = EEGProcessor()
    # windows, sfreq = processor.process_file("path/to/sample.edf")
    # print(f"Processed shape: {windows.shape}")
    # print(f"Sampling rate: {sfreq} Hz")
    
    print("EEG Processor module loaded successfully!")
    print(f"Channels: {len(EEG_CHANNELS)}")
    print(f"Filter: {LOW_FREQ}-{HIGH_FREQ} Hz")
    print(f"Window: {WINDOW_SECONDS}s with {OVERLAP*100}% overlap")
    print("="*60 + "\n")
