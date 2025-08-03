
"""
Last modified on Tue Mar 11 12:29:21 2025

@author: DevPortal
"""

import os
os.environ['TF_ENABLE_LAYOUT_OPTIMIZER'] = '0'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=0'
os.environ['TF_DISABLE_LAYOUT_OPTIMIZER'] = '1'
import time
import json
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
from PIL import Image, ImageSequence
from scipy.io import loadmat
from sklearn.metrics import (roc_auc_score, average_precision_score, 
                             precision_recall_fscore_support, accuracy_score, roc_curve)
from sklearn.model_selection import train_test_split
import keras_tuner as kt
import seaborn as sns
from tensorflow.keras.models import Model
import scipy.io
import traceback  


# Disable problematic optimizations
tf.config.optimizer.set_experimental_options({
    'layout_optimizer': False,
    'remapping': False,
})

#++++++++++++++++++++++++++++++++++++++++++++++
#Define Ground Truths
#++++++++++++++++++++++++++++++++++++++++++++++
# Specify the directory to save the .mat file
destination_dir = '/datasets/nndou/UCSD_Anomaly_Dataset/UCSDped1/Test'  # Replace with your desired directory

# Confirm the directory exists
os.makedirs(destination_dir, exist_ok=True)
# Define the ground truth data as a list of dictionaries, these are given 
GROUND_TRUTH = {
    # UCSDped1 Test Annotations
    'Test001': {'gt_frame': list(range(60, 154))},  
    'Test002': {'gt_frame': list(range(50, 177))},   
    'Test003': {'gt_frame': list(range(91, 202))},
    'Test004': {'gt_frame': list(range(31, 169))},
    'Test005': {'gt_frame': list(range(5, 91))},
    'Test006': {'gt_frame': list(range(1, 101))},
    'Test007': {'gt_frame': list(range(1, 176))},
    'Test008': {'gt_frame': list(range(1, 95))},
    'Test009': {'gt_frame': list(range(1, 49))},
    'Test010': {'gt_frame': list(range(1, 141))},
    'Test011': {'gt_frame': list(range(70, 166))},
    'Test012': {'gt_frame': list(range(130, 201))},
    'Test013': {'gt_frame': list(range(1, 157))},
    'Test014': {'gt_frame': list(range(1, 201))},
    'Test015': {'gt_frame': list(range(138, 201))},
    'Test016': {'gt_frame': list(range(123, 201))},
    'Test017': {'gt_frame': list(range(1, 48))},
    'Test018': {'gt_frame': list(range(54, 121))},
    'Test019': {'gt_frame': list(range(64, 139))},
    'Test020': {'gt_frame': list(range(45, 176))},
    'Test021': {'gt_frame': list(range(31, 201))},
    'Test022': {'gt_frame': list(range(16, 108))},
    'Test023': {'gt_frame': list(range(8, 166))},
    'Test024': {'gt_frame': list(range(50, 172))},
    'Test025': {'gt_frame': list(range(40, 136))},
    'Test026': {'gt_frame': list(range(77, 145))},
    'Test027': {'gt_frame': list(range(10, 123))},
    'Test028': {'gt_frame': list(range(105, 201))},
    'Test029': {'gt_frame': list(range(1, 16))},
    'Test030': {'gt_frame': list(range(175, 201))},
    'Test031': {'gt_frame': list(range(1, 181))},
    'Test032': {'gt_frame': list(range(1, 53))},
    'Test033': {'gt_frame': list(range(5, 166))},
    'Test034': {'gt_frame': list(range(1, 122))},
    'Test035': {'gt_frame': list(range(86, 201))},
    'Test036': {'gt_frame': list(range(15, 109))}
}

#++++++++++++++++++++++++++++++++++++++++++++++
# Set up MirroredStrategy to use GPU
#++++++++++++++++++++++++++++++++++++++++++++++
if __name__ == "__main__":
    strategy = None
    try:        
        strategy = tf.distribute.MirroredStrategy()
        print('Number of devices: {}'.format(strategy.num_replicas_in_sync))        
        # Wrap the entire code in strategy.scope()
        with strategy.scope():

            # =============================================
            # Store History
            # =============================================

            # =============================================
            # Enhanced History Tracker with Smoothing
            # =============================================
            class HistoryTracker(tf.keras.callbacks.Callback):
                def __init__(self, smoothing_factor=0.65):
                    super().__init__()
                    self.smoothing_factor = smoothing_factor
                    self.history = {
                        'loss': [],
                        'val_loss': [],
                        'smooth_loss': [],
                        'smooth_val_loss': []
                    }
            
                def on_epoch_end(self, epoch, logs=None):
                    logs = logs or {}
                    # Always store raw values
                    self.history['loss'].append(logs.get('loss', np.nan))
                    self.history['val_loss'].append(logs.get('val_loss', np.nan))
                    
                    # Calculate smoothed values
                    if epoch == 0:
                        self.history['smooth_loss'].append(logs.get('loss', np.nan))
                        self.history['smooth_val_loss'].append(logs.get('val_loss', np.nan))
                    else:
                        # Handle potential missing values
                        prev_smooth_train = self.history['smooth_loss'][-1] if self.history['smooth_loss'] else 0
                        prev_smooth_val = self.history['smooth_val_loss'][-1] if self.history['smooth_val_loss'] else 0
                        
                        new_smooth_train = self.smoothing_factor * prev_smooth_train + (1 - self.smoothing_factor) * logs.get('loss', np.nan)
                        new_smooth_val = self.smoothing_factor * prev_smooth_val + (1 - self.smoothing_factor) * logs.get('val_loss', np.nan)
                        
                        self.history['smooth_loss'].append(new_smooth_train)
                        self.history['smooth_val_loss'].append(new_smooth_val)
            
            
            # =============================================
            # Added these new classes in Part 1 after HistoryTracker
            # =============================================
            class CosineDecayWithRestarts(tf.keras.optimizers.schedules.LearningRateSchedule):
                def __init__(self, initial_lr, initial_epochs, decay_steps, t_mul=2.0, m_mul=0.5):
                    super().__init__()
                    self.initial_lr = initial_lr
                    self.initial_epochs = initial_epochs
                    self.decay_steps = decay_steps
                    self.t_mul = t_mul
                    self.m_mul = m_mul
                    self.epoch = 0
            
                def __call__(self, step):
                    cycle = tf.cast(
                        tf.math.floor(1 + (self.epoch - self.initial_epochs) / self.decay_steps),
                        tf.float32
                    )
                    x = 1 + (self.epoch - self.initial_epochs - self.decay_steps * (cycle - 1)) / self.decay_steps
                    return self.initial_lr * (self.m_mul ** cycle) * 0.5 * (1 + tf.cos(x * np.pi))
            
                def update_epoch(self, epoch):
                    self.epoch = epoch
            
                def get_config(self):
                    return {
                        'initial_lr': self.initial_lr,
                        'initial_epochs': self.initial_epochs,
                        'decay_steps': self.decay_steps,
                        't_mul': self.t_mul,
                        'm_mul': self.m_mul
                    }
            
                @classmethod
                def from_config(cls, config):
                    return cls(**config)      
            
            
            # =============================================
            # Configuration (Update paths to run)
            # =============================================
            CONFIG = {
                'data_root': '/datasets/nndou/UCSD_Anomaly_Dataset/UCSDped1',
                'train_folders': [f'Train/Train{i:03d}' for i in range(1, 34)],  
                'test_folders': [f'Test/Test{i:03d}' for i in range(1, 36)],     
                'output_dir': '/datasets/nndou/anomaly_output_v',
                'gt_file': 'ground_truth.mat',
                'patch_size': (32, 32),
                'stride': 16,
                'farneback_params': {
                    'pyr_scale': 0.5,
                    'levels': 3,
                    'winsize': 15,
                    'iterations': 3,
                    'poly_n': 5,
                    'poly_sigma': 1.2,
                    'flags': cv2.OPTFLOW_FARNEBACK_GAUSSIAN
                },
                'flow_shape': (256, 256, 2),
                'random_search': {
                    'max_trials': 5,
                    'executions_per_trial': 1,
                    'directory': 'random_search_v',
                    'project_name': 'anomaly_detection_v',
                    'patience': 8
                },
                'hparam_space': {
                    'latent_dim': (64, 128),  # Expanded based on SOTA VAE designs
                    'learning_rate': [1e-5, 3e-5, 1e-4, 3e-4, 1e-3],  # Wider range
                    'beta': [0.02, 0.05, 0.1, 0.15],  # Allow higher β values
                    'conv_layers': [3, 4],  # Deeper architectures
                    'filters_base': [64, 128],  # Larger base filters
                    'batch_size': [64, 128],  # Add batch size search
                    'use_batchnorm': [True, False],  # Batch normalization option
                    'activation': ['relu', 'leaky_relu']  # Activation function choice
                },
                
                # Add learning rate schedule config
                'lr_schedule': {
                    'initial_epochs': 10,
                    'decay_steps': 20,
                    't_mul': 2,
                    'm_mul': 0.5
                },
                
                'seed': 42,
                'vis_freq': 10,
                'max_train_frames': 200,  # For quick training tests
                'max_test_frames': 200    # Full test sequences
        
            } 

            #======================================================================
            #
            #=====================================================================
            class SafeDropout(tf.keras.layers.Layer):
                """Channel-aligned dropout for NHWC format stability"""
                def __init__(self, rate, **kwargs):
                    super().__init__(**kwargs)
                    self.rate = rate  # Dropout rate (0.0-1.0)
                    
                def build(self, input_shape):
                    # Validate input format (NHWC)
                    if len(input_shape) != 4:
                        raise ValueError("SafeDropout requires 4D input (NHWC format)")
                    self.built = True
                    
                def call(self, inputs, training=None):
                    if training and self.rate > 0:
                        # Create 4D mask matching [batch, H, W, 1]
                        mask_shape = tf.shape(inputs)  # [batch, H, W, C]
                        mask_shape = mask_shape[0], mask_shape[1], mask_shape[2], 1
                        mask = tf.nn.dropout(
                            tf.ones(mask_shape, dtype=inputs.dtype), 
                            rate=self.rate
                        )
                        return inputs * mask
                    return inputs
                
                def get_config(self):
                    return {'rate': self.rate}                       
            
            #=======================================================        
            from tensorflow.keras import backend as K
            K.set_image_data_format('channels_last')  
            # =============================================
            # Enhanced Data Handler with Correct Frame Loading
            # =============================================
            class UCSDDataHandler:
                def __init__(self, config):
                    self.config = config
                    self.prev_valid_flow = None  
                    self.patch_size = config['patch_size']
                    self.stride = config['stride']
                    self.ground_truth = GROUND_TRUTH  
        
                def reset_flow_history(self):
                    """Call before processing a new video"""
                    self.prev_valid_flow = None


                def _load_ground_truth(self):
                    gt_path = os.path.join(self.config['data_root'], 'Test', self.config['gt_file'])
                    print(f"Loading ground truth from: {gt_path}")
                    if not os.path.exists(gt_path):
                        raise FileNotFoundError(f"Ground truth file missing: {gt_path}")
                    mat_data = loadmat(gt_path)
                    self.gt_data = mat_data['TestVideoFile']
                    
                def get_test_gt(self, test_folder):
                    """Convert 1-based frame ranges to 0-based binary labels"""
                    test_name = os.path.basename(test_folder)  # e.g., 'Test001'
                    gt_entry = self.ground_truth.get(test_name, {'gt_frame': []})
                    
                    # Get actual video length
                    folder_path = os.path.join(self.config['data_root'], test_folder)
                    video_frames = self.load_video_frames(folder_path)
                    video_length = len(video_frames)
                    
                    # Initialize all-normal labels
                    gt_labels = np.zeros(video_length, dtype=np.int32)
                    
                    # Convert 1-based to 0-based indices
                    if gt_entry['gt_frame']:
                        start = max(0, gt_entry['gt_frame'][0] - 1)
                        end = min(video_length, gt_entry['gt_frame'][-1] - 1)
                        gt_labels[start:end+1] = 1  # Inclusive
                        
                        print(f"Anomaly frames: {start}-{end} ({end-start+1} frames)")
                    else:
                        print("No anomalies in this sequence")
                        
                    return gt_labels
        
                # =============================================
                # Modified UCSDDataHandler.load_video_frames
                # =============================================
                def load_video_frames(self, folder_path, is_training=False):
                    """Load all numbered TIFF frames from a directory"""
                    frames = []
                    try:
                        """Load frames with mode-specific limits"""
                        max_frames = (
                            self.config['max_train_frames'] 
                            if is_training 
                            else self.config['max_test_frames']
                        )
        
                        # Case-insensitive .tif extension check and numeric sorting
                        frame_files = sorted(
                            [f for f in os.listdir(folder_path) if f.lower().endswith('.tif')],
                            key=lambda x: int(''.join(filter(str.isdigit, x)))  # Extract all digits from filename
                        )
                        
                        print(f"Loading {len(frame_files)} frames from: {folder_path}")  # Debug log
                
                        if not frame_files:
                            raise FileNotFoundError(f"No TIFF files found in {folder_path}")
                
                        # Load frames with progress indication
                        for idx, frame_file in enumerate(frame_files[:max_frames]):
                            frame_path = os.path.join(folder_path, frame_file)
                            with Image.open(frame_path) as img:
                                frame = img.convert('L').resize((256, 256))  # Grayscale
                                frame = np.array(frame) / 255.0
                                frames.append(np.expand_dims(frame, axis=-1))  # Add channel dim
                                
                        return np.array(frames)
                    
                    except Exception as e:
                        print(f"Error loading frames from {folder_path}: {str(e)}")
                        traceback.print_exc()  # Show full error context
                        return np.array([])
            
                def compute_optical_flow(self, rgb_frames):
                    flow_sequence = []
                    try:
                        if len(rgb_frames) < 2:
                            print("⚠️ Using previously computed flow")
                            # Pad with a single zero flow frame to avoid empty array
                            #dummy_flow = np.zeros(self.config['flow_shape'], dtype=np.float32)
                            return np.array([self.prev_valid_flow])                
                        
                        prev_frame = (rgb_frames[0].squeeze() * 255).astype(np.uint8)
                        
                        for frame in rgb_frames[1:]:
                            curr_frame = (frame.squeeze() * 255).astype(np.uint8)
                            flow = cv2.calcOpticalFlowFarneback(
                                prev=prev_frame,
                                next=curr_frame,
                                flow=None,
                                **self.config['farneback_params']
                            )
                            flow_sequence.append(flow)
                            self.prev_valid_flow = flow  # Update tracker
                            prev_frame = curr_frame
                            
                        return np.array(flow_sequence)
                    
                    except Exception as e:
                        print(f"🚨 Optical flow computation failed: {str(e)}")
                        traceback.print_exc()
                        return np.empty((0,) + self.config['flow_shape'])
        
                def extract_patches(self, data, is_flow=False):
                    patches = []
                    if len(data) == 0:
                        print("Empty input in extract_patches!")
                        return np.array(patches)
                        
                    for idx, item in enumerate(data):
                        # Validate frame dimensions
                        if item.shape[0] < self.patch_size[0] or item.shape[1] < self.patch_size[1]:
                            print(f"Invalid frame {idx}: {item.shape} < {self.patch_size}")
                            continue
                            
                        # Ensure proper channel dimension
                        if len(item.shape) == 2:  # Grayscale
                            item = np.expand_dims(item, axis=-1)
                            
                        h, w = item.shape[:2]
                        for y in range(0, h - self.patch_size[0] + 1, self.stride):
                            for x in range(0, w - self.patch_size[1] + 1, self.stride):
                                patch = item[y:y+self.patch_size[0], x:x+self.patch_size[1]]
                                patches.append(patch)
                    
                    print(f"Extracted {len(patches)} patches from {len(data)} frames")
                    
                    return np.array(patches)
                
                def load_training_data(self, model_type):
                    all_patches = []
                    for folder in self.config['train_folders']:
                        folder_path = os.path.join(self.config['data_root'], folder)
                        frames = self.load_video_frames(folder_path, is_training=True)
        
                        if model_type == 'spatial':
                            patches = self.extract_patches(frames)
                        else:  # temporal
                            flow_frames = self.compute_optical_flow(frames)
                            patches = self.extract_patches(flow_frames, is_flow=True)
        
                        all_patches.extend(patches)
                        print(f"Processed {folder} - {len(patches)} patches")
        
                    return np.array(all_patches)
        
                
            # =============================================
            # Visualization Callback
            # =============================================
            class VisualizationCallback(tf.keras.callbacks.Callback):
                def __init__(self, val_data, output_dir, model_type):
                    super().__init__()
                    self.val_data = val_data
                    self.output_dir = output_dir
                    self.model_type = model_type
                    os.makedirs(self.output_dir, exist_ok=True)
            
                def on_epoch_end(self, epoch, logs=None):
                    if (epoch + 1) % CONFIG['vis_freq'] == 0:
                        try:
                            samples = self.val_data[np.random.choice(len(self.val_data), size=5, replace=False)]
                            reconstructions = self.model.predict(samples, verbose=0)
                            
                            fig, axs = plt.subplots(2, 5, figsize=(15, 6))
                            
                            for i in range(5):
                                # Handle different input types
                                if self.model_type == 'spatial':
                                    # Grayscale images (1 channel)
                                    axs[0, i].imshow(samples[i].squeeze(), cmap='gray')
                                    axs[1, i].imshow(reconstructions[i].squeeze(), cmap='gray')
                                else:
                                    # Optical flow (2 channels) - show magnitude
                                    sample_mag = np.linalg.norm(samples[i], axis=-1)
                                    recon_mag = np.linalg.norm(reconstructions[i], axis=-1)
                                    
                                    axs[0, i].imshow(sample_mag, cmap='viridis', vmin=0, vmax=1)
                                    axs[1, i].imshow(recon_mag, cmap='viridis', vmin=0, vmax=1)
                                
                                axs[0, i].axis('off')
                                axs[1, i].axis('off')
            
                            plt.suptitle(f"{self.model_type.capitalize()} Reconstructions at Epoch {epoch+1}")
                            plt.savefig(
                                os.path.join(self.output_dir, f"{self.model_type}_recon_epoch{epoch+1}.png"),
                                bbox_inches='tight'
                            )
                            plt.close(fig)
                            
                        except Exception as e:
                            print(f"⚠️ Visualization failed at epoch {epoch+1}: {str(e)}")
                            plt.close('all')
                
            # =============================================
            # Improved Late Fusion Model
            # =============================================
            class ImprovedLateFusion:
                def __init__(self, spatial_model, temporal_model, data_handler):
                    self.spatial_model = spatial_model
                    self.temporal_model = temporal_model
                    self.weights = np.array([0.5, 0.5])  # Initial weights
                    self.data_handler = data_handler  # Critical addition
                    
                def compute_score(self, rgb_frame, flow_frame):
                    try:
                        # Spatial score (always available)
                        spatial_patches = self.data_handler.extract_patches([rgb_frame])
                        if len(spatial_patches) == 0:
                            return 0.0
                        spatial_recon = self.spatial_model.predict(spatial_patches, verbose=0)
                        spatial_error = np.mean(np.square(spatial_patches - spatial_recon))
                
                        # Temporal score (only if valid flow exists)
                        temporal_error = 0.0
                        if flow_frame is not None and flow_frame.size > 0:
                            temporal_patches = self.data_handler.extract_patches([flow_frame], is_flow=True)
                            if len(temporal_patches) > 0:
                                temporal_recon = self.temporal_model.predict(temporal_patches, verbose=0)
                                temporal_error = np.mean(np.square(temporal_patches - temporal_recon))
                
                        # Weighted fusion
                        return (spatial_error + temporal_error) / 2
                
                    except Exception as e:
                        print(f"🚨 Fusion error: {str(e)}")
                        return spatial_error  # Fallback to spatial score
                    
            # =============================================
            # Complete Training Manager
            # =============================================
            class TrainingManager:
                def __init__(self, config):
                    self.config = config
                    self.data_handler = UCSDDataHandler(config)
                    self.models = {
                        'spatial': None,
                        'temporal': None
                    }
                    self.histories = {'spatial': {'loss': [], 'val_loss': []},  # Initialize with empty lists
                                     'temporal': {'loss': [], 'val_loss': []}}
                    
                def train_all_models(self):
                    # Train Spatial VAE
                    print("\n=== Training Spatial VAE ===")
                    spatial_data = self.data_handler.load_training_data('spatial')
                    self.models['spatial'] = self._train_model('spatial', spatial_data)
                    
                    # Train Temporal VAE
                    print("\n=== Training Temporal VAE ===")
                    temporal_data = self.data_handler.load_training_data('temporal')
                    self.models['temporal'] = self._train_model('temporal', temporal_data)

                    #self.histories = {'spatial': None, 'temporal': None}  # NEW: Store histories
                                
                def _train_model(self, model_type, data):
                    try:
                        # 1. Initialize Tuner
                        tuner = kt.RandomSearch(
                            lambda hp: self._build_vae(hp, model_type),
                            objective='val_loss',
                            max_trials=self.config['random_search']['max_trials'],
                            executions_per_trial=1,
                            directory=os.path.join(self.config['random_search']['directory'], model_type),
                            project_name=self.config['random_search']['project_name'],
                            overwrite=True
                        )
            
                        # 2. Prepare Data
                        train_data, val_data = train_test_split(
                            data, 
                            test_size=0.2, 
                            random_state=self.config['seed']
                        )
            
                        # 3. Hyperparameter Search
                        search_history = HistoryTracker()
                        tuner.search(
                            train_data, train_data,
                            epochs=50,
                            validation_data=(val_data, val_data),
                            callbacks=[
                                search_history,
                                tf.keras.callbacks.EarlyStopping(
                                    patience=self.config['random_search']['patience'],
                                    restore_best_weights=True,
                                    monitor='val_loss'
                                ),
                                VisualizationCallback(
                                    val_data[:5],
                                    os.path.join(self.config['output_dir'], 'training_vis', model_type),
                                    model_type)
                            ]
                        )
            
                        # 4. Final Training
                        best_hparams = tuner.get_best_hyperparameters()[0]
                        best_model = self._build_vae(best_hparams, model_type)

                        # best_model = tuner.get_best_models(num_models=1)[0]
                        final_history = HistoryTracker()
                        
                        best_model.fit(
                            train_data, train_data,
                            epochs=50,
                            batch_size=128,
                            validation_data=(val_data, val_data),
                            callbacks=[
                                final_history,
                                tf.keras.callbacks.EarlyStopping(
                                    patience=self.config['random_search']['patience']*3,
                                    restore_best_weights=True,
                                    monitor='val_loss'
                                ),
                                VisualizationCallback(
                                    val_data[:5],
                                    os.path.join(self.config['output_dir'], 'training_vis', model_type),
                                    model_type)
                            ]
                        )
            
                        # 5. Store Combined History
                        self.histories[model_type] = {
                            'loss': search_history.history['loss'] + final_history.history['loss'],
                            'val_loss': search_history.history['val_loss'] + final_history.history['val_loss'],
                            'smooth_loss': search_history.history['smooth_loss'] + final_history.history['smooth_loss'],
                            'smooth_val_loss': search_history.history['smooth_val_loss'] + final_history.history['smooth_val_loss']
                        }
                                    
                        # 6. Save Model
                        model_path = os.path.join(self.config['output_dir'], 'models', f'{model_type}_vae')
                        best_model.save(model_path, save_format="tf")
                        
                        return best_model
            
                    except Exception as e:
                        print(f"Training failed for {model_type}: {str(e)}")
                        self.histories[model_type] = {'loss': [], 'val_loss': []}
                        return None


                def plot_loss_curves(self, output_dir):
                    os.makedirs(output_dir, exist_ok=True)
                    
                    for model_type in ['spatial', 'temporal']:
                        # Safely get history from initialized structure
                        history = self.histories[model_type]  # Direct access since we pre-initialize
                        
                        # Extract data with type safety
                        loss = np.array(history['loss'], dtype=np.float32)
                        val_loss = np.array(history.get('val_loss', []), dtype=np.float32)
                        
                        # Enhanced validation
                        if len(loss) == 0:
                            print(f"⚠️ No training data for {model_type.upper()} model")
                            continue
                            
                        # Handle validation data presence
                        has_validation = len(val_loss) > 0
                        min_length = min(len(loss), len(val_loss)) if has_validation else len(loss)
                        
                        plt.figure(figsize=(14, 7))
                        epochs = range(min_length)
                        
                        # Main plot with improved styling
                        plt.plot(epochs, loss[:min_length], 
                                'b-o',  # Blue circles
                                label='Training Loss', 
                                markersize=4,
                                linewidth=1.5)
                        
                        if has_validation:
                            plt.plot(epochs, val_loss[:min_length], 
                                    'r--s',  # Red squares
                                    label='Validation Loss',
                                    markersize=4,
                                    linewidth=1.5)
                        
                        # Enhanced formatting
                        plt.title(f"{model_type.upper()} Model: Loss Progression", fontsize=14)
                        plt.xlabel("Epochs", fontsize=12)
                        plt.ylabel("Reconstruction Loss", fontsize=12)
                        plt.legend(frameon=True, facecolor='white')
                        plt.grid(True, linestyle='--', alpha=0.7)
                        
                        # Set consistent axis limits
                        max_loss = max(loss.max(), val_loss.max()) if has_validation else loss.max()
                        plt.ylim(0, max_loss * 1.1)
                        
                        # Save with metadata
                        try:
                            plot_path = os.path.join(output_dir, f"{model_type}_loss_curve.png")
                            plt.savefig(
                                plot_path,
                                bbox_inches='tight',
                                dpi=300,  # Higher resolution
                                metadata={
                                    'Title': f"{model_type} Loss Curve",
                                    'Author': 'Your Name',
                                    'Description': 'VAE training progress'
                                }
                            )
                            plt.close()
                            print(f"✅ Success: {model_type} curve saved to {plot_path}")
                        except Exception as e:
                            print(f"🚨 Critical error saving {model_type} plot: {str(e)}")
                            plt.close('all')

                def plot_training_curves(self, output_dir):
                    import matplotlib.colors as mcolors
                    
                    os.makedirs(output_dir, exist_ok=True)
                    
                    # Define light colors
                    light_blue = '#ADD8E6'
                    light_red = '#FFA07A'
                    
                    for model_type in ['spatial', 'temporal']:
                        history = self.histories.get(model_type, {})
                        if not history:
                            print(f"No history found for {model_type}")
                            continue
                        
                        ## -------- RAW CURVES -------- ##
                        raw_epochs = range(1, len(history.get('loss', [])) + 1)
                        plt.figure(figsize=(14, 7))
                        
                        if len(raw_epochs) > 0:
                            plt.plot(raw_epochs, history['loss'], color=light_blue, label='Raw Train', alpha=0.8)
                            if 'val_loss' in history and len(history['val_loss']) == len(raw_epochs):
                                plt.plot(raw_epochs, history['val_loss'], color=light_red, label='Raw Val', alpha=0.8)
                        
                        plt.title(f"{model_type.capitalize()} Raw Training Curves")
                        plt.xlabel('Epochs')
                        plt.ylabel('Loss')
                        plt.legend()
                        plt.grid(True, linestyle='--', alpha=0.7)
                        
                        try:
                            plot_path = os.path.join(output_dir, f"{model_type}_raw_training_curves.png")
                            plt.savefig(plot_path, bbox_inches='tight', dpi=300)
                            plt.close()
                            print(f"Saved RAW curves to {plot_path}")
                        except Exception as e:
                            print(f"Error saving raw plot: {str(e)}")
                            plt.close('all')
                        
                        ## -------- SMOOTHED CURVES -------- ##
                        smooth_epochs = range(1, len(history.get('smooth_loss', [])) + 1)
                        plt.figure(figsize=(14, 7))
                        
                        if len(smooth_epochs) > 0:
                            plt.plot(smooth_epochs, history['smooth_loss'], color=light_blue, label='Smoothed Train', linewidth=2)
                            if 'smooth_val_loss' in history and len(history['smooth_val_loss']) == len(smooth_epochs):
                                plt.plot(smooth_epochs, history['smooth_val_loss'], color=light_red, label='Smoothed Val', linewidth=2)
                        
                        plt.title(f"{model_type.capitalize()} Smoothed Training Curves")
                        plt.xlabel('Epochs')
                        plt.ylabel('Loss')
                        plt.legend()
                        plt.grid(True, linestyle='--', alpha=0.7)
                        
                        try:
                            plot_path = os.path.join(output_dir, f"{model_type}_smoothed_training_curves.png")
                            plt.savefig(plot_path, bbox_inches='tight', dpi=300)
                            plt.close()
                            print(f"Saved SMOOTHED curves to {plot_path}")
                        except Exception as e:
                            print(f"Error saving smoothed plot: {str(e)}")
                            plt.close('all')
                
                def _build_vae(self, hp, model_type):
                    
                    # Add input dimension validation
                    input_size = 32  # Based on your patch size
                    # max_conv_layers = int(np.log2(input_size))  # 5 for 32x32 input
                    
                    # Hyperparameter Integration
                    use_batchnorm = hp.Boolean('use_batchnorm')
                    activation_type = hp.Choice('activation', ['relu', 'leaky_relu'])
                    conv_layers = hp.Int('conv_layers', 3, 4)  # 3-5 layers for 32x32
                    
                    # Add this check after defining conv_layers
                    if input_size // (2 ** conv_layers) < 1:
                        raise ValueError(f"Too many conv layers ({conv_layers}) for input size {input_size}")
                    
                    filters_base = hp.Int('filters_base', 64, 128, step=32)  # Larger filters
                    latent_dim = hp.Int('latent_dim', 64, 128)
                    beta = hp.Float('beta', 0.05, 0.15, step=0.02)  # Wider beta range
                    
                    # Input Configuration
                    input_shape = (32, 32, 1) if model_type == 'spatial' else (32, 32, 2)
                    final_activation = 'sigmoid' if model_type == 'spatial' else 'tanh'
                    
                    # Dynamic Activation
                    activation = tf.keras.layers.LeakyReLU() if activation_type == 'leaky_relu' else tf.keras.layers.ReLU()
                    
                    # Learning Rate Schedule Integration
                    lr_schedule = CosineDecayWithRestarts(
                        initial_lr=hp.Float('learning_rate', 1e-5, 1e-3, sampling='log'),
                        initial_epochs=CONFIG['lr_schedule']['initial_epochs'],
                        decay_steps=CONFIG['lr_schedule']['decay_steps'],
                        t_mul=CONFIG['lr_schedule']['t_mul'],
                        m_mul=CONFIG['lr_schedule']['m_mul']
                    )
                    
                    # Optimizer with Weight Decay
                    optimizer = tf.keras.optimizers.AdamW(
                        learning_rate=lr_schedule,
                        weight_decay=hp.Float('weight_decay', 1e-4, 3e-4, sampling='log'),
                        global_clipnorm=1.0
                    )
                    
                    # Encoder
                    encoder_inputs = tf.keras.Input(shape=input_shape, dtype=tf.float32)
                    x = encoder_inputs
                    
                    for i in range(conv_layers):
                        x = tf.keras.layers.Conv2D(
                            filters_base * (2 ** i),
                            kernel_size=3,
                            strides=2,
                            padding='same',
                            kernel_initializer='he_normal'
                        )(x)
                        if use_batchnorm:
                            x = tf.keras.layers.BatchNormalization()(x)
                        x = activation(x)
                        x = SafeDropout(hp.Float('dropout_rate', 0.0, 0.15))(x)  # Modified line
                    
                    x = tf.keras.layers.Flatten()(x)
                    z_mean = tf.keras.layers.Dense(latent_dim)(x)
                    z_log_var = tf.keras.layers.Dense(latent_dim)(x)
                    z = tf.keras.layers.Lambda(
                        lambda args: args[0] + tf.exp(0.5 * args[1]) * tf.random.normal(tf.shape(args[0]))
                    )([z_mean, z_log_var])
                    
                    encoder = tf.keras.Model(encoder_inputs, [z_mean, z_log_var, z], name=f"{model_type}_encoder")
                    
                    
                    # Decoder
                    decoder_input = tf.keras.Input(shape=(latent_dim,))
                    
                    decoder_init_size = input_size // (2 ** conv_layers)
                    x = tf.keras.layers.Dense(
                        decoder_init_size * decoder_init_size * filters_base * (2 ** (conv_layers - 1))
                    )(decoder_input)
                    x = tf.keras.layers.Reshape((
                        decoder_init_size,
                        decoder_init_size,
                        filters_base * (2 ** (conv_layers - 1))
                    ))(x)
                    
                    x = tf.keras.layers.Dense(
                        (input_shape[0] // (2 ** conv_layers)) * 
                        (input_shape[1] // (2 ** conv_layers)) * 
                        filters_base * (2 ** (conv_layers - 1))
                    )(decoder_input)
                    x = tf.keras.layers.Reshape((
                        input_shape[0] // (2 ** conv_layers),
                        input_shape[1] // (2 ** conv_layers),
                        filters_base * (2 ** (conv_layers - 1))
                    ))(x)
                    
                    for i in reversed(range(conv_layers)):
                        x = tf.keras.layers.Conv2DTranspose(
                            filters_base * (2 ** i),
                            kernel_size=3,
                            strides=2,
                            padding='same',
                            kernel_initializer='he_normal'
                        )(x)
                        if use_batchnorm:
                            x = tf.keras.layers.BatchNormalization()(x)
                        x = activation(x)
                        x = SafeDropout(hp.Float('dropout_rate', 0.0, 0.15))(x)  # Modified line
                    
                    decoder_outputs = tf.keras.layers.Conv2DTranspose(
                        input_shape[-1],
                        3,
                        activation=final_activation,
                        padding='same'
                    )(x)
                    
                    decoder = tf.keras.Model(decoder_input, decoder_outputs, name=f"{model_type}_decoder")
                    
                    # VAE Model
                    vae_outputs = decoder(encoder(encoder_inputs)[2])
                    vae = tf.keras.Model(encoder_inputs, vae_outputs, name=f"{model_type}_vae")
                    
                    # Critical addition: Attach submodels
                    vae.encoder = encoder
                    vae.decoder = decoder
                    
                    # Loss Calculation
                    reconstruction_loss = tf.reduce_mean(
                        tf.keras.losses.mse(encoder_inputs, vae_outputs)
                    )
                    kl_loss = -0.5 * beta * tf.reduce_mean(
                        1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
                    )
                    vae.add_loss(reconstruction_loss + kl_loss)
                    vae.compile(optimizer=optimizer)  # Use the created optimizer
                    
                    return vae
                
                    
            # =============================================
            # Complete Evaluation System
            # =============================================
            class UnifiedEvaluator:
                def __init__(self, models, data_handler, config):  # Add config parameter
                    self.models = models
                    self.data_handler = data_handler
                    self.config = config  # Store config
                    self.metrics = {m: {} for m in ['spatial', 'temporal', 'fusion']}
                    # Add model verification
                    print("\nModel Architecture Verification:")
                    print("Spatial VAE:")
                    models['spatial'].summary()
                    print("\nTemporal VAE:")
                    models['temporal'].summary()
                        
                def evaluate_all_models(self):
                    for test_folder in CONFIG['test_folders']:
                        print(f"\n=== Evaluating {test_folder} ===")
                        self.evaluate_test_folder(test_folder)
                

                def reset_flow_history(self):
                    """Call before processing a new video"""
                    self.prev_valid_flow = None


                def evaluate_test_folder(self, test_folder):
                    try:
                        folder_path = os.path.join(self.config['data_root'], test_folder)
                        self.data_handler.reset_flow_history()  # Reset for new video
                        rgb_frames = self.data_handler.load_video_frames(folder_path)
                        
                        # ============================================================
                        # CRITICAL: Compute flow frames once and store them
                        # ============================================================
                        self.flow_frames = self.data_handler.compute_optical_flow(rgb_frames)
                        print(f"Computed {len(self.flow_frames)} flow frames")
                        # Critical check for empty flow data
                        if self.flow_frames.size == 0:
                            print("⚠️ Skipping temporal/fusion models: No optical flow data")
                            return  # Skip temporal and fusion processing
                        
                        # Validate flow frame count matches RGB frames
                        assert len(self.flow_frames) == len(rgb_frames) - 1, \
                            f"Flow frames mismatch: {len(self.flow_frames)} vs {len(rgb_frames)-1}"
                        
                        if len(rgb_frames) == 0:
                            print(f"Skipping {test_folder} - No frames loaded")
                            return
                            

                        gt_labels = self.data_handler.get_test_gt(test_folder)
                        gt_labels = gt_labels[:len(rgb_frames)]  # <--- CRITICAL FIX HERE                        
                        if len(gt_labels) != len(rgb_frames):
                            print(f"⚠️ GT labels mismatch: {len(gt_labels)} vs {len(rgb_frames)}")
                            gt_labels = gt_labels[:len(rgb_frames)]  # Force alignment
                        
                        for model_type in ['spatial', 'temporal', 'fusion']:
                            try:
                                
                                # Skip temporal/fusion if no flow data
                                if model_type in ['temporal', 'fusion'] and self.flow_frames.size == 0:
                                    print(f"⚠️ Skipping {model_type} model: No optical flow data")
                                    continue
                                
                                scores, timings = self._process_model(
                                    model_type, 
                                    rgb_frames, 
                                    self.flow_frames  # Pass precomputed flow frames
                                )
                                
                                if len(scores) == 0:
                                    print(f"No scores generated for {model_type} in {test_folder}")
                                    continue
                                
                                gt_labels = gt_labels[:len(rgb_frames)]  # Force alignment
                                metrics = self._calculate_metrics(scores, gt_labels)
                                self.metrics[model_type][test_folder] = metrics
                                self._save_results(test_folder, model_type, metrics)
                                
                                # Visualization pipeline
                                output_dir = os.path.join(CONFIG['output_dir'], test_folder, model_type)
                                os.makedirs(output_dir, exist_ok=True)
                                
                                # Core visualizations
                                self._visualize_results(test_folder, model_type, rgb_frames, scores, gt_labels)
                                self._plot_score_distributions(scores, gt_labels[:len(scores)], output_dir, model_type)
                                
                                # Model-specific visualizations
                                if model_type != 'fusion':
                                    sample_frame = rgb_frames[len(rgb_frames)//2]
                                    self._visualize_feature_maps(model_type, rgb_frames, sample_frame, output_dir)  # ✅ Add rgb_frames                               
                                # Key frame analysis
                                for idx in [0, len(rgb_frames)//2, -1]:
                                    self._visualize_example_frame(rgb_frames[idx], model_type, output_dir, idx)
                                    
                            except Exception as model_error:
                                print(f"Failed processing {model_type}: {str(model_error)}")
                                continue
                
                    except Exception as main_error:
                        print(f"Critical error in {test_folder}: {str(main_error)}")
                        traceback.print_exc() 
                        
                        
                def _process_model(self, model_type, rgb_frames, flow_frames):
                    scores = []
                    timings = {'total': []}
                    fusion_model = None  # Initialize only when needed
                    
                    print(f"\n{'='*40}")
                    print(f"Processing {model_type.upper()} model")
                    print(f"{'='*40}")
                    # Initialize fusion model only for fusion processing
                    if model_type == 'fusion':
                        fusion_model = ImprovedLateFusion(
                            spatial_model=self.models['spatial'],
                            temporal_model=self.models['temporal'],
                            data_handler=self.data_handler  # MUST pass data handler
                        ) 
                        
                    for flow_idx in range(len(self.flow_frames)):
                        frame_idx = flow_idx + 1  # flow_idx 0 corresponds to frame 1
                        
                        frame_start_time = time.time()
                        score = 0.0  # Default score
                
                        try:
                            if model_type == 'spatial':
                                frame = rgb_frames[frame_idx]
                                patches = self.data_handler.extract_patches([frame])
                                
                                if len(patches) == 0:
                                    print(f"No patches in frame {frame_idx}")
                                    raise ValueError("Empty patches")
                                
                                pred = self.models['spatial'].predict(patches, verbose=0)
                                patch_errors = np.mean(np.square(patches - pred), axis=(1,2,3))
                                score = np.mean(patch_errors)
                                
                                print(f"Frame {frame_idx} spatial score: {score:.4f}")
                
                            # In UnifiedEvaluator._process_model() for temporal model
                            elif model_type == 'temporal':
                                scores = [0.0]  # Initialize with frame 0 (no flow)
                                # Limit processing to available flow frames                                
                                try:
                                    if frame_idx == 0:
                                        score = 0.0  # First frame has no flow
                                    else:
                                        # FIXED: Proper NumPy array validation
                                        if flow_frames is None or not isinstance(flow_frames, np.ndarray):
                                            print("🚨 Invalid flow frames format")
                                            return np.array(scores), timings
                                            
                                        if flow_frames.size == 0:  # Use .size instead of len() for NumPy arrays
                                            print("⚠️ Empty flow frames array")
                                            return np.array(scores), timings
                            
                                        # FIXED: Array index validation using numpy methods
                                        valid_indices = np.arange(len(flow_frames))
                                        flow_idx = min(frame_idx-1, valid_indices[-1])
                                        flow_idx = max(flow_idx, valid_indices[0])
                                        
                                        flow_frame = flow_frames[flow_idx]
                                        
                                        # FIXED: Explicit array content check
                                        if not np.any(flow_frame):  # Check if array has any non-zero values
                                            print(f"⚠️ Empty flow frame at index {flow_idx}")
                                            score = 0.0
                                        else:
                                            patches = self.data_handler.extract_patches([flow_frame], is_flow=True)
                                            if len(patches) == 0:
                                                print(f"⚠️ No patches in frame {frame_idx}")
                                                score = 0.0
                                            else:
                                                pred = self.models['temporal'].predict(patches, verbose=0)
                                                score = np.mean(np.square(patches - pred))                                                
                                                scores.append(score)
                                                scores = scores[:len(rgb_frames)]

                                except IndexError as ie:
                                    print(f"🚨 Temporal index error: {str(ie)}")
                                    scores.append(0.0)
                                except Exception as e:
                                    print(f"🚨 Temporal error: {str(e)}")
                                    scores.append(0.0)
                                
                                print(f"Frame {frame_idx} temporal score: {score:.4f}")

                                while len(scores) < len(rgb_frames):
                                    scores.append(0.0)  # Pad with zeros for remaining frames
                                    scores = scores[:len(rgb_frames)]
                                
                            elif model_type == 'fusion':
                                if frame_idx == 0:
                                    score = 0.0  # First frame has no temporal data
                                else:
                                    try:
                                        # ===================================================
                                        # Enhanced Bounds Checking
                                        # ===================================================
                                        max_rgb_idx = len(rgb_frames) - 1
                                        max_flow_idx = len(flow_frames) - 1  # Will be -1 if flow_frames is empty
                                        
                                        # Skip if no valid flow data exists
                                        if max_flow_idx < 0:
                                            print("⚠️ Skipping fusion: No optical flow data")
                                            score = 0.0
                                        else:
                                            # Clamp indices to valid ranges
                                            rgb_idx = min(frame_idx, max_rgb_idx)
                                            flow_idx = min(frame_idx - 1, max_flow_idx)
                                            flow_idx = max(flow_idx, 0)  # Ensure flow_idx >= 0
                                            
                                            # Validate flow frame is not zero-padded garbage
                                            flow_frame = flow_frames[flow_idx]
                                            if np.all(flow_frame == 0):
                                                print(f"⚠️ Invalid flow frame at {flow_idx}")
                                                score = 0.0
                                            else:
                                                score = fusion_model.compute_score(
                                                    rgb_frames[rgb_idx],
                                                    flow_frame
                                                )
                                    except Exception as e:
                                        print(f"Fusion error: {str(e)}")
                                        score = 0.0
                                
                                print(f"Frame {frame_idx} fusion score: {score:.4f}")
                
                        except Exception as e:
                            print(f"Error processing frame {frame_idx}: {str(e)}")
                            score = 0.0
                
                        scores.append(score)
                        scores = scores[:len(rgb_frames)]
                        timings['total'].append(time.time() - frame_start_time)
                
                    return np.array(scores), timings        
                
                def _calculate_metrics(self, scores, labels):
                    
                    # Clean scores
                    scores = np.nan_to_num(scores, nan=np.median(scores), posinf=np.nanmax(scores), neginf=np.nanmin(scores))

                    # Handle edge cases
                    if len(np.unique(scores)) < 2:
                        return {
                            'AUC': 0.5,
                            'EER': 1.0,
                            'status': 'constant_scores'
                        }
                    
                    
                    # Trim to match lengths
                    min_len = min(len(scores), len(labels))
                    scores = scores[:min_len]
                    labels = labels[:min_len]
                    
                    # Handle edge cases
                    if min_len == 0:
                        return {"error": "No valid data"}
                    
                    metrics = {
                        'AUC': np.nan,
                        'EER': np.nan,
                        'F1': np.nan,
                        'Precision': np.nan,
                        'Recall': np.nan,
                        'AP': np.nan,
                        'Accuracy': np.nan
                    }
                    
                    unique_classes = np.unique(labels)
                    
                    # Handle single-class scenarios
                    if len(unique_classes) == 1:
                        metrics['Accuracy'] = accuracy_score(labels, np.zeros_like(labels))
                        return metrics
                        
                    try:
                        # ROC metrics
                        fpr, tpr, thresholds = roc_curve(labels, scores)
                        metrics['AUC'] = roc_auc_score(labels, scores)
                        eer_idx = np.nanargmin(np.abs(fpr - (1 - tpr)))
                        metrics['EER'] = fpr[eer_idx]
                        
                        # Replace inf thresholds
                        thresholds = np.nan_to_num(thresholds, nan=np.nanmax(scores), posinf=np.nanmax(scores), neginf=np.nanmin(scores))
                        
                        # Precision-Recall
                        metrics['AP'] = average_precision_score(labels, scores)
                        
                        # Optimal threshold (Youden's J)
                        j_scores = tpr - fpr
                        opt_idx = np.argmax(j_scores)
                        opt_threshold = thresholds[opt_idx]
                        
                        # Replace any inf/nan thresholds explicitly
                        if not np.isfinite(opt_threshold):
                            opt_threshold = np.nanmax(scores)  # fallback
                        
                        # Classification metrics
                        preds = (scores >= opt_threshold).astype(int)
                        prec, recall, f1, _ = precision_recall_fscore_support(
                            labels, preds, average='binary', zero_division=0
                        )
                        metrics.update({
                            'F1': f1,
                            'Precision': prec,
                            'Recall': recall,
                            'Threshold': float(opt_threshold),
                            'Accuracy': accuracy_score(labels, preds)
                        })
                        
                    except Exception as e:
                        print(f"Metric calculation error: {str(e)}")
                    
                    return metrics        
                
                def _save_results(self, test_folder, model_type, metrics):
                    output_dir = os.path.join(CONFIG['output_dir'], test_folder, model_type)
                    os.makedirs(output_dir, exist_ok=True)
                    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
                        json.dump(metrics, f, indent=2)
                
                def _visualize_results(self, test_folder, model_type, rgb_frames, scores, gt_labels):
                    # Add safety checks
                    if len(scores) == 0 or len(rgb_frames) == 0:
                        print(f"Skipping visualization for {test_folder} - No valid data")
                        return
                    
                    output_dir = os.path.join(self.config['output_dir'], test_folder, model_type, 'visualizations')
                    os.makedirs(output_dir, exist_ok=True)
                
                    # Score vs GT plot
                    plt.figure(figsize=(12, 6))
                    plt.plot(scores, label='Anomaly Score')
                    
                    # Handle possible GT length mismatch
                    if len(gt_labels) != len(scores):
                        gt_labels = gt_labels[:len(scores)]
                    
                    plt.plot(gt_labels, label='Ground Truth', alpha=0.5)
                    plt.title(f"{model_type.capitalize()} Model Results - {test_folder}")
                    plt.xlabel('Frame Number')
                    plt.ylabel('Score')
                    plt.legend()
                    plt.savefig(os.path.join(output_dir, 'score_plot.png'))
                    plt.close()
                
                    # Heatmap example - only if frames exist
                    if len(rgb_frames) > 0:
                        mid_idx = len(rgb_frames) // 2
                        plt.imshow(rgb_frames[mid_idx])
                        score_idx = min(mid_idx, len(scores)-1)  # Handle score length
                        plt.title(f"Sample Frame with Score: {scores[score_idx]:.3f}")
                        plt.savefig(os.path.join(output_dir, 'heatmap_example.png'))
                        plt.close()
        
                
                def _plot_score_distributions(self, scores, labels, output_dir, model_type):
                    plt.figure(figsize=(10, 6))
                    
                    # Check for valid distributions
                    normal_scores = scores[labels == 0]
                    abnormal_scores = scores[labels == 1]
                    
                    if len(normal_scores) > 1: 
                        sns.kdeplot(normal_scores, label='Normal Frames', fill=True, warn_singular=False)
                    if len(abnormal_scores) > 1:
                        sns.kdeplot(abnormal_scores, label='Abnormal Frames', fill=True, warn_singular=False)
                    
                    # Only add legend if we have plottable distributions
                    if len(normal_scores) > 1 or len(abnormal_scores) > 1:
                        plt.legend()
                    
                    plt.title(f'{model_type} Score Distribution')
                    plt.savefig(os.path.join(output_dir, 'score_distribution.png'))
                    plt.close()
                    
                def _visualize_feature_maps(self, model_type, rgb_frames, sample_frame, output_dir):
                    """Visualize encoder feature maps for spatial/temporal patterns"""
                    if model_type == 'spatial':
                        model = self.models['spatial'].encoder
                        sample = self.data_handler.extract_patches([sample_frame])[0]
                        
                        # --- Move these lines INSIDE the spatial block ---
                        layer_outputs = [layer.output for layer in model.layers if 'conv' in layer.name]
                        activation_model = Model(inputs=model.input, outputs=layer_outputs)
                        activations = activation_model.predict(np.expand_dims(sample, 0))
                        
                        # Plotting code
                        first_layer_activation = activations[0][0]
                        plt.figure(figsize=(15, 5))
                        plt.suptitle(f'Spatial Feature Maps - Layer 1')
                        for i in range(min(8, first_layer_activation.shape[-1])):
                            plt.subplot(2, 4, i+1)
                            plt.imshow(first_layer_activation[..., i])
                            plt.axis('off')
                        plt.savefig(os.path.join(output_dir, 'spatial_feature_maps.png'))
                        plt.close()
                
                    elif model_type == 'temporal':
                        # --- Temporal-specific code ---
                        if len(rgb_frames) < 2:
                            print("⚠️ Not enough frames for temporal visualization")
                            return
                        
                        mid_idx = len(rgb_frames) // 2
                        frame_pair = rgb_frames[mid_idx:mid_idx+2]
                        flow = self.data_handler.compute_optical_flow(frame_pair)
                        
                        if len(flow) == 0:
                            print("⚠️ No flow computed for visualization")
                            return
                        
                        patches = self.data_handler.extract_patches(flow, is_flow=True)
                        if len(patches) == 0:
                            print("⚠️ No patches for temporal visualization")
                            return
                        
                        sample = patches[0]
                        model = self.models['temporal'].encoder
                        
                        # --- Move these lines INSIDE the temporal block ---
                        layer_outputs = [layer.output for layer in model.layers if 'conv' in layer.name]
                        activation_model = Model(inputs=model.input, outputs=layer_outputs)
                        activations = activation_model.predict(np.expand_dims(sample, 0))
                        
                        # Plotting code
                        first_layer_activation = activations[0][0]
                        plt.figure(figsize=(15, 5))
                        plt.suptitle(f'Temporal Feature Maps - Layer 1')
                        for i in range(min(8, first_layer_activation.shape[-1])):
                            plt.subplot(2, 4, i+1)
                            plt.imshow(first_layer_activation[..., i])
                            plt.axis('off')
                        plt.savefig(os.path.join(output_dir, 'temporal_feature_maps.png'))
                        plt.close()
            
                def _visualize_example_frame(self, frame, model_type, output_dir, idx):
                    """Generate comprehensive visualization for example frame"""
                    plt.figure(figsize=(18, 6))
                    
                    # Original Frame
                    plt.subplot(1, 4, 1)
                    plt.imshow(frame)
                    plt.title('Original Frame')
                    
                    # Spatial Reconstruction
                    if model_type in ['spatial', 'fusion']:
                        spatial_patches = self.data_handler.extract_patches([frame])
                        spatial_recon = self.models['spatial'].predict(spatial_patches)[0]
                        plt.subplot(1, 4, 2)
                        plt.imshow(spatial_recon)
                        plt.title('Spatial Reconstruction')
                    
                    # Temporal Reconstruction
                    if model_type in ['temporal', 'fusion'] and idx > 0:
                        flow = self.data_handler.compute_optical_flow([frame])[0]
                        temporal_patches = self.data_handler.extract_patches([flow], is_flow=True)
                        temporal_recon = self.models['temporal'].predict(temporal_patches)[0]
                        plt.subplot(1, 4, 3)
                        plt.imshow(temporal_recon[..., 0], cmap='coolwarm')
                        plt.title('Temporal Reconstruction (Flow X)')
                    
                    # Fusion Heatmap
                    if model_type == 'fusion' and idx > 0:
                        spatial_patches = self.data_handler.extract_patches([frame])
                        flow = self.data_handler.compute_optical_flow([frame])[0]
                        temporal_patches = self.data_handler.extract_patches([flow], is_flow=True)
                        heatmap = self.models['fusion'].predict([spatial_patches, temporal_patches])
                        
                        plt.subplot(1, 4, 4)
                        plt.imshow(frame)
                        plt.imshow(heatmap[0,...,0], cmap='jet', alpha=0.5)
                        plt.title('Fusion Anomaly Heatmap')
                        plt.colorbar()
            
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, f'comprehensive_visualization_{idx}.png'))
                    plt.close()
            
            
            # =============================================
            # Main Execution
            # =============================================
            if __name__ == "__main__":
                os.makedirs(CONFIG['output_dir'], exist_ok=True)
                os.makedirs(os.path.join(CONFIG['output_dir'], 'models'), exist_ok=True)
                os.makedirs(os.path.join(CONFIG['output_dir'], 'training_vis'), exist_ok=True)
                
                # Initialize and train models
                trainer = TrainingManager(CONFIG)
                trainer.train_all_models()
                
                # NEW: Plot loss curves
                trainer.plot_loss_curves(os.path.join(CONFIG['output_dir'], 'loss_curves'))

                trainer.plot_training_curves(os.path.join(CONFIG['output_dir'], 'loss_curves'))

                
                # Evaluate trained models
                evaluator = UnifiedEvaluator(trainer.models, trainer.data_handler, CONFIG)
                evaluator.evaluate_all_models()
                
                # Save final report
                final_report_path = os.path.join(CONFIG['output_dir'], 'final_report.json')
                with open(final_report_path, 'w') as f:
                    json.dump(evaluator.metrics, f, indent=2)
                
                print("Training and evaluation complete.")
 
    finally:
        # Strategy cleanup
        if strategy:
            print("Cleaning up strategy...")
            # Explicitly close the strategy
            strategy._extended._container_strategy = None  # pylint: disable=protected-access
            tf.distribute.experimental_set_strategy(None)

            del strategy                    
