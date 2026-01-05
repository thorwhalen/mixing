# AudioWidget - Interactive Audio Editor for Jupyter

An interactive, extensible audio widget for Jupyter Notebooks with DAW-like features.

## Features

- **Interactive Waveform Visualization**: Beautiful waveform display using Wavesurfer.js
- **Playback Controls**: Play/pause with synchronized playhead
- **Time-based Selection**: Click and drag to select audio segments
- **Zoom Operation**: Focus view on selected segment
- **Crop Operation**: Trim audio to selected segment
- **Audio Effects**: Fade in/out support
- **Extensible Architecture**: Easy to add custom visualizations and operations

## Installation

```bash
# Install with widget support
pip install -e ".[widget]"

# Or install dependencies manually
pip install ipywidgets soundfile jupyter
```

## Quick Start

### Basic Usage

```python
from mixing.audio.audio_widget import AudioWidget

# Create widget from file path
widget = AudioWidget("my_audio.wav")
display(widget)

# Or from (waveform, sample_rate) tuple
import numpy as np
sr = 44100
t = np.linspace(0, 1, sr)
wf = np.sin(2 * np.pi * 440 * t).astype(np.float32)
widget = AudioWidget((wf, sr))
display(widget)
```

### Interactive Operations

```python
# User selects a region by clicking and dragging on the waveform

# Zoom to selected region (non-destructive view change)
widget.zoom()

# Crop to selected region (destructive operation)
widget.crop()

# Reset to original audio
widget.reset()

# Get current waveform data
wf, sr = widget.get_waveform()

# Get selected segment
if widget.has_selection:
    selected_wf, selected_sr = widget.get_selection()
```

### Saving Audio

```python
# Save current audio
widget.save("output.wav")

# Save with specific format
widget.save("output.mp3", format="mp3")
```

### Audio Effects

```python
# Apply fade-in (1 second)
widget.apply_fade_in(duration=1.0)

# Apply fade-out (2 seconds)
widget.apply_fade_out(duration=2.0)
```

### Customization

```python
# Customize appearance
widget = AudioWidget(
    "audio.wav",
    height=200,                    # Waveform height in pixels
    waveform_color='#FF6B6B',     # Color of waveform
    progress_color='#4A90E2'       # Color of playhead
)
```

## Complete Workflow Example

```python
from mixing.audio.audio_widget import AudioWidget

# 1. Load audio
widget = AudioWidget("podcast.wav")
display(widget)

# 2. User clicks and drags to select intro section (0s - 5s)

# 3. Crop to remove intro
widget.crop()

# 4. Apply fade-in
widget.apply_fade_in(duration=0.5)

# 5. User selects outro section

# 6. Apply fade-out before saving
widget.apply_fade_out(duration=2.0)

# 7. Save final result
widget.save("podcast_edited.wav")
```

## Data Model

The widget uses a standardized data format:
- **Waveform (wf)**: NumPy array with shape `(n_samples,)` for mono or `(n_samples, n_channels)` for stereo
- **Sample Rate (sr)**: Integer representing Hz (e.g., 44100)

### The `ensure_wfsr` Utility Function

Converts various input types to standardized `(wf, sr)` tuple:

```python
from mixing.audio.audio_widget import ensure_wfsr

# From file path (string or Path)
wf, sr = ensure_wfsr("audio.wav")
wf, sr = ensure_wfsr(Path("audio.mp3"))

# From raw bytes
audio_bytes = Path("audio.wav").read_bytes()
wf, sr = ensure_wfsr(audio_bytes)

# From existing tuple (pass-through)
wf, sr = ensure_wfsr((wf, sr))
```

## Advanced Usage

### Programmatic Selection

```python
# Set selection programmatically
widget.selection_start = 2.5  # seconds
widget.selection_end = 5.0    # seconds
widget.has_selection = True

# Now perform operations
widget.zoom()
```

### Accessing Widget State

```python
# Check selection state
print(f"Has selection: {widget.has_selection}")
print(f"Selection: {widget.selection_start}s - {widget.selection_end}s")

# Check playback state
print(f"Playing: {widget.is_playing}")
print(f"Current time: {widget.current_time}s")
print(f"Duration: {widget.duration}s")

# Get audio properties
print(f"Sample rate: {widget.sr} Hz")
print(f"Samples: {len(widget.wf)}")
```

### Multiple Operations

```python
# Chain multiple operations
widget = AudioWidget("song.wav")

# Extract middle section
widget.selection_start = 30.0
widget.selection_end = 60.0
widget.has_selection = True
widget.crop()

# Further refine selection
widget.selection_start = 5.0
widget.selection_end = 20.0
widget.has_selection = True
widget.crop()

# Apply effects
widget.apply_fade_in(1.0)
widget.apply_fade_out(2.0)

# Save result
widget.save("song_excerpt.wav")
```

## Architecture

### Frontend (JavaScript)
- **Wavesurfer.js**: Handles waveform rendering and interaction
- **Custom ipywidget view**: Bridges Python and JavaScript
- **Event handling**: Captures user interactions (selection, playback)

### Backend (Python)
- **ipywidgets.DOMWidget**: Base class for widget
- **Traitlets**: Synchronizes state between Python and JavaScript
- **soundfile**: Handles audio I/O operations
- **NumPy**: Audio processing and manipulation

### Data Flow
1. User loads audio → `ensure_wfsr` → standardized `(wf, sr)`
2. Widget converts to base64 WAV → sent to JavaScript
3. Wavesurfer.js renders waveform → user interacts
4. Selection events → synced to Python via traitlets
5. Python operations (zoom/crop) → update waveform → refresh display

## Extensibility

The architecture is designed for easy extension:

### Adding Custom Operations

```python
def apply_custom_effect(widget, param1, param2):
    """Apply custom audio effect."""
    wf = widget.wf
    sr = widget.sr

    # Your signal processing here
    processed_wf = custom_processing(wf, sr, param1, param2)

    # Update widget
    widget.wf = processed_wf
    widget._update_audio_data()

# Use it
widget = AudioWidget("audio.wav")
apply_custom_effect(widget, param1=0.5, param2=10)
```

### Future Roadmap

Planned features for future versions:
- Multi-track display with synchronized selections
- Segment playback (play only selected region)
- Copy/paste segments
- Undo/redo functionality
- Spectrogram visualization
- Annotation and markers
- Real-time effects preview

## Testing

Run the comprehensive test suite:

```bash
# Run all tests
pytest mixing/audio/test_audio_widget.py -v

# Run specific test class
pytest mixing/audio/test_audio_widget.py::TestEnsureWfsr -v

# Run with coverage
pytest mixing/audio/test_audio_widget.py --cov=mixing.audio.audio_widget --cov-report=html
```

## Troubleshooting

### "soundfile not installed" error
```bash
pip install soundfile
```

### Widget not displaying
Make sure you have ipywidgets enabled:
```bash
jupyter nbextension enable --py widgetsnbextension
# For JupyterLab
jupyter labextension install @jupyter-widgets/jupyterlab-manager
```

### JavaScript not loading
The widget auto-initializes when imported. If issues persist, manually initialize:
```python
from mixing.audio.audio_widget import initialize_widget_javascript
initialize_widget_javascript()
```

### Selection not working
- Make sure you're clicking and dragging on the waveform area
- The selection is shown with a semi-transparent red overlay
- Selection info appears in the bottom-right of the widget

## API Reference

### `ensure_wfsr(audio_input) -> (wf, sr)`
Convert various audio inputs to standardized format.

### `AudioWidget(audio_input, height=128, waveform_color='#4A90E2', progress_color='#FF6B6B')`
Main widget class.

**Methods:**
- `zoom()`: Zoom to selected region
- `crop()`: Crop to selected region
- `reset()`: Reset to original audio
- `get_waveform()`: Get current (wf, sr)
- `get_selection()`: Get selected segment
- `save(filepath, format=None)`: Save audio
- `apply_fade_in(duration=1.0)`: Apply fade-in
- `apply_fade_out(duration=1.0)`: Apply fade-out

**Properties:**
- `wf`: Current waveform array
- `sr`: Sample rate
- `duration`: Audio duration in seconds
- `has_selection`: Whether a region is selected
- `selection_start`: Selection start time in seconds
- `selection_end`: Selection end time in seconds
- `is_playing`: Playback state
- `current_time`: Current playback position

## License

MIT License - See project LICENSE file

## Contributing

Contributions welcome! Areas of interest:
- Additional audio effects
- Multi-track support
- Spectrogram visualization
- Performance optimizations
- Documentation improvements
