# AudioWidget Example Notebook

This document shows example notebook cells for using the AudioWidget.

## Setup Cell

```python
# Install dependencies if needed
# !pip install ipywidgets soundfile

# Import the widget
from mixing.audio.audio_widget import AudioWidget, ensure_wfsr
import numpy as np
from IPython.display import display
```

## Example 1: Load and Display Audio from File

```python
# Load an audio file
widget = AudioWidget("path/to/your/audio.wav")
display(widget)

# Now you can:
# 1. Click the Play button to listen
# 2. Click and drag on the waveform to select a region
# 3. The selection info will show at the bottom right
```

## Example 2: Create Synthetic Audio

```python
# Generate a simple sine wave (1 second at 440 Hz)
sr = 44100  # Sample rate
duration = 1.0  # seconds
t = np.linspace(0, duration, int(sr * duration))
wf = np.sin(2 * np.pi * 440 * t).astype(np.float32)

# Create widget
widget = AudioWidget((wf, sr))
display(widget)
```

## Example 3: Interactive Editing

```python
# After creating and displaying the widget above...
# User selects a region by clicking and dragging

# Zoom to the selected region (non-destructive)
widget.zoom()

# Or reset
widget.reset()

# Make another selection, then crop (destructive)
widget.crop()

# Check what you have
print(f"Audio duration: {widget.duration:.2f} seconds")
print(f"Sample rate: {widget.sr} Hz")
```

## Example 4: Programmatic Selection and Editing

```python
# Create widget
widget = AudioWidget("audio.wav")
display(widget)

# Set selection programmatically (in seconds)
widget.selection_start = 5.0
widget.selection_end = 10.0
widget.has_selection = True

# Now crop to that selection
widget.crop()

print(f"Cropped to 5s-10s. New duration: {widget.duration:.2f}s")
```

## Example 5: Apply Effects

```python
# Create or load audio
widget = AudioWidget("audio.wav")
display(widget)

# Apply fade-in (0.5 seconds)
widget.apply_fade_in(duration=0.5)

# Apply fade-out (1.0 seconds)
widget.apply_fade_out(duration=1.0)

print("Fades applied!")
```

## Example 6: Complete Editing Workflow

```python
# Load audio
widget = AudioWidget("podcast.wav")
display(widget)

# Step 1: Remove intro
# (User selects from 0s to 8s, which is the intro)
widget.crop()  # This keeps the selection, removing everything else
# Actually, let's be more explicit:

# Let me restart with a clearer workflow:
widget = AudioWidget("podcast.wav")
display(widget)
```

```python
# After displaying above, select the part you WANT TO KEEP (8s onwards)
# by clicking at 8s and dragging to the end

# Crop to keep only the selected portion
widget.crop()
print(f"Removed intro. Duration now: {widget.duration:.2f}s")
```

```python
# Step 2: Remove outro
# Select from the beginning up to where you want to cut
# For example, if the outro starts at 30 minutes in:
widget.selection_start = 0.0
widget.selection_end = 30 * 60  # 30 minutes in seconds
widget.has_selection = True

widget.crop()
print(f"Removed outro. Final duration: {widget.duration:.2f}s")
```

```python
# Step 3: Apply fades
widget.apply_fade_in(duration=1.0)
widget.apply_fade_out(duration=2.0)
print("Applied fades")
```

```python
# Step 4: Save the result
widget.save("podcast_edited.wav")
print("Saved edited podcast!")
```

## Example 7: Extract Multiple Segments

```python
# Load audio
from mixing.audio.audio_widget import AudioWidget

original_path = "long_recording.wav"

# Extract first segment (10s-20s)
widget1 = AudioWidget(original_path)
widget1.selection_start = 10.0
widget1.selection_end = 20.0
widget1.has_selection = True
widget1.crop()
widget1.save("segment_1.wav")

# Extract second segment (45s-60s)
widget2 = AudioWidget(original_path)
widget2.selection_start = 45.0
widget2.selection_end = 60.0
widget2.has_selection = True
widget2.crop()
widget2.save("segment_2.wav")

print("Extracted 2 segments!")
```

## Example 8: Working with the Selection

```python
widget = AudioWidget("audio.wav")
display(widget)

# After user makes a selection...

# Check if there's a selection
if widget.has_selection:
    print(f"Selection: {widget.selection_start:.2f}s to {widget.selection_end:.2f}s")
    print(f"Duration: {widget.selection_end - widget.selection_start:.2f}s")

    # Get the selected audio as a separate array
    selected_wf, selected_sr = widget.get_selection()
    print(f"Selected segment: {len(selected_wf)} samples")

    # You can now process this separately or save it
    # Save just the selection without modifying the widget
    import soundfile as sf
    sf.write("selection_only.wav", selected_wf, selected_sr)
else:
    print("No selection made")
```

## Example 9: Custom Appearance

```python
# Create widget with custom colors and size
widget = AudioWidget(
    "audio.wav",
    height=200,                      # Taller waveform
    waveform_color='#FF6B6B',       # Red waveform
    progress_color='#4ECDC4'         # Turquoise progress/playhead
)
display(widget)
```

## Example 10: Monitoring Widget State

```python
widget = AudioWidget("audio.wav")
display(widget)

# Check various states
print(f"Duration: {widget.duration:.2f}s")
print(f"Sample rate: {widget.sr} Hz")
print(f"Playing: {widget.is_playing}")
print(f"Current position: {widget.current_time:.2f}s")

# These values update as the user interacts
# You can use them to build custom UI or logic
```

## Example 11: Reset After Multiple Edits

```python
widget = AudioWidget("audio.wav")
display(widget)

# Make some edits
widget.selection_start = 1.0
widget.selection_end = 5.0
widget.has_selection = True
widget.crop()

print(f"After first crop: {widget.duration:.2f}s")

# Make more edits
widget.apply_fade_in(0.5)

# Oops, want to start over!
widget.reset()

print(f"After reset: {widget.duration:.2f}s (back to original)")
```

## Example 12: Stereo Audio

```python
# Create stereo audio (different tones in each channel)
sr = 44100
duration = 2.0
t = np.linspace(0, duration, int(sr * duration))

# Left channel: 440 Hz
left = 0.3 * np.sin(2 * np.pi * 440 * t)

# Right channel: 880 Hz
right = 0.3 * np.sin(2 * np.pi * 880 * t)

# Combine into stereo
stereo_wf = np.column_stack([left, right]).astype(np.float32)

# Create widget (all operations work with stereo)
widget = AudioWidget((stereo_wf, sr))
display(widget)

print(f"Stereo audio shape: {widget.wf.shape}")  # (samples, 2)
```

## Example 13: Quick Audio Inspection

```python
# Load and immediately display properties
widget = AudioWidget("mystery_audio.wav")

print(f"Duration: {widget.duration:.2f} seconds")
print(f"Sample rate: {widget.sr} Hz")
print(f"Channels: {1 if widget.wf.ndim == 1 else widget.wf.shape[1]}")
print(f"Samples: {len(widget.wf):,}")

display(widget)
```

## Tips and Tricks

### Tip 1: Selection Precision

For precise selections, you can set times programmatically:

```python
widget.selection_start = 2.5
widget.selection_end = 2.7  # Precisely 200ms
widget.has_selection = True
widget.crop()
```

### Tip 2: Non-destructive Preview

Use `zoom()` instead of `crop()` to preview a section without modifying the audio:

```python
# User selects region
widget.zoom()  # Preview

# If you like it, reset and crop
widget.reset()
# Make same selection again
widget.crop()  # Now it's permanent
```

### Tip 3: Save Selection Without Cropping

```python
if widget.has_selection:
    selected_wf, selected_sr = widget.get_selection()
    import soundfile as sf
    sf.write("just_selection.wav", selected_wf, selected_sr)
    # Original widget is unchanged!
```

### Tip 4: Chain Multiple Crops

```python
# Progressive refinement
widget = AudioWidget("long_audio.wav")

# First pass: get rough section
widget.selection_start = 60.0
widget.selection_end = 180.0
widget.has_selection = True
widget.crop()

# Second pass: fine-tune
widget.selection_start = 5.0
widget.selection_end = 95.0
widget.has_selection = True
widget.crop()

# Result: samples 65s-175s from original
```

## Troubleshooting in Notebooks

### Widget Not Displaying?

```python
# Make sure ipywidgets is installed and enabled
# !pip install ipywidgets
# !jupyter nbextension enable --py widgetsnbextension

# For JupyterLab:
# !jupyter labextension install @jupyter-widgets/jupyterlab-manager
```

### JavaScript Not Loading?

```python
# Manually initialize
from mixing.audio.audio_widget import initialize_widget_javascript
initialize_widget_javascript()
```

### "soundfile not installed"?

```python
# Install soundfile
# !pip install soundfile
```

## Next Steps

- Experiment with different audio files
- Try the demo script: `python mixing/audio/demo_audio_widget.py`
- Read the full documentation: `mixing/audio/README_audio_widget.md`
- Run the tests: `pytest mixing/audio/test_audio_widget.py`
