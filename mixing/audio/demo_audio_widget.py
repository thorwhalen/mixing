"""
Demo script for AudioWidget - can be run in a Jupyter notebook.

This demonstrates the basic usage patterns and features of the AudioWidget.
"""

import numpy as np
from audio_widget import AudioWidget, ensure_wfsr

# ============================================================================
# Demo 1: Create a synthetic audio signal
# ============================================================================

def create_demo_audio(duration=5.0, sr=44100):
    """Create a demo audio signal with multiple frequencies."""
    t = np.linspace(0, duration, int(sr * duration))

    # Create a more interesting signal with multiple frequencies
    # Base tone at 440 Hz (A4)
    signal = 0.3 * np.sin(2 * np.pi * 440 * t)

    # Add harmonic at 880 Hz
    signal += 0.2 * np.sin(2 * np.pi * 880 * t)

    # Add some modulation
    envelope = 0.1 * np.sin(2 * np.pi * 2 * t)
    signal += envelope * np.sin(2 * np.pi * 660 * t)

    return signal.astype(np.float32), sr


print("Creating demo audio...")
wf, sr = create_demo_audio(duration=10.0)
print(f"Created {len(wf)} samples at {sr} Hz ({len(wf)/sr:.2f} seconds)")

# ============================================================================
# Demo 2: Create and display the widget
# ============================================================================

print("\nInitializing AudioWidget...")
widget = AudioWidget(
    (wf, sr),
    height=150,
    waveform_color='#2E86AB',  # Blue
    progress_color='#A23B72'    # Purple
)

print("\nWidget created! In a Jupyter notebook, you would display it with:")
print("display(widget)")
print("\nYou can then:")
print("1. Click and drag to select a region")
print("2. Click the Play button to hear the audio")
print("3. Call widget.zoom() to zoom to the selection")
print("4. Call widget.crop() to crop to the selection")

# ============================================================================
# Demo 3: Programmatic operations
# ============================================================================

print("\n" + "="*70)
print("DEMO: Programmatic operations")
print("="*70)

# Simulate selection (normally done by user in UI)
print("\nSimulating user selection from 2s to 5s...")
widget.selection_start = 2.0
widget.selection_end = 5.0
widget.has_selection = True

print(f"Selection: {widget.selection_start}s - {widget.selection_end}s")
print(f"Selection duration: {widget.selection_end - widget.selection_start}s")

# Get the selected audio
selected = widget.get_selection()
if selected:
    selected_wf, selected_sr = selected
    print(f"Selected segment: {len(selected_wf)} samples ({len(selected_wf)/selected_sr:.2f}s)")

# ============================================================================
# Demo 4: Zoom operation
# ============================================================================

print("\n" + "="*70)
print("DEMO: Zoom operation (non-destructive)")
print("="*70)

original_length = len(widget.wf)
print(f"Before zoom: {original_length} samples ({original_length/sr:.2f}s)")

widget.zoom()

print(f"After zoom: {len(widget.wf)} samples ({len(widget.wf)/sr:.2f}s)")
print(f"Zoomed to the selected region")

# ============================================================================
# Demo 5: Reset and crop
# ============================================================================

print("\n" + "="*70)
print("DEMO: Reset and Crop operation")
print("="*70)

# Reset to original
widget.reset()
print(f"After reset: {len(widget.wf)} samples ({len(widget.wf)/sr:.2f}s)")

# Make a new selection and crop
widget.selection_start = 3.0
widget.selection_end = 7.0
widget.has_selection = True

print(f"\nNew selection: {widget.selection_start}s - {widget.selection_end}s")
widget.crop()
print(f"After crop: {len(widget.wf)} samples ({len(widget.wf)/sr:.2f}s)")

# ============================================================================
# Demo 6: Audio effects
# ============================================================================

print("\n" + "="*70)
print("DEMO: Audio effects")
print("="*70)

# Apply fade-in
print("\nApplying 0.5s fade-in...")
widget.apply_fade_in(duration=0.5)

# Apply fade-out
print("Applying 1.0s fade-out...")
widget.apply_fade_out(duration=1.0)

# Check the result
print(f"First few samples (should be near zero): {widget.wf[:5]}")
print(f"Last few samples (should be near zero): {widget.wf[-5:]}")

# ============================================================================
# Demo 7: Save audio
# ============================================================================

print("\n" + "="*70)
print("DEMO: Saving audio")
print("="*70)

# In a real scenario, you would save like this:
print("\nTo save the audio, use:")
print('widget.save("output.wav")')
print('widget.save("output.mp3", format="mp3")')

# Uncomment to actually save:
# widget.save("demo_output.wav")

# ============================================================================
# Demo 8: Multiple operations workflow
# ============================================================================

print("\n" + "="*70)
print("DEMO: Complete workflow")
print("="*70)

# Start fresh
wf, sr = create_demo_audio(duration=10.0)
widget = AudioWidget((wf, sr))

print(f"1. Initial audio: {len(widget.wf)} samples ({len(widget.wf)/sr:.2f}s)")

# Extract middle section
widget.selection_start = 2.0
widget.selection_end = 8.0
widget.has_selection = True
widget.crop()
print(f"2. After cropping to 2s-8s: {len(widget.wf)} samples ({len(widget.wf)/sr:.2f}s)")

# Further refinement
widget.selection_start = 1.0
widget.selection_end = 4.0
widget.has_selection = True
widget.crop()
print(f"3. After second crop to 1s-4s: {len(widget.wf)} samples ({len(widget.wf)/sr:.2f}s)")

# Apply effects
widget.apply_fade_in(0.3)
widget.apply_fade_out(0.5)
print(f"4. Applied fade-in (0.3s) and fade-out (0.5s)")

print(f"5. Final audio ready: {len(widget.wf)} samples ({len(widget.wf)/sr:.2f}s)")

# ============================================================================
# Demo 9: Working with existing audio files
# ============================================================================

print("\n" + "="*70)
print("DEMO: Working with audio files")
print("="*70)

print("\nExample code for working with files:")
print("""
# Load from file
widget = AudioWidget("my_audio.wav")

# Or using ensure_wfsr first
wf, sr = ensure_wfsr("my_audio.mp3")
widget = AudioWidget((wf, sr))

# Save after editing
widget.save("edited_audio.wav")
""")

# ============================================================================
# Demo 10: Stereo audio
# ============================================================================

print("\n" + "="*70)
print("DEMO: Stereo audio support")
print("="*70)

# Create stereo audio
duration = 3.0
sr = 44100
t = np.linspace(0, duration, int(sr * duration))

# Left channel: 440 Hz
left = 0.3 * np.sin(2 * np.pi * 440 * t)

# Right channel: 880 Hz
right = 0.3 * np.sin(2 * np.pi * 880 * t)

# Combine into stereo
stereo_wf = np.column_stack([left, right]).astype(np.float32)

print(f"Created stereo audio: shape {stereo_wf.shape}")

widget_stereo = AudioWidget((stereo_wf, sr))
print(f"Stereo widget created: {widget_stereo.wf.shape}")

# All operations work with stereo
widget_stereo.selection_start = 0.5
widget_stereo.selection_end = 2.0
widget_stereo.has_selection = True
widget_stereo.crop()

print(f"After crop: {widget_stereo.wf.shape}")
print("All operations (zoom, crop, fade) work seamlessly with stereo audio!")

# ============================================================================
# Summary
# ============================================================================

print("\n" + "="*70)
print("DEMO COMPLETE")
print("="*70)
print("\nKey takeaways:")
print("1. AudioWidget works with (wf, sr) tuples or file paths")
print("2. Interactive selection via click-and-drag in the UI")
print("3. Programmatic operations: zoom(), crop(), reset()")
print("4. Audio effects: apply_fade_in(), apply_fade_out()")
print("5. Easy saving: save(filepath)")
print("6. Full stereo support")
print("7. Extensible architecture for custom operations")
print("\nFor interactive use, run this in a Jupyter notebook!")
print("See README_audio_widget.md for complete documentation.")
