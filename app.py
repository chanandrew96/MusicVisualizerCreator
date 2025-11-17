import os
import tempfile
from flask import Flask, request, render_template, send_file, jsonify
from werkzeug.utils import secure_filename
import numpy as np
from PIL import Image, ImageFilter
import librosa
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import moviepy.editor as mpy
from moviepy.video.io.bindings import mplfig_to_npimage

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'mp3', 'wav', 'm4a', 'flac', 'ogg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_audio(audio_path):
    """Load and analyze audio file"""
    y, sr = librosa.load(audio_path)
    duration = librosa.get_duration(y=y, sr=sr)
    return y, sr, duration

def create_frame(t, y, sr, duration, width, height, bg_image=None, thumbnail=None):
    """Create a single frame of the video"""
    # Create base image
    if bg_image is not None:
        frame = bg_image.copy()
    else:
        frame = Image.new('RGB', (width, height), color='black')
    
    # Convert to numpy array for drawing
    frame_array = np.array(frame)
    
    # Calculate current audio position
    current_sample = int(t * sr)
    if current_sample > len(y):
        current_sample = len(y)
    
    # Get audio segment for visualization (last 5 seconds or available)
    window_seconds = 5.0
    start_sample = max(0, current_sample - int(window_seconds * sr))
    audio_segment = y[start_sample:current_sample]
    
    if len(audio_segment) > 0:
        # Create matplotlib figure for waveform
        dpi = 100
        fig, ax = plt.subplots(figsize=(width/dpi, height/dpi), facecolor='black', dpi=dpi)
        ax.set_facecolor('black')
        ax.set_xlim(0, window_seconds)
        ax.set_ylim(-1, 1)
        ax.axis('off')
        
        # Calculate time axis
        time_axis = np.linspace(0, len(audio_segment) / sr, len(audio_segment))
        
        # Draw waveform
        ax.plot(time_axis, audio_segment, color='cyan', linewidth=3, alpha=0.9)
        
        # Fill area
        ax.fill_between(time_axis, audio_segment, 0, color='cyan', alpha=0.4)
        ax.fill_between(time_axis, audio_segment, 0, where=(audio_segment >= 0), 
                        color='cyan', alpha=0.6)
        
        # Convert to image using moviepy helper
        fig.canvas.draw()
        buf = mplfig_to_npimage(fig)
        plt.close(fig)
        
        # Resize to match frame dimensions
        waveform_img = Image.fromarray(buf)
        waveform_img = waveform_img.resize((width, height), Image.Resampling.LANCZOS)
        waveform_array = np.array(waveform_img)
        
        # Blend waveform with background
        alpha = 0.7
        frame_array = (alpha * waveform_array + (1 - alpha) * frame_array).astype(np.uint8)
    
    # Add thumbnail in corner if provided
    if thumbnail is not None:
        thumb_size = (width // 6, height // 6)
        thumb_resized = thumbnail.copy()
        thumb_resized.thumbnail(thumb_size, Image.Resampling.LANCZOS)
        
        # Create thumbnail with border
        thumb_with_border = Image.new('RGB', 
                                     (thumb_resized.width + 10, thumb_resized.height + 10), 
                                     'white')
        thumb_with_border.paste(thumb_resized, (5, 5))
        
        # Paste thumbnail in top-right corner
        thumb_array = np.array(thumb_with_border)
        x_offset = width - thumb_with_border.width - 20
        y_offset = 20
        frame_array[y_offset:y_offset+thumb_with_border.height, 
                   x_offset:x_offset+thumb_with_border.width] = thumb_array
    
    return Image.fromarray(frame_array)

def create_video_with_thumbnail(audio_path, image_path=None, output_path=None, 
                                width=1920, height=1080, format='mp4'):
    """Create video with audio waveform and optional image background"""
    
    # Process audio
    y, sr, duration = process_audio(audio_path)
    
    # Load and process background image
    bg_image = None
    thumbnail = None
    if image_path and os.path.exists(image_path):
        img = Image.open(image_path)
        # Resize and blur for background
        bg_image = img.resize((width, height), Image.Resampling.LANCZOS)
        bg_image = bg_image.filter(ImageFilter.GaussianBlur(radius=15))
        
        # Create thumbnail (smaller, no blur)
        thumbnail = img.copy()
        thumbnail.thumbnail((width // 4, height // 4), Image.Resampling.LANCZOS)
    
    # Create video frames function
    fps = 30
    
    def make_frame(t):
        frame = create_frame(t, y, sr, duration, width, height, bg_image, thumbnail)
        return np.array(frame)
    
    # Create video clip
    clip = mpy.VideoClip(make_frame, duration=duration)
    clip = clip.set_fps(fps)
    
    # Add audio
    audio_clip = mpy.AudioFileClip(audio_path)
    clip = clip.set_audio(audio_clip)
    
    # Write video file
    if not output_path:
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], f'output.{format}')
    
    if format == 'mp4':
        clip.write_videofile(output_path, fps=fps, codec='libx264', audio_codec='aac', 
                           preset='medium', bitrate='8000k', threads=4, logger=None)
    elif format == 'webm':
        clip.write_videofile(output_path, fps=fps, codec='libvpx-vp9', audio_codec='libopus',
                           bitrate='8000k', logger=None)
    elif format == 'avi':
        clip.write_videofile(output_path, fps=fps, codec='libx264', audio_codec='aac',
                           logger=None)
    else:
        clip.write_videofile(output_path, fps=fps, logger=None)
    
    clip.close()
    audio_clip.close()
    return output_path

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    try:
        # Check if audio file is present
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        audio_file = request.files['audio']
        if audio_file.filename == '':
            return jsonify({'error': 'No audio file selected'}), 400
        
        if not allowed_file(audio_file.filename):
            return jsonify({'error': 'Invalid audio file format'}), 400
        
        # Handle optional image file
        image_file = request.files.get('image')
        image_path = None
        if image_file and image_file.filename != '':
            if not allowed_file(image_file.filename):
                return jsonify({'error': 'Invalid image file format'}), 400
            
            # Save image temporarily
            image_filename = secure_filename(image_file.filename)
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_filename)
            image_file.save(image_path)
        
        # Get parameters
        width = int(request.form.get('width', 1920))
        height = int(request.form.get('height', 1080))
        format = request.form.get('format', 'mp4')
        
        # Validate format
        if format not in ['mp4', 'webm', 'avi', 'mov']:
            format = 'mp4'
        
        # Validate dimensions
        if width < 320 or width > 7680:
            width = 1920
        if height < 240 or height > 4320:
            height = 1080
        
        # Save audio temporarily
        audio_filename = secure_filename(audio_file.filename)
        audio_path = os.path.join(app.config['UPLOAD_FOLDER'], audio_filename)
        audio_file.save(audio_path)
        
        # Generate output filename
        output_filename = f'output.{format}'
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
        
        # Create video
        result_path = create_video_with_thumbnail(
            audio_path, 
            image_path, 
            output_path,
            width=width,
            height=height,
            format=format
        )
        
        # Clean up temporary files
        if os.path.exists(audio_path):
            os.remove(audio_path)
        if image_path and os.path.exists(image_path):
            os.remove(image_path)
        
        # Return video file
        return send_file(
            result_path,
            mimetype=f'video/{format}',
            as_attachment=True,
            download_name=output_filename
        )
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
