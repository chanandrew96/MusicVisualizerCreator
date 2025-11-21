import os
import tempfile
import math
import uuid
import threading
from flask import Flask, request, render_template, send_file, jsonify
from werkzeug.utils import secure_filename
import numpy as np
from PIL import Image, ImageFilter, ImageOps, ImageEnhance, ImageDraw, ImageFont
import librosa
import moviepy.editor as mpy

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()

# 进度跟踪存储
task_progress = {}
task_lock = threading.Lock()

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'mp3', 'wav', 'm4a', 'flac', 'ogg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_audio(audio_path):
    """Load and analyze audio file"""
    y, sr = librosa.load(audio_path)
    duration = librosa.get_duration(y=y, sr=sr)
    return y, sr, duration

def normalize_feature(arr, eps=1e-9):
    arr = arr.astype(np.float32)
    min_val = np.min(arr)
    max_val = np.max(arr)
    if max_val - min_val < eps:
        return np.zeros_like(arr)
    return (arr - min_val) / (max_val - min_val)


def generate_particles(count, width, height, seed=42):
    rng = np.random.default_rng(seed)
    particles = []
    for _ in range(count):
        particles.append({
            'base_x': rng.uniform(0.1 * width, 0.9 * width),
            'base_y': rng.uniform(0.1 * height, 0.9 * height),
            'speed': rng.uniform(0.3, 1.2),
            'radius': rng.uniform(1.0, 3.0),
            'phase': rng.uniform(0, 2 * np.pi)
        })
    return particles


def prepare_visual_context(y, sr, width, height, bg_image=None, thumbnail=None):
    hop_length = 512
    n_fft = 2048
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=96, fmax=sr / 2
    )
    mel_db = librosa.power_to_db(mel + 1e-9)
    mel_norm = normalize_feature(mel_db)

    rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop_length)[0]
    rms_norm = normalize_feature(rms)

    if thumbnail is not None:
        thumb_size = min(width, height) // 3
        thumb_square = ImageOps.fit(thumbnail, (thumb_size, thumb_size), Image.Resampling.LANCZOS)
        mask = Image.new('L', (thumb_size, thumb_size), 0)
        mask_draw = ImageDraw.Draw(mask)
        mask_draw.ellipse((0, 0, thumb_size, thumb_size), fill=255)
        thumb_circle = Image.new('RGBA', (thumb_size, thumb_size))
        thumb_circle.paste(thumb_square, (0, 0), mask)
        thumbnail_processed = thumb_circle
    else:
        thumbnail_processed = None

    particles = generate_particles(120, width, height)

    return {
        'y': y,
        'sr': sr,
        'width': width,
        'height': height,
        'hop_length': hop_length,
        'mel': mel_norm,
        'rms': rms_norm,
        'bg_image': bg_image,
        'thumbnail': thumbnail_processed,
        'particles': particles
    }


def sample_feature_at_time(feature, t_idx):
    if feature.ndim == 2:
        num = feature.shape[1]
    else:
        num = feature.shape[0]
    idx = max(0, min(num - 1, t_idx))
    if feature.ndim == 2:
        return feature[:, idx]
    return feature[idx]


def build_gradient(width, height, progress, intensity):
    color_sets = [
        ((46, 4, 74), (16, 76, 140)),
        ((18, 57, 130), (237, 30, 121)),
        ((255, 77, 90), (255, 184, 108))
    ]
    idx = int(progress * len(color_sets)) % len(color_sets)
    start_color, end_color = color_sets[idx]
    start = np.array(start_color, dtype=np.float32)
    end = np.array(end_color, dtype=np.float32)
    gradient = np.zeros((height, width, 3), dtype=np.float32)
    vertical = np.linspace(0, 1, height)[:, None]
    gradient = start + (end - start) * vertical
    gradient = np.clip(gradient * (0.7 + 0.3 * intensity), 0, 255)
    return Image.fromarray(gradient.astype(np.uint8))


def overlay_particles(image, particles, t, amplitude):
    layer = Image.new('RGBA', image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer, 'RGBA')
    width, height = image.size
    for particle in particles:
        offset = math.sin(t * particle['speed'] + particle['phase']) * 60 * (0.3 + amplitude)
        x = (particle['base_x'] + offset) % width
        y = (particle['base_y'] + offset) % height
        alpha = int(120 + 80 * amplitude)
        radius = particle['radius'] * (1.0 + amplitude * 0.8)
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill=(255, 255, 255, alpha)
        )
    layer = layer.filter(ImageFilter.GaussianBlur(radius=2))
    image.alpha_composite(layer)


def draw_radial_spectrum(image, spectrum, amplitude):
    width, height = image.size
    center = (width // 2, height // 2)
    num_bars = 120
    indices = np.linspace(0, len(spectrum) - 1, num_bars).astype(int)
    bars = spectrum[indices]
    bars = (bars ** 1.5)  # emphasize peaks
    bars = normalize_feature(bars)
    base_radius = min(width, height) * 0.18
    max_height = min(width, height) * 0.28 * (0.6 + amplitude)

    layer = Image.new('RGBA', image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer, 'RGBA')
    for i, value in enumerate(bars):
        angle = (2 * np.pi / num_bars) * i
        length = base_radius + value * max_height
        inner = (
            center[0] + base_radius * math.cos(angle),
            center[1] + base_radius * math.sin(angle)
        )
        outer = (
            center[0] + length * math.cos(angle),
            center[1] + length * math.sin(angle)
        )
        color = (
            int(255 * (0.4 + 0.6 * value)),
            int(120 + 100 * value),
            255,
            int(180 + 60 * amplitude)
        )
        draw.line([inner, outer], fill=color, width=6)
    blurred = layer.filter(ImageFilter.GaussianBlur(radius=6))
    image.alpha_composite(blurred)
    image.alpha_composite(layer)


def add_centerpiece(image, thumbnail, amplitude):
    if thumbnail is None:
        return
    width, height = image.size
    layer = Image.new('RGBA', image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer, 'RGBA')
    center = (width // 2, height // 2)
    radius = thumbnail.width // 2
    glow_radius = int(radius * (1.2 + amplitude * 0.2))
    draw.ellipse(
        (
            center[0] - glow_radius,
            center[1] - glow_radius,
            center[0] + glow_radius,
            center[1] + glow_radius
        ),
        fill=(255, 255, 255, 90)
    )
    glow = layer.filter(ImageFilter.GaussianBlur(radius=25))
    image.alpha_composite(glow)
    paste_pos = (center[0] - radius, center[1] - radius)
    image.paste(thumbnail, paste_pos, thumbnail)


def add_hud(draw, width, height, t, duration):
    progress = t / max(duration, 1e-6)
    bar_width = int(width * 0.6)
    bar_x = (width - bar_width) // 2
    bar_y = int(height * 0.82)
    draw.rectangle(
        (bar_x, bar_y, bar_x + bar_width, bar_y + 8),
        fill=(255, 255, 255, 80), outline=None
    )
    draw.rectangle(
        (bar_x, bar_y, bar_x + int(bar_width * progress), bar_y + 8),
        fill=(255, 255, 255, 220), outline=None
    )
    minutes = int(t // 60)
    seconds = int(t % 60)
    timestamp = f"{minutes:02d}:{seconds:02d}"
    font = ImageFont.load_default()
    draw.text(
        (bar_x, bar_y - 20),
        "Now Playing",
        fill=(255, 255, 255, 240),
        font=font
    )
    draw.text(
        (bar_x + bar_width - 60, bar_y - 20),
        timestamp,
        fill=(255, 255, 255, 240),
        font=font
    )


def create_frame(t, context, duration):
    width = context['width']
    height = context['height']
    sr = context['sr']
    hop_length = context['hop_length']
    mel = context['mel']
    rms = context['rms']

    frame_idx = min(int(t * sr / hop_length), rms.shape[0] - 1)
    spectrum = sample_feature_at_time(mel, frame_idx)
    amplitude = float(sample_feature_at_time(rms, frame_idx))

    progress = (t / duration) % 1.0
    base = build_gradient(width, height, progress, amplitude)

    if context['bg_image'] is not None:
        bg = context['bg_image'].copy().convert('RGBA')
        bg = ImageEnhance.Brightness(bg).enhance(0.5 + amplitude * 0.4)
        base = Image.blend(base.convert('RGB'), bg.convert('RGB'), 0.55)

    frame = base.convert('RGBA')
    overlay_particles(frame, context['particles'], t, amplitude)
    draw_radial_spectrum(frame, spectrum, amplitude)
    add_centerpiece(frame, context['thumbnail'], amplitude)

    hud_layer = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    hud_draw = ImageDraw.Draw(hud_layer, 'RGBA')
    add_hud(hud_draw, width, height, t, duration)
    frame.alpha_composite(hud_layer)

    return frame.convert('RGB')

def update_progress(task_id, status, progress=0, message=""):
    """更新任务进度"""
    with task_lock:
        task_progress[task_id] = {
            'status': status,  # 'processing', 'completed', 'error'
            'progress': progress,  # 0-100
            'message': message
        }

def create_video_with_thumbnail(audio_path, image_path=None, output_path=None, 
                                width=1920, height=1080, format='mp4', 
                                task_id=None, preview_duration=None):
    """Create video with audio waveform and optional image background"""
    
    try:
        if task_id:
            update_progress(task_id, 'processing', 5, '正在加载音频文件...')
        
        # Process audio
        y, sr, duration = process_audio(audio_path)
        
        # 如果是预览模式，限制时长
        if preview_duration:
            duration = min(duration, preview_duration)
            y = y[:int(duration * sr)]
        
        if task_id:
            update_progress(task_id, 'processing', 15, '正在处理图像...')
        
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

        if task_id:
            update_progress(task_id, 'processing', 25, '正在分析音频特征...')

        visual_context = prepare_visual_context(y, sr, width, height, bg_image, thumbnail)
        
        if task_id:
            update_progress(task_id, 'processing', 40, '正在生成视频帧...')
        
        # Create video frames function
        fps = 30
        total_frames = int(duration * fps)
        frame_count = [0]
        
        def make_frame(t):
            frame = create_frame(t, visual_context, duration)
            frame_count[0] += 1
            if task_id and total_frames > 0:
                frame_progress = 40 + int((frame_count[0] / total_frames) * 30)
                update_progress(task_id, 'processing', frame_progress, 
                              f'正在渲染帧 {frame_count[0]}/{total_frames}...')
            return np.array(frame)
        
        if task_id:
            update_progress(task_id, 'processing', 70, '正在创建视频剪辑...')
        
        # Create video clip
        clip = mpy.VideoClip(make_frame, duration=duration)
        clip = clip.set_fps(fps)
        
        if task_id:
            update_progress(task_id, 'processing', 80, '正在添加音频...')
        
        # Add audio
        audio_clip = mpy.AudioFileClip(audio_path)
        if preview_duration:
            audio_clip = audio_clip.subclip(0, duration)
        clip = clip.set_audio(audio_clip)
        
        # Write video file
        if not output_path:
            output_path = os.path.join(app.config['UPLOAD_FOLDER'], f'output.{format}')
        
        if task_id:
            update_progress(task_id, 'processing', 85, '正在编码视频...')
        
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
        
        if task_id:
            update_progress(task_id, 'completed', 100, '视频生成完成！')
        
        return output_path
    except Exception as e:
        if task_id:
            update_progress(task_id, 'error', 0, f'错误: {str(e)}')
        raise

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/progress/<task_id>', methods=['GET'])
def get_progress(task_id):
    """获取任务进度"""
    with task_lock:
        if task_id in task_progress:
            return jsonify(task_progress[task_id])
        return jsonify({'status': 'not_found', 'progress': 0, 'message': '任务不存在'}), 404

def process_video_generation(audio_path, image_path, width, height, format, task_id, preview_duration=None):
    """在后台线程中处理视频生成"""
    try:
        output_filename = f'output_{task_id}.{format}'
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
        
        result_path = create_video_with_thumbnail(
            audio_path, 
            image_path, 
            output_path,
            width=width,
            height=height,
            format=format,
            task_id=task_id,
            preview_duration=preview_duration
        )
        
        with task_lock:
            if task_id in task_progress:
                task_progress[task_id]['output_path'] = result_path
    except Exception as e:
        with task_lock:
            if task_id in task_progress:
                task_progress[task_id]['status'] = 'error'
                task_progress[task_id]['message'] = f'错误: {str(e)}'
                task_progress[task_id]['error'] = str(e)
    finally:
        # 清理临时音频和图片文件（但不删除生成的视频）
        try:
            if os.path.exists(audio_path):
                os.remove(audio_path)
            if image_path and os.path.exists(image_path):
                os.remove(image_path)
        except:
            pass

@app.route('/preview', methods=['POST'])
def preview():
    """生成预览视频（5-10秒）"""
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
        preview_duration = float(request.form.get('preview_duration', 8.0))  # 默认8秒
        
        # Validate format
        if format not in ['mp4', 'webm', 'avi', 'mov']:
            format = 'mp4'
        
        # Validate dimensions
        if width < 320 or width > 7680:
            width = 1920
        if height < 240 or height > 4320:
            height = 1080
        
        # Validate preview duration
        if preview_duration < 1 or preview_duration > 10:
            preview_duration = 8.0
        
        # Save audio temporarily
        audio_filename = secure_filename(audio_file.filename)
        audio_path = os.path.join(app.config['UPLOAD_FOLDER'], audio_filename)
        audio_file.save(audio_path)
        
        # Generate task ID
        task_id = str(uuid.uuid4())
        update_progress(task_id, 'processing', 0, '准备生成预览...')
        
        # Start video generation in background thread
        thread = threading.Thread(
            target=process_video_generation,
            args=(audio_path, image_path, width, height, format, task_id, preview_duration)
        )
        thread.daemon = True
        thread.start()
        
        return jsonify({'task_id': task_id, 'message': '预览生成已开始'})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/generate', methods=['POST'])
def generate():
    """生成完整视频"""
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
        
        # Generate task ID
        task_id = str(uuid.uuid4())
        update_progress(task_id, 'processing', 0, '准备生成完整视频...')
        
        # Start video generation in background thread
        thread = threading.Thread(
            target=process_video_generation,
            args=(audio_path, image_path, width, height, format, task_id, None)
        )
        thread.daemon = True
        thread.start()
        
        return jsonify({'task_id': task_id, 'message': '视频生成已开始'})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/download/<task_id>', methods=['GET'])
def download(task_id):
    """下载生成的视频"""
    with task_lock:
        if task_id not in task_progress:
            return jsonify({'error': '任务不存在'}), 404
        
        task = task_progress[task_id]
        if task['status'] != 'completed':
            return jsonify({'error': '视频尚未生成完成'}), 400
        
        if 'output_path' not in task or not os.path.exists(task['output_path']):
            return jsonify({'error': '视频文件不存在'}), 404
        
        output_path = task['output_path']
        format = os.path.splitext(output_path)[1][1:] or 'mp4'
        
        return send_file(
            output_path,
            mimetype=f'video/{format}',
            as_attachment=True,
            download_name=f'output.{format}'
        )

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
