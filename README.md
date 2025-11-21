# Music Visualizer Creator

一個可以將音頻文件轉換為帶有聲波可視化視頻的 Web 應用程序。支持可選的背景圖片（帶模糊效果）和縮略圖。

## 功能特點

- 🎵 **音頻處理**: 支持多種音頻格式 (MP3, WAV, M4A, FLAC, OGG)
- 🌀 **視覺風格**: 以 [YouTube 參考影片](https://www.youtube.com/watch?v=jPzkNvWOcGc) 為靈感，提供漸層背景、放射光柱、粒子以及圓形專輯封面中心件的沉浸式動態
- 🖼️ **背景圖片**: 可選的背景圖片，自動應用模糊效果並與漸層融合
- 📸 **縮略圖**: 圓形裁切的封面置中顯示與柔光同步脈動
- 📐 **自定義尺寸**: 支持自定義視頻寬度和高度
- 🎬 **多種格式**: 支持 MP4, WebM, AVI, MOV 輸出格式
- 🐳 **Docker 支持**: 可作為 Docker 容器運行
- ☁️ **雲端部署**: 可部署到 Render 等雲平台

## 技術棧

- **後端**: Flask (Python)
- **音頻處理**: librosa
- **圖像處理**: Pillow (PIL)
- **視頻生成**: MoviePy
- **可視化**: Matplotlib

## 安裝和運行

### 本地運行

1. 克隆或下載此項目

2. 安裝依賴:
```bash
pip install -r requirements.txt
```

3. 運行應用:
```bash
python app.py
```

4. 在瀏覽器中打開 `http://localhost:5000`

### 使用 Docker

1. 構建 Docker 鏡像:
```bash
docker build -t music-visualizer .
```

2. 運行容器:
```bash
docker run -p 5000:5000 music-visualizer
```

3. 在瀏覽器中打開 `http://localhost:5000`

### 部署到 Render

1. 將代碼推送到 GitHub
2. 在 Render 創建新的 Web Service
3. 連接您的 GitHub 倉庫
4. 使用以下設置:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python app.py`
   - **Environment**: Python 3

或者使用 Docker:
   - **Dockerfile Path**: `./Dockerfile`

## 使用方法

1. **上傳音頻文件** (必需)
   - 選擇一個音頻文件 (MP3, WAV, M4A, FLAC, OGG)

2. **上傳背景圖片** (可選)
   - 選擇一個圖片文件 (PNG, JPG, JPEG, GIF)
   - 圖片將被用作背景（帶模糊效果）和縮略圖

3. **設置視頻參數**
   - **寬度和高度**: 自定義視頻尺寸，或使用預設按鈕 (1080p, 720p, 480p, 4K)
   - **輸出格式**: 選擇 MP4, WebM, AVI 或 MOV

4. **創建視頻**
   - 點擊 "創建視頻" 按鈕
   - 等待處理完成（處理時間取決於音頻長度和視頻尺寸）
   - 視頻將自動下載

## 系統要求

- Python 3.8+
- FFmpeg (用於視頻編碼)
- 至少 2GB RAM (建議 4GB+)
- 足夠的磁盤空間用於臨時文件

## 注意事項

- 處理大型音頻文件或高分辨率視頻可能需要較長時間
- 最大文件大小限制為 100MB
- 視頻尺寸建議在 320x240 到 7680x4320 之間
- 生成的視頻文件會自動清理

## 許可證

查看 LICENSE 文件了解詳細信息。
