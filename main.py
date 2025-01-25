from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
import yt_dlp
import whisper_timestamped
import moviepy.editor as mp
import cv2
import os
from typing import List
import json
from dotenv import load_dotenv

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class VideoRequest(BaseModel):
    youtube_url: str
    api_key: str
    language: str = "english"

def download_youtube_video(url: str) -> str:
    ydl_opts = {
        'format': 'best',
        'outtmpl': 'downloads/%(id)s.%(ext)s'
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        return f"downloads/{info['id']}.{info['ext']}"

def extract_audio(video_path: str) -> str:
    audio_path = video_path.rsplit('.', 1)[0] + '.wav'
    video = mp.VideoFileClip(video_path)
    video.audio.write_audiofile(audio_path)
    return audio_path

def get_transcription(audio_path: str, language: str) -> dict:
    model = whisper_timestamped.load_model("base")
    result = whisper_timestamped.transcribe(
        model,
        audio_path,
        language=language
    )
    return result

def get_highlights(transcription: dict, api_key: str) -> List[dict]:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-pro')
    
    # Format transcription for Gemini
    text = json.dumps(transcription, indent=2)
    prompt = f"""
    Analyze this transcription and identify the top 5 highlights between 45-59 seconds:
    {text}
    
    Return only a JSON array with objects containing:
    - start_time (in seconds)
    - end_time (in seconds)
    - text (the highlight text)
    """
    
    response = model.generate_text(prompt)
    return json.loads(response.text)

def create_highlight_clips(video_path: str, highlights: List[dict]) -> List[str]:
    output_clips = []
    video = mp.VideoFileClip(video_path)
    
    for idx, highlight in enumerate(highlights):
        start_time = highlight['start_time']
        end_time = highlight['end_time']
        
        # Cut the clip
        clip = video.subclip(start_time, end_time)
        
        # Resize to 9:16 ratio
        w, h = clip.size
        target_w = h * 9 // 16
        x_center = (w - target_w) // 2
        
        clip = clip.crop(x1=x_center, y1=0, x2=x_center+target_w, y2=h)
        
        # Add captions
        txt_clip = mp.TextClip(
            highlight['text'],
            fontsize=24,
            color='white',
            bg_color='black',
            font='Arial'
        )
        txt_clip = txt_clip.set_position(('center', 'bottom')).set_duration(clip.duration)
        
        final_clip = mp.CompositeVideoClip([clip, txt_clip])
        
        # Save the clip
        output_path = f"output/highlight_{idx}.mp4"
        final_clip.write_videofile(output_path)
        output_clips.append(output_path)
    
    return output_clips

@app.post("/process-video")
async def process_video(request: VideoRequest):
    try:
        # Create necessary directories
        os.makedirs("downloads", exist_ok=True)
        os.makedirs("output", exist_ok=True)
        
        # Download video
        video_path = download_youtube_video(request.youtube_url)
        
        # Extract audio
        audio_path = extract_audio(video_path)
        
        # Get transcription
        transcription = get_transcription(audio_path, request.language)
        
        # Get highlights using Gemini
        highlights = get_highlights(transcription, request.api_key)
        
        # Create highlight clips
        output_clips = create_highlight_clips(video_path, highlights)
        
        # Clean up temporary files
        os.remove(video_path)
        os.remove(audio_path)
        
        return {
            "status": "success",
            "highlights": highlights,
            "output_clips": output_clips
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
