from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
from typing import Optional, List
import re
from fireredtts2 import FireRedTTS2

app = FastAPI()
model: Optional[FireRedTTS2] = None  # 修复：使用 Optional 类型

@app.get("/")
async def health_check():
    """健康检查端点"""
    return {"message": "FireRedTTS2 API is running", "status": "healthy"}

class SynthesisRequest(BaseModel):
    voice_mode: int = 0
    spk1_audio: Optional[str] = None
    spk1_text: Optional[str] = None
    spk2_audio: Optional[str] = None
    spk2_text: Optional[str] = None
    dialogue_text: str

@app.on_event("startup")
def load_model():
    global model
    model = FireRedTTS2(
        pretrained_dir="./pretrained_models/FireRedTTS2",
        gen_type="dialogue",
        device="cuda"
    )

@app.post("/synthesize", status_code=status.HTTP_201_CREATED)
async def synthesize(request: SynthesisRequest):
    # 输入验证逻辑
    # 修复：添加 voice_mode 验证
    if request.voice_mode not in [0, 1]:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Invalid voice_mode. Must be 0 (voice cloning) or 1 (random voice)")
    
    if request.voice_mode == 0:
        if not all([request.spk1_audio, request.spk1_text, request.spk2_audio, request.spk2_text]):
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "Missing required prompt information")
        
        # 修复：放宽说话人文本验证，允许自定义内容
        if request.spk1_text is not None and request.spk2_text is not None:
            if not validate_speaker_text_flexible(request.spk1_text, "[S1]") or \
               not validate_speaker_text_flexible(request.spk2_text, "[S2]"):
                raise HTTPException(400, "Invalid speaker prompt format")

    # 对话文本处理
    dialogue_segments = parse_dialogue_text(request.dialogue_text)
    if not dialogue_segments:
        raise HTTPException(400, "Invalid dialogue text format")

    # 执行语音合成
    try:
        # 修复：确保 model 不为 None
        if model is None:
            raise HTTPException(500, "Model not loaded")
            
        audio_output = model.generate_dialogue(
            text_list=dialogue_segments,
            prompt_wav_list=[request.spk1_audio, request.spk2_audio] if request.voice_mode == 0 else None,
            prompt_text_list=[request.spk1_text, request.spk2_text] if request.voice_mode == 0 else None,
            temperature=0.9,
            topk=30
        )
        return {"audio": audio_output.squeeze(0).numpy().tolist(), "sample_rate": 24000}
    except Exception as e:
        raise HTTPException(500, f"Synthesis error: {str(e)}")

def validate_speaker_text(text: str, prefix: str) -> bool:
    text = text.strip()
    if not text.startswith(prefix):
        return False
    return len(text[len(prefix):].strip()) > 0

# 新增：更灵活的说话人文本验证
def validate_speaker_text_flexible(text: str, prefix: str) -> bool:
    """
    验证说话人文本格式，支持自定义内容
    例如：[S1] Hello, this is speaker one. 或 [S1] Hello, peter.
    """
    text = text.strip()
    if not text.startswith(prefix):
        return False
    # 检查前缀后是否有内容（允许任意内容）
    content = text[len(prefix):].strip()
    return len(content) > 0

def parse_dialogue_text(text: str) -> List[str]:
    segments = re.findall(r'(\[S[0-9]\][^\[]*)', text)
    return [s.strip() for s in segments if validate_dialogue_segment(s)]

def validate_dialogue_segment(segment: str) -> bool:
    return any(segment.startswith(f"[S{i}]") for i in range(1,5))