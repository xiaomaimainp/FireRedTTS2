# 测试脚本内容示例
import requests
import pytest
import json
import os
from pathlib import Path

# 包含语音克隆和随机音色测试用例

BASE_URL = 'http://localhost:8000'

# 测试用例实现

def test_health_check():
    """测试API健康检查"""
    response = requests.get(f"{BASE_URL}/")
    assert response.status_code == 200
    assert "message" in response.json()

def test_voice_cloning_synthesis():
    """测试语音克隆合成功能"""
    # 语音克隆模式 (voice_mode=0)
    test_data = {
        "voice_mode": 0,
        "spk1_audio": "examples/chat_prompt/en/S1.flac",
        "spk1_text": "[S1] Hello, this is speaker one.",
        "spk2_audio": "examples/chat_prompt/en/S2.flac", 
        "spk2_text": "[S2] Hi there, I'm speaker two.",
        "dialogue_text": "[S1] How are you today? [S2] I'm doing great, thanks for asking!"
    }
    
    response = requests.post(f"{BASE_URL}/synthesize", json=test_data)
    
    # 检查响应状态
    assert response.status_code == 201, f"Expected 201, got {response.status_code}: {response.text}"
    
    # 检查响应结构
    result = response.json()
    assert "audio" in result
    assert "sample_rate" in result
    assert result["sample_rate"] == 24000
    assert isinstance(result["audio"], list)
    assert len(result["audio"]) > 0

def test_random_voice_synthesis():
    """测试随机音色合成功能"""
    # 随机音色模式 (voice_mode=1)
    test_data = {
        "voice_mode": 1,
        "spk1_audio": None,
        "spk1_text": None,
        "spk2_audio": None,
        "spk2_text": None,
        "dialogue_text": "[S1] This is a test with random voices. [S2] Yes, it should work without prompts."
    }
    
    response = requests.post(f"{BASE_URL}/synthesize", json=test_data)
    
    # 检查响应状态
    assert response.status_code == 201, f"Expected 201, got {response.status_code}: {response.text}"
    
    # 检查响应结构
    result = response.json()
    assert "audio" in result
    assert "sample_rate" in result
    assert result["sample_rate"] == 24000
    assert isinstance(result["audio"], list)
    assert len(result["audio"]) > 0

def test_invalid_voice_mode():
    """测试无效的语音模式"""
    test_data = {
        "voice_mode": 999,  # 无效模式
        "dialogue_text": "[S1] Test invalid mode."
    }
    
    response = requests.post(f"{BASE_URL}/synthesize", json=test_data)
    
    # 应该返回错误
    assert response.status_code == 400

def test_missing_required_fields_voice_cloning():
    """测试语音克隆模式下缺少必填字段"""
    test_data = {
        "voice_mode": 0,
        "spk1_audio": "examples/chat_prompt/en/S1.flac",
        # 缺少 spk1_text, spk2_audio, spk2_text
        "dialogue_text": "[S1] Test missing fields."
    }
    
    response = requests.post(f"{BASE_URL}/synthesize", json=test_data)
    
    # 应该返回错误
    assert response.status_code == 400

def test_invalid_speaker_text_format():
    """测试无效的说话人文本格式"""
    test_data = {
        "voice_mode": 0,
        "spk1_audio": "examples/chat_prompt/en/S1.flac",
        "spk1_text": "Invalid format without [S1] prefix",  # 无效格式
        "spk2_audio": "examples/chat_prompt/en/S2.flac",
        "spk2_text": "[S2] Valid format.",
        "dialogue_text": "[S1] Test invalid format."
    }
    
    response = requests.post(f"{BASE_URL}/synthesize", json=test_data)
    
    # 应该返回错误
    assert response.status_code == 400

def test_invalid_dialogue_format():
    """测试无效的对话文本格式"""
    test_data = {
        "voice_mode": 0,
        "spk1_audio": "examples/chat_prompt/en/S1.flac",
        "spk1_text": "[S1] Valid prompt.",
        "spk2_audio": "examples/chat_prompt/en/S2.flac",
        "spk2_text": "[S2] Valid prompt.",
        "dialogue_text": "Invalid dialogue format without speaker tags"  # 无效格式
    }
    
    response = requests.post(f"{BASE_URL}/synthesize", json=test_data)
    
    # 应该返回错误
    assert response.status_code == 400

if __name__ == "__main__":
    # 运行所有测试
    pytest.main([__file__, "-v"])