"""
Multimodal helpers for image encoding and content building.
"""

import os
import base64
import mimetypes
from typing import List, Dict, Any


def encode_image(image_path: str) -> str:
    """
    Encode an image file to base64 string.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Base64-encoded string of the image
        
    Raises:
        FileNotFoundError: If the image file doesn't exist
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def build_multimodal_content(prompt: str, image_paths: List[str]) -> List[Dict[str, Any]]:
    """
    Build a multimodal content array with text and images.
    
    Args:
        prompt: Text prompt to include
        image_paths: List of paths to images
        
    Returns:
        List of content dictionaries suitable for OpenAI API
    """
    # Build content array with text first, then images
    content = [
        {
            "type": "input_text",
            "text": prompt
        }
    ]
    
    # Add each image
    for image_path in image_paths:
        base64_image = encode_image(image_path)
        content.append({
            "type": "input_image",
            "image_url": f"data:image/png;base64,{base64_image}"
        })
    
    return content


def build_openrouter_multimodal_content(prompt: str, image_paths: List[str]) -> List[Dict[str, Any]]:
    """
    Build a multimodal content array for OpenRouter/OpenAI chat completions.

    OpenRouter expects text parts to use ``type: text`` and local images to use
    ``type: image_url`` with a data URL.
    """
    content = [
        {
            "type": "text",
            "text": prompt,
        }
    ]

    for image_path in image_paths:
        mime_type, _ = mimetypes.guess_type(image_path)
        if mime_type not in {"image/png", "image/jpeg", "image/webp", "image/gif"}:
            mime_type = "image/png"
        base64_image = encode_image(image_path)
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:{mime_type};base64,{base64_image}",
            },
        })

    return content
