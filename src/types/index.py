from typing import List, Dict, Any

class UserProfile:
    def __init__(self, username: str, target_audience: str, uploaded_images: List[str], captions: List[str]):
        self.username = username
        self.target_audience = target_audience
        self.uploaded_images = uploaded_images
        self.captions = captions

class ImageMetadata:
    def __init__(self, image_path: str, description: str, upload_time: str):
        self.image_path = image_path
        self.description = description
        self.upload_time = upload_time

class CaptionFormat:
    def __init__(self, text: str, hashtags: List[str]):
        self.text = text
        self.hashtags = hashtags

class Recommendation:
    def __init__(self, image: ImageMetadata, caption: CaptionFormat):
        self.image = image
        self.caption = caption

def get_default_recommendation() -> Recommendation:
    return Recommendation(
        image=ImageMetadata(image_path="", description="", upload_time=""),
        caption=CaptionFormat(text="", hashtags=[])
    )