import streamlit as st
import datetime
import numpy as np
from loguru import logger
from PIL import Image
import io
import torch
import open_clip
import requests

POST_TIMES = {
    "소비재 / 소매 (Consumer goods and retail)": [
        ("월요일", "07:00", "11:00"),
        ("화요일", "17:00", "19:00"),
        ("목요일", "09:00", "09:00"),
        ("금요일", "06:00", "10:00"),
        ("금요일", "17:00", "21:00"),
    ],
    "식음료 / 호텔 / 관광 (Dining, hospitality, and tourism)": [
        ("수요일", "11:00", "15:00"),
    ],
    "금융 서비스 (Financial services)": [
        ("수요일", "11:00", "14:00"),
        ("금요일", "11:00", "16:00"),
    ],
    "미디어 / 엔터테인먼트 (Media and entertainment)": [
        ("일요일", "13:00", "13:00"),
        ("화요일", "07:00", "08:00"),
        ("수요일", "17:00", "17:00"),
        ("금요일", "14:00", "16:00"),
    ],
    "헬스케어 / 바이오 / 제약 (Healthcare, pharma, and biotech)": [
        ("월요일", "13:00", "15:00"),
        ("목요일", "09:00", "12:00"),
        ("토요일", "08:00", "11:00"),
    ],
    "마케팅 / 광고 (Marketing agencies)": [
        ("월요일", "09:00", "11:00"),
        ("금요일", "08:00", "10:00"),
    ],
    "비영리 (Nonprofit)": [
        ("수요일", "10:00", "13:00"),
        ("목요일", "14:00", "17:00"),
    ],
    "건설 / 제조 / 광업 (Construction, mining, and manufacturing)": [
        ("화요일", "15:00", "18:00"),
        ("수요일", "14:00", "17:00"),
        ("목요일", "16:00", "18:00"),
    ],
    "에너지 / 공공서비스 (Utilities and energy)": [
        ("화요일", "14:00", "17:00"),
        ("목요일", "13:00", "15:00"),
    ],
    "정부 (Government)": [
        ("목요일", "12:00", "15:00"),
    ],
}

GENERIC_POST_TIMES = [
    ("월요일", "15:00", "21:00"),
    ("화요일", "05:00", "08:00"),
    ("화요일", "15:00", "19:00"),
    ("수요일", "17:00", "17:00"),
    ("목요일", "16:00", "17:00"),
    ("금요일", "16:00", "16:00"),
    ("토요일", "11:00", "11:00"),
    ("토요일", "17:00", "17:00"),
    ("일요일", "12:00", "15:00"),
]

CREATOR_TO_FIELD = {
    "뷰티": "소비재 / 소매 (Consumer goods and retail)",
    "패션": "소비재 / 소매 (Consumer goods and retail)",
    "식품": "소비재 / 소매 (Consumer goods and retail)",
    "생활용품": "소비재 / 소매 (Consumer goods and retail)",
    "홈쇼핑": "소비재 / 소매 (Consumer goods and retail)",
    "쇼핑몰": "소비재 / 소매 (Consumer goods and retail)",
    "제품 협찬": "소비재 / 소매 (Consumer goods and retail)",
    "여행": "식음료 / 호텔 / 관광 (Dining, hospitality, and tourism)",
    "호텔": "식음료 / 호텔 / 관광 (Dining, hospitality, and tourism)",
    "레스토랑": "식음료 / 호텔 / 관광 (Dining, hospitality, and tourism)",
    "투자": "금융 서비스 (Financial services)",
    "재테크": "금융 서비스 (Financial services)",
    "경제": "금융 서비스 (Financial services)",
    "금융": "금융 서비스 (Financial services)",
    "Q&A": "금융 서비스 (Financial services)",
    "연예": "미디어 / 엔터테인먼트 (Media and entertainment)",
    "밴드": "미디어 / 엔터테인먼트 (Media and entertainment)",
    "밈": "미디어 / 엔터테인먼트 (Media and entertainment)",
    "유머": "미디어 / 엔터테인먼트 (Media and entertainment)",
    "게임": "미디어 / 엔터테인먼트 (Media and entertainment)",
    "웹툰": "미디어 / 엔터테인먼트 (Media and entertainment)",
    "건강": "헬스케어 / 바이오 / 제약 (Healthcare, pharma, and biotech)",
    "운동": "헬스케어 / 바이오 / 제약 (Healthcare, pharma, and biotech)",
    "피트니스": "헬스케어 / 바이오 / 제약 (Healthcare, pharma, and biotech)",
    "영양": "헬스케어 / 바이오 / 제약 (Healthcare, pharma, and biotech)",
    "헬스케어": "헬스케어 / 바이오 / 제약 (Healthcare, pharma, and biotech)",
    "디지털 마케팅": "마케팅 / 광고 (Marketing agencies)",
    "마케팅": "마케팅 / 광고 (Marketing agencies)",
    "툴": "마케팅 / 광고 (Marketing agencies)",
    "노하우": "마케팅 / 광고 (Marketing agencies)",
    "특수 직군": "건설 / 제조 / 광업 (Construction, mining, and manufacturing)",
    "산업기술": "건설 / 제조 / 광업 (Construction, mining, and manufacturing)",
    "건설": "건설 / 제조 / 광업 (Construction, mining, and manufacturing)",
    "제조": "건설 / 제조 / 광업 (Construction, mining, and manufacturing)",
    "광업": "건설 / 제조 / 광업 (Construction, mining, and manufacturing)",
    "친환경": "에너지 / 공공서비스 (Utilities and energy)",
    "에너지": "에너지 / 공공서비스 (Utilities and energy)",
    "공공서비스": "에너지 / 공공서비스 (Utilities and energy)",
    "공공 정책": "정부 (Government)",
    "법률": "정부 (Government)",
    "공공 데이터": "정부 (Government)",
}

def load_test_images():
    """테스트용 이미지들을 로드하는 함수"""
    try:
        # 데이터 디렉토리 경로
        data_dir = Path("data")
        user_images_dir = data_dir / "user_images"
        candidate_images_dir = data_dir / "candidate_images"
        
        # 지원하는 이미지 확장자
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        
        # 사용자 이미지들 로드
        user_images = []
        if user_images_dir.exists():
            for file_path in user_images_dir.iterdir():
                if file_path.suffix.lower() in image_extensions:
                    try:
                        with open(file_path, 'rb') as f:
                            image_bytes = f.read()
                            # StreamlitUploadedFile과 유사한 객체 생성
                            class MockUploadedFile:
                                def __init__(self, name, data):
                                    self.name = name
                                    self._data = data
                                    self._position = 0
                                
                                def read(self):
                                    return self._data
                                
                                def seek(self, position):
                                    self._position = position
                            
                            user_images.append(MockUploadedFile(file_path.name, image_bytes))
                        logger.info(f"사용자 이미지 로드: {file_path.name}")
                    except Exception as e:
                        logger.error(f"사용자 이미지 로드 실패 {file_path.name}: {e}")
        
        # 후보 이미지들 로드
        candidate_images = []
        if candidate_images_dir.exists():
            for file_path in candidate_images_dir.iterdir():
                if file_path.suffix.lower() in image_extensions:
                    try:
                        with open(file_path, 'rb') as f:
                            image_bytes = f.read()
                            candidate_images.append(MockUploadedFile(file_path.name, image_bytes))
                        logger.info(f"후보 이미지 로드: {file_path.name}")
                    except Exception as e:
                        logger.error(f"후보 이미지 로드 실패 {file_path.name}: {e}")
        
        logger.info(f"테스트 이미지 로드 완료: 사용자 {len(user_images)}장, 후보 {len(candidate_images)}장")
        return user_images, candidate_images
        
    except Exception as e:
        logger.error(f"테스트 이미지 로드 중 오류: {e}")
        return [], []

def map_creator_to_field(user_input: str) -> str:
    for keyword, field in CREATOR_TO_FIELD.items():
        if keyword in user_input:
            return field
    return None

def get_next_best_time(audience: str, now: datetime.datetime = None) -> str:
    if now is None:
        now = datetime.datetime.now()
    today_idx = now.weekday()  # 월:0, 일:6
    days = ["월요일", "화요일", "수요일", "목요일", "금요일", "토요일", "일요일"]

    if not audience or audience not in POST_TIMES:
        time_table = GENERIC_POST_TIMES
        info = "분야 미선택/모름/해당 없음 - 보편적 추천"
    else:
        time_table = POST_TIMES[audience]
        info = f"{audience} 분야"

    candidates = []
    for day, start, end in time_table:
        day_idx = days.index(day)
        delta_days = (day_idx - today_idx) % 7
        post_date = now + datetime.timedelta(days=delta_days)
        start_dt = datetime.datetime.combine(post_date.date(), datetime.time.fromisoformat(start))
        if start_dt < now:
            start_dt += datetime.timedelta(days=7)
        if start==end:
            candidates.append((start_dt, f"{day} {start}"))
        else:
            candidates.append((start_dt, f"{day} {start}~{end}"))
    if not candidates:
        return "추천 업로드 시간이 없습니다."
    candidates.sort()
    return f"{info}의 가장 가까운 추천 업로드 시간: {candidates[0][1]}"

def load_clip_model():
    """open_clip 기반 CLIP 모델 로드"""
    try:
        logger.info("open_clip 모델 로딩 중...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
        model = model.to(device)
        logger.info(f"open_clip 모델 로딩 완료 (디바이스: {device})")
        return model, preprocess, device
    except Exception as e:
        logger.error(f"open_clip 모델 로딩 실패: {e}")
        return None, None, None

def calculate_clip_similarity(img1_bytes, img2_bytes, model, preprocess, device):
    """open_clip을 사용한 이미지 유사도 계산"""
    try:
        img1 = Image.open(io.BytesIO(img1_bytes)).convert('RGB')
        img2 = Image.open(io.BytesIO(img2_bytes)).convert('RGB')
        img1_tensor = preprocess(img1).unsqueeze(0).to(device)
        img2_tensor = preprocess(img2).unsqueeze(0).to(device)
        with torch.no_grad():
            img1_features = model.encode_image(img1_tensor)
            img2_features = model.encode_image(img2_tensor)
            img1_features = img1_features / img1_features.norm(dim=-1, keepdim=True)
            img2_features = img2_features / img2_features.norm(dim=-1, keepdim=True)
            similarity = (img1_features @ img2_features.T).item()
        return similarity
    except Exception as e:
        logger.error(f"open_clip 유사도 계산 실패: {e}")
        return 0.0

_clip_model = None
_clip_preprocess = None
_clip_device = None

def get_clip_model():
    global _clip_model, _clip_preprocess, _clip_device
    if _clip_model is None:
        _clip_model, _clip_preprocess, _clip_device = load_clip_model()
    return _clip_model, _clip_preprocess, _clip_device

def calculate_image_similarity(img1_bytes, img2_bytes):
    model, preprocess, device = get_clip_model()
    if model is not None:
        return calculate_clip_similarity(img1_bytes, img2_bytes, model, preprocess, device)
    else:
        logger.warning("open_clip 모델 로딩 실패")
        return 0.0

def find_most_similar_image(user_images, candidate_images):
    logger.info("clip 방법으로 이미지 유사도 분석 시작")
    best_idx = 0
    best_similarity = -1
    for i, candidate_img in enumerate(candidate_images):
        candidate_bytes = candidate_img.read()
        candidate_img.seek(0)
        similarities = []
        for user_img in user_images:
            user_bytes = user_img.read()
            user_img.seek(0)
            similarity = calculate_image_similarity(user_bytes, candidate_bytes)
            similarities.append(similarity)
        avg_similarity = np.mean(similarities)
        logger.info(f"후보 이미지 {i+1} 평균 유사도: {avg_similarity:.4f}")
        if avg_similarity > best_similarity:
            best_similarity = avg_similarity
            best_idx = i
    logger.info(f"가장 유사한 이미지 인덱스: {best_idx}, 유사도: {best_similarity:.4f}")
    return best_idx

def generate_caption_with_llm(captions, image_desc="사진"):
    API_URL = "https://api-inference.huggingface.co/models/google/gemma-2b-it"
    headers = {}
    prompt = (
        "아래는 인스타그램 사진 캡션 예시입니다:\n"
        + "\n".join(captions[:5])
        + f"\n\n위 스타일을 참고해서, '{image_desc}'에 어울리는 짧은 코멘트와 해시태그 2개를 추천해줘."
    )
    payload = {"inputs": prompt, "parameters": {"max_new_tokens": 60}}
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
        result = response.json()
        if isinstance(result, list) and len(result) > 0 and "generated_text" in result[0]:
            return result[0]["generated_text"]
        elif "generated_text" in result:
            return result["generated_text"]
        elif "error" in result:
            return f"LLM 오류: {result['error']}"
        else:
            return str(result)
    except Exception as e:
        return f"캡션 생성 중 오류: {e}"

def main():
    st.title("PicPilot Agent")
    st.markdown(
        """
        안녕하세요. 당신의 인스타그램 업로드용 사진을 골라드릴 Picpilot 입니다.

        <span style="font-size:1.00em; font-weight:300;">
        Picpilot은<br>
        ✅ 당신의 선호 이미지 및 캡션 스타일을 분석하고<br>
        ✅ 여러 후보 사진 중 가장 적합한 이미지와 그에 어울리는 캡션을 추천 하며<br>
        ✅ 추천 업로드 요일/시간까지 제안하는 AI 기반 인스타그램 업로드 보조 서비스입니다.
        </span>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<div style="font-size:1.25em; font-weight:600; margin-top:1.5em;">🔹 본인이 주로 활동하거나, 타겟으로 하는 분야</div>', unsafe_allow_html=True)
    user_field_input = st.text_input("예시: 뷰티 리뷰, 여행, 투자 콘텐츠 크리에이터 등 / 분야 특정을 원치 않거나 모호한 경우 '없음' 으로 표기 가능")

    audience = None
    if user_field_input:
        audience = map_creator_to_field(user_field_input)
        if audience:
            st.success(f"입력하신 내용이 '{audience}' 분야로 인식되었습니다.")
        else:
            st.warning("입력하신 내용이 기존 분야 리스트와 매칭되지 않아, 보편적 추천이 적용됩니다.")

    st.markdown('<div style="font-size:1.25em; font-weight:600; margin-top:1.5em;">🔹 본인의 스타일을 보여주거나, 선호하는 인스타그램 사진 5-10장</div>', unsafe_allow_html=True)
    user_images = st.file_uploader("사진 업로드 (최대 10장)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    st.markdown('<div style="font-size:1.25em; font-weight:600; margin-top:1.5em;">🔹 기존에 올렸던 이미지 캡션과 태그 예시 5개 이상 (한 줄에 하나씩)</div>', unsafe_allow_html=True)
    captions = st.text_area("예시: 너무 행복했던 일본 여행!💗 #여행스타그램 #OOTD 등").splitlines()

    st.markdown('<div style="font-size:1.25em; font-weight:600; margin-top:1.5em;">🔹 업로드를 희망하는 후보 사진 2-10장</div>', unsafe_allow_html=True)
    candidate_images = st.file_uploader("사진 업로드 (최대 10장)", type=["jpg", "jpeg", "png"], accept_multiple_files=True, key="candidate")

    # 분석 방법 선택 드롭다운 (보여주기용, 실제로는 clip만 동작)
    analysis_method = st.selectbox(
        "이미지 유사도 분석 방법을 선택하세요",
        [
            "clip (이미지 스타일/의미 기반)",
            "histogram (이미지 색상 기반)",
            "orb (이미지 구조 기반, 특징 매칭)"
        ],
        index=0,
        help="기획서에 명시된 다양한 분석 방법을 보여주기 위한 옵션입니다. 현재 버전 오류로 CLIP 기반으로만 동작합니다."
    )
    # st.info("※ 현재 Streamlit Cloud에서는 CLIP 기반 추천만 실제로 동작합니다. (색상/구조 기반은 시연용 UI)")

    if st.button("업로드 이미지 추천"):
        if user_images and candidate_images:
            if len(candidate_images) < 2:
                st.warning("후보 사진을 2장 이상 업로드해주세요.")
            else:
                with st.spinner("이미지 유사도 분석 중..."):
                    try:
                        logger.info("이미지 유사도 분석 시작")
                        # 실제 분석은 clip만 사용
                        best_idx = find_most_similar_image(user_images, candidate_images)
                        logger.info("이미지 유사도 분석 완료")
                        best_image = candidate_images[best_idx]
                        st.image(best_image, caption="가장 유사한 스타일의 추천 이미지")
                        if captions:
                            with st.spinner("추천 캡션 생성 중..."):
                                gen_caption = generate_caption_with_llm(captions, image_desc="추천 이미지")
                            st.markdown("**추천 코멘트 및 해시태그:**")
                            st.write(gen_caption)
                        best_time = get_next_best_time(audience)
                        st.write(best_time)
                    except Exception as e:
                        logger.error(f"이미지 분석 실패: {e}")
                        st.error(f"이미지 분석 중 오류가 발생했습니다: {str(e)}")
                        st.info("잠시 후 다시 시도해주세요.")
        else:
            st.warning("기존 사진과 후보 사진을 모두 업로드 해주세요.")

if __name__ == "__main__":
    main()