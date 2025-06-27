import streamlit as st
import datetime
import numpy as np
from loguru import logger
import cv2
from PIL import Image
import io

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

# 크리에이터 키워드와 분야 매핑
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
        candidates.append((start_dt, f"{day} {start}~{end}"))
    if not candidates:
        return "추천 업로드 시간이 없습니다."
    candidates.sort()
    return f"{info}의 가장 가까운 추천 업로드 시간: {candidates[0][1]}"

def calculate_histogram_similarity(img1_bytes, img2_bytes):
    """히스토그램 기반 이미지 유사도 계산"""
    try:
        # 바이트를 numpy 배열로 변환
        img1_array = np.frombuffer(img1_bytes, np.uint8)
        img2_array = np.frombuffer(img2_bytes, np.uint8)
        
        # OpenCV로 이미지 읽기
        img1 = cv2.imdecode(img1_array, cv2.IMREAD_COLOR)
        img2 = cv2.imdecode(img2_array, cv2.IMREAD_COLOR)
        
        # 이미지 크기 통일 (메모리 효율성)
        img1 = cv2.resize(img1, (256, 256))
        img2 = cv2.resize(img2, (256, 256))
        
        # 히스토그램 계산
        hist1 = cv2.calcHist([img1], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist2 = cv2.calcHist([img2], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        
        # 히스토그램 정규화
        hist1 = cv2.normalize(hist1, hist1).flatten()
        hist2 = cv2.normalize(hist2, hist2).flatten()
        
        # 코사인 유사도 계산
        similarity = np.dot(hist1, hist2) / (np.linalg.norm(hist1) * np.linalg.norm(hist2))
        
        return similarity
    except Exception as e:
        logger.error(f"히스토그램 유사도 계산 실패: {e}")
        return 0.0

def calculate_orb_similarity(img1_bytes, img2_bytes):
    """ORB 특징점 기반 이미지 유사도 계산"""
    try:
        # 바이트를 numpy 배열로 변환
        img1_array = np.frombuffer(img1_bytes, np.uint8)
        img2_array = np.frombuffer(img2_bytes, np.uint8)
        
        # OpenCV로 이미지 읽기
        img1 = cv2.imdecode(img1_array, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imdecode(img2_array, cv2.IMREAD_GRAYSCALE)
        
        # 이미지 크기 통일
        img1 = cv2.resize(img1, (512, 512))
        img2 = cv2.resize(img2, (512, 512))
        
        # ORB 특징점 검출기 생성
        orb = cv2.ORB_create()
        
        # 특징점과 디스크립터 검출
        kp1, des1 = orb.detectAndCompute(img1, None)
        kp2, des2 = orb.detectAndCompute(img2, None)
        
        if des1 is None or des2 is None:
            return 0.0
        
        # 특징점 매칭
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        
        # 유사도 계산 (매칭된 특징점 수 기반)
        similarity = len(matches) / max(len(kp1), len(kp2)) if max(len(kp1), len(kp2)) > 0 else 0.0
        
        return similarity
    except Exception as e:
        logger.error(f"ORB 유사도 계산 실패: {e}")
        return 0.0

def calculate_image_similarity(img1_bytes, img2_bytes, method="histogram"):
    """이미지 유사도 계산 (통합 함수)"""
    if method == "histogram":
        return calculate_histogram_similarity(img1_bytes, img2_bytes)
    elif method == "orb":
        return calculate_orb_similarity(img1_bytes, img2_bytes)
    else:
        logger.warning(f"알 수 없는 방법: {method}, 히스토그램 방법 사용")
        return calculate_histogram_similarity(img1_bytes, img2_bytes)

def find_most_similar_image(user_images, candidate_images, method="histogram"):
    """가장 유사한 이미지 찾기"""
    logger.info(f"{method} 방법으로 이미지 유사도 분석 시작")
    
    best_idx = 0
    best_similarity = -1
    
    for i, candidate_img in enumerate(candidate_images):
        candidate_bytes = candidate_img.read()
        candidate_img.seek(0)  # 파일 포인터 리셋
        
        # 사용자 이미지들과의 평균 유사도 계산
        similarities = []
        for user_img in user_images:
            user_bytes = user_img.read()
            user_img.seek(0)  # 파일 포인터 리셋
            
            similarity = calculate_image_similarity(user_bytes, candidate_bytes, method)
            similarities.append(similarity)
        
        avg_similarity = np.mean(similarities)
        logger.info(f"후보 이미지 {i+1} 평균 유사도: {avg_similarity:.4f}")
        
        if avg_similarity > best_similarity:
            best_similarity = avg_similarity
            best_idx = i
    
    logger.info(f"가장 유사한 이미지 인덱스: {best_idx}, 유사도: {best_similarity:.4f}")
    return best_idx

def main():
    st.title("PicPilot Agent")

    # 맨 처음 인사
    st.markdown("안녕하세요 :) 당신의 인스타그램 업로드용 사진을 골라드릴 Picpilot 입니다.")

    st.markdown('<div style="font-size:1.25em; font-weight:600; margin-top:1.5em;">본인이 주로 활동하거나, 타겟으로 하는 분야</div>', unsafe_allow_html=True)
    user_field_input = st.text_input("예시: 뷰티 리뷰, 여행 블로거, 투자 콘텐츠 creator 등")

    audience = None
    if user_field_input:
        audience = map_creator_to_field(user_field_input)
        if audience:
            st.success(f"입력하신 내용이 '{audience}' 분야로 인식되었습니다.")
        else:
            st.warning("입력하신 내용이 기존 분야 리스트와 매칭되지 않아, 보편적 추천이 적용됩니다.")

    st.markdown('<div style="font-size:1.25em; font-weight:600; margin-top:1.5em;">본인 스타일을 보여주는 인스타그램 사진 5-10장</div>', unsafe_allow_html=True)
    user_images = st.file_uploader("기존 인스타그램 사진 업로드 (최대 10장)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    st.markdown('<div style="font-size:1.25em; font-weight:600; margin-top:1.5em;">기존에 올렸던 사진 캡션과 태그 예시 5개 이상 (한 줄에 하나씩)</div>', unsafe_allow_html=True)
    captions = st.text_area("예시: 너무 행복했던 일본 여행!💗 #여행스타그램 #OOTD 등").splitlines()

    st.markdown('<div style="font-size:1.25em; font-weight:600; margin-top:1.5em;">다음 업로드를 희망하는 후보 사진들</div>', unsafe_allow_html=True)
    candidate_images = st.file_uploader("후보 사진 업로드 (2장 이상)", type=["jpg", "jpeg", "png"], accept_multiple_files=True, key="candidate")

    # 이미지 유사도 분석 방법 선택
    st.markdown('<div style="font-size:1.25em; font-weight:600; margin-top:1.5em;">이미지 유사도 분석 방법</div>', unsafe_allow_html=True)
    analysis_method = st.selectbox(
        "분석 방법을 선택하세요",
        ["histogram", "orb"],
        format_func=lambda x: {
            "histogram": "히스토그램 비교 (빠름, 색상 기반)",
            "orb": "ORB 특징점 비교 (정확함, 구조 기반)"
        }[x]
    )

    if st.button("업로드 이미지 추천"):
        if user_images and candidate_images:
            if len(candidate_images) < 2:
                st.warning("후보 사진을 2장 이상 업로드해주세요.")
            else:
                with st.spinner("이미지 유사도 분석 중..."):
                    try:
                        logger.info("이미지 유사도 분석 시작")
                        best_idx = find_most_similar_image(user_images, candidate_images, analysis_method)
                        logger.info("이미지 유사도 분석 완료")
                        best_image = candidate_images[best_idx]
                        st.image(best_image, caption="가장 유사한 스타일의 추천 이미지")
                        if captions:
                            st.write("추천 캡션:", captions[0])
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