import streamlit as st
import datetime
import requests
import numpy as np

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

# DINOv2 Embedding Space API 엔드포인트로 변경
API_URL = "https://havepaws-dinov2-embedding.hf.space/run/predict"

def get_image_embedding(image_bytes, api_url):
    response = requests.post(
        api_url,
        files={"image": image_bytes}
    )
    response.raise_for_status()
    # DINOv2 Embedding Space는 {"data": [[...]]} 형태로 반환
    return np.array(response.json()["data"][0])

def find_most_similar_image(user_images, candidate_images, api_url):
    print("이미지 1단계 임베딩 시작")
    user_embeds = [get_image_embedding(img.read(), api_url) for img in user_images]
    print("이미지 2단계 임베딩 시작")
    candidate_embeds = [get_image_embedding(img.read(), api_url) for img in candidate_images]
    user_mean = np.mean(user_embeds, axis=0)
    sims = [np.dot(user_mean, c) / (np.linalg.norm(user_mean) * np.linalg.norm(c)) for c in candidate_embeds]
    best_idx = int(np.argmax(sims))
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

    if st.button("업로드 이미지 추천"):
        if user_images and candidate_images:
            with st.spinner("이미지 유사도 분석 중..."):
                print("이미지 유사도 분석 시작")
                best_idx = find_most_similar_image(user_images, candidate_images, API_URL)
                print("이미지 유사도 분석 완료")
                best_image = candidate_images[best_idx]
                st.image(best_image, caption="가장 유사한 스타일의 추천 이미지")
                if captions:
                    st.write("추천 캡션:", captions[0])
                best_time = get_next_best_time(audience)
                st.write(best_time)
        else:
            st.warning("기존 사진과 후보 사진을 모두 업로드 해주세요.")

if __name__ == "__main__":
    main()