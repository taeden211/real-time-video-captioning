# Project Overview
이 프로젝트는 이미지 캡셔닝 + 객체 관계 분석 + 유사도 검색 시스템이다.

# Pipeline
1. GPT / Gemini API로 이미지 캡셔닝 생성
2. JSON 형태로 저장 (객체, 관계 포함)
3. 해당 JSON을 유사도에 따라 클러스터링 진행
4. 각 클러스터의 대표 템플릿을 생성 후 DB에 저장
4. 실시간 객체 탐지
5. DB에 저장되어있는 템플릿과 유사도 비교

# Tech Stack
- Python
- OpenAI API / Gemini
- Vector DB
- Cosine Similarity

# Goal
- 실시간 입력 이미지와 가장 유사한 기존 데이터 검색

# Constraints
- 데이터는 대량이 아님 (약 5000개)
- 실시간 처리 필요