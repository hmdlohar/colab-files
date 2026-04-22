.PHONY: dev dev-web install-web install-audio install-video install-all

dev:
	uvicorn app.main:app --reload

install-web:
	pip install -r requirements-web.txt

install-audio:
	pip install -r requirements-audio.txt

install-video:
	pip install -r requirements-video.txt

install-all:
	pip install -r requirements.txt

