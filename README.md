# Litera

A Flask web application for recognizing handwritten text from an image and analyzing writing quality with AI: spelling, punctuation, style, vocabulary, and short improvement recommendations.

## Features

- image upload with handwritten text;
- image preprocessing before OCR;
- text recognition via Yandex Vision;
- editable recognized text before analysis;
- grammar and style analysis via YandexGPT;
- result storage in SQLite;
- analysis page with a downloadable text report.

## Tech Stack

- Python / Flask
- Flask-SQLAlchemy
- SQLite
- OpenCV
- Yandex Vision API
- YandexGPT
- Bootstrap

## Local Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create `.env` from `.env.example` and fill in the required API credentials.
4. Run the application:

```bash
python run.py
```

The application will be available at `http://127.0.0.1:5000`.

## Project Structure

- `app/__init__.py` - Flask application factory;
- `app/config.py` - application configuration;
- `app/routes.py` - web pages and API routes;
- `app/models.py` - database models;
- `app/services/` - OCR, text analysis, and Yandex IAM integration;
- `app/templates/` - HTML templates;
- `app/static/` - CSS and JavaScript assets.

## Notes

The `.env` file, `sa-key.json`, local database, uploaded files, IDE settings, and virtual environment are excluded from git via `.gitignore`.
