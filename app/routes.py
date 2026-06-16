import json

from flask import Blueprint, Response, current_app, jsonify, render_template, request

from app.database import db
from app.models import AnalysisResult
from app.services.image_processor import (
    hash_image,
    ocr_extract_text,
    preprocess_image,
    save_uploaded_file,
)
from app.services.text_analyzer import analyze_text, clean_text

bp = Blueprint("main", __name__)


@bp.route("/api/ping")
def ping():
    return jsonify({
        "status": "ok",
        "message": "Liter backend online",
    })


@bp.route("/api/process", methods=["POST"])
def process_image():
    try:
        if "file" not in request.files:
            return jsonify({"error": "no file provided"}), 400

        file = request.files["file"]
        if not file or file.filename == "":
            return jsonify({"error": "empty file"}), 400

        path = save_uploaded_file(file, current_app.config["UPLOAD_FOLDER"])
        preprocess_image(path)
        file_hash = hash_image(path)

        cached = AnalysisResult.query.filter_by(filename=file_hash).first()
        if cached:
            return json_response({
                "id": cached.id,
                "recognized_text": cached.recognized_text,
            })

        text = ocr_extract_text(path, current_app.config)
        if not text:
            return jsonify({"error": "OCR failed or empty text"}), 500

        cleaned_text = clean_text(text)
        result = AnalysisResult(
            filename=file_hash,
            recognized_text=cleaned_text,
            result_json=None,
        )

        db.session.add(result)
        db.session.commit()

        return json_response({
            "id": result.id,
            "recognized_text": cleaned_text,
        })

    except Exception as e:
        current_app.logger.exception("Image processing error")
        return jsonify({"error": str(e)}), 500


@bp.route("/api/analyze/<int:result_id>", methods=["POST"])
def analyze_existing_result(result_id):
    try:
        result = db.get_or_404(AnalysisResult, result_id)
        data = request.get_json()
        if not data or "text" not in data:
            return jsonify({"error": "text is required"}), 400

        text = clean_text(data["text"])
        analysis_data = analyze_text(text, current_app.config)

        result.recognized_text = text
        result.result_json = analysis_data

        db.session.commit()

        return jsonify({
            "status": "ok",
            "id": result.id,
        }), 200

    except Exception as e:
        current_app.logger.exception("Analysis error")
        return jsonify({"error": str(e)}), 500


@bp.route("/api/results/<int:result_id>", methods=["GET"])
def get_result(result_id):
    result = db.session.get(AnalysisResult, result_id)
    if not result:
        return jsonify({"error": "not found"}), 404
    return jsonify(result.to_dict())


@bp.route("/")
def index():
    return render_template("index_dark.html")


@bp.route("/results/<int:result_id>")
def results_page(result_id):
    result = db.get_or_404(AnalysisResult, result_id)
    return render_template(
        "result.html",
        recognized_text=result.recognized_text,
        result_id=result_id,
    )


@bp.route("/analysis/<int:result_id>")
def analysis_page(result_id):
    result = db.get_or_404(AnalysisResult, result_id)

    if not result.result_json:
        return "Анализ ещё не выполнен", 400

    return render_template(
        "analysis.html",
        analysis_data=result.result_json,
        result_id=result_id,
        original_text=result.recognized_text,
    )


def json_response(payload, status=200):
    return Response(
        json.dumps(payload, ensure_ascii=False),
        mimetype="application/json",
        status=status,
    )
