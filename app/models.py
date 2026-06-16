from app.database import db
from datetime import datetime

class AnalysisResult(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(256))
    recognized_text = db.Column(db.Text)
    result_json = db.Column(db.JSON)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            'id': self.id,
            'filename': self.filename,
            'recognized_text': self.recognized_text,
            'result_json': self.result_json,
            'created_at': self.created_at.isoformat(),
        }