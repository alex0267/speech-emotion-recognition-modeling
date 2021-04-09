import subprocess
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import filetype
from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.inspection import inspect
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import func

from config import DevConfig
from gc_utils import download_blob

EMOTION_MAP = {
    "Joie": "joy",
    "Colère": "anger",
    "Peur": "fear",
    "Tristesse": "sad",
    "Neutre": "neutral",
    "Dégoût": "disgust",
}

Base = declarative_base()


class Record(Base):
    __tablename__ = "record"
    __table_args__ = {"extend_existing": True}
    id = Column(Integer, primary_key=True, index=True)
    record_url = Column(String, unique=False)
    emotion = Column(String, unique=False)
    timestamp = Column(DateTime, unique=False, default=func.now())
    uuid = Column(String, unique=True)
    sentence_id = Column(Integer, ForeignKey("sentence.id"))


def format_data(data):
    records = []
    for obj in data:
        instance = inspect(obj)
        items = instance.attrs.items()
        records.append([x.value for _, x in items])
    return records


def guess_filetype(filename):
    return filetype.guess(filename).extension


def convert(inputfile, outputfile):
    command = [
        "ffmpeg",
        "-i",
        inputfile,
        "-c:a",
        "pcm_f32le",
        "-hide_banner",
        "-loglevel",
        "error",
        outputfile,
    ]
    subprocess.run(command, stdout=subprocess.PIPE, stdin=subprocess.PIPE)


def main():
    engine = create_engine(DevConfig.db_url_string())
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = SessionLocal()
    records = format_data(
        db.query(Record).filter(Record.timestamp >= datetime(2021, 3, 1)).all()
    )

    # Create subfolders per emotion if they do not exist
    root_final_output = Path("../data/raw_data/french")
    for emotion in EMOTION_MAP.values():
        Path.mkdir(root_final_output.joinpath(emotion), exist_ok=True)
    with tempfile.TemporaryDirectory() as tmpdirname:
        print("Created temporary directory: ", tmpdirname)
        for r in records[:200]:
            temp_file = Path(tmpdirname).joinpath("_".join([EMOTION_MAP[r[2]], r[-2]]))
            final_file = root_final_output.joinpath(
                EMOTION_MAP[r[2]],
                "_".join([EMOTION_MAP[r[2]], str(r[-1]), r[-2] + ".wav"]),
            )
            # Download the blob
            download_blob("swa-dev-bucket", Path(r[1]).name, temp_file)
            # Convert to wav format
            convert(temp_file, final_file)
            # Add to dvc tracking
            subprocess.run(
                ["dvc", "add", final_file],
                stdout=subprocess.PIPE,
                stdin=subprocess.PIPE,
            )


if __name__ == "__main__":
    main()
