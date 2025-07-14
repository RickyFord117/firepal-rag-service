from sqlalchemy import Column, Integer, String, Text
from .database import Base


class WebContentChunk(Base):
    __tablename__ = "web_content_chunks"

    id = Column(Integer, primary_key=True, index=True)
    chunk_id = Column(String, unique=True, index=True, nullable=False)
    text_content = Column(Text, nullable=False)
    source_url = Column(String, index=True)
    source_title = Column(String)
    heading = Column(String, nullable=True)
    chunk_type = Column(String)

    def __repr__(self):
        return f"<WebContentChunk(id={self.id}, url='{self.source_url}')>"
