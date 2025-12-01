from sqlalchemy import Column, Integer, String, Text
from app.database import Base 


class Article(Base):
    __tablename__ = "articles"

    id = Column(Integer, primary_key=True, index=True)
    so_dieu = Column(Integer, index=True)  
    chuong = Column(String(255))
    tieu_de = Column(String(255))
    noi_dung = Column(Text)
    tom_tat = Column(Text, nullable=True)
    tags = Column(String(255), nullable=True)
