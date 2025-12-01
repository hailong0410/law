from pydantic import BaseModel

class ArticleBase(BaseModel):
    so_dieu: int
    chuong: str
    tieu_de: str
    tom_tat: str | None = None

class ArticleDetail(ArticleBase):
    id: int
    noi_dung: str
    tags: str | None = None

    class Config:
        orm_mode = True


class ArticleListItem(BaseModel):
    id: int
    so_dieu: int
    tieu_de: str
    tom_tat: str | None = None

    class Config:
        orm_mode = True
