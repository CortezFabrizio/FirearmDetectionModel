import os 

from sqlalchemy import Boolean,Float  , create_engine
from sqlalchemy.orm import DeclarativeBase , Mapped , mapped_column  , sessionmaker


db_username = os.getenv('DB_USERNAME')
db_password = os.getenv('DB_PASSWORD')
db_endpoint = os.getenv('DB_ENDPOINT')

engine = create_engine(f'postgresql://{db_username}:{db_password}@{db_endpoint}')

Session = sessionmaker(bind=engine)

class Base(DeclarativeBase):
    pass


class Predictions(Base):
    __tablename__ = "images_results"

    key: Mapped[int] = mapped_column(primary_key=True)
    boolean_presence: Mapped[bool] = mapped_column(Boolean())
    percentage_probability : Mapped[float] = mapped_column(Float(decimal_return_scale=2))



if __name__ == '__main__':
    Base.metadata.create_all(engine)
