from sqlalchemy import Column, Integer, String, DateTime, Float
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import datetime
from collections import namedtuple
from pathlib import Path
from copy import deepcopy

Base = declarative_base()
Metric = namedtuple('Metric', ['name', 'value'])


class MetaData(Base):
    __tablename__ = "metadata"

    id = Column(Integer, primary_key=True)
    dataset = Column(String)
    model = Column(String)
    hyper = Column(String)
    metric = Column(String)
    value = Column(Float)
    fold = Column(Integer)
    path = Column(String)
    ts = Column(DateTime, default=datetime.datetime.now)

    def update_metrics(self, new_metrics: Metric):
        new_row = deepcopy(self)
        new_row.metric = new_metrics.name
        new_row.value = new_metrics.value
        return new_row


def get_engine(db_path):
    path = Path(db_path)
    engine = create_engine(db_path, echo=False)

    if not path.exists():
        Base.metadata.create_all(engine)

    return engine


def get_session(engine):
    return sessionmaker(bind=engine)()


def insert(rows, engine):
    session = get_session(engine)
    [session.add(row) for row in rows]
    session.commit()


def generate_rows(tuples, base_row: MetaData):
    rows = [deepcopy(base_row) for _ in range(len(tuples))]
    rows = [r.update_metrics(Metric._make(t)) for r, t in zip(rows, tuples)]
    return rows
