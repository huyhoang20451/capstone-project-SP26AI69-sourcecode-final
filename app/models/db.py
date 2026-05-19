import os
import time
import datetime
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, ForeignKey, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from sqlalchemy.exc import OperationalError
from dotenv import load_dotenv
from memori import Memori

# 1. Nạp file .env ngay khi khởi động
load_dotenv()

# 2. Lấy URL từ biến môi trường
# Nếu file .env chuẩn, DATABASE_URL sẽ là link PostgreSQL của bạn
DATABASE_URL = os.getenv("DATABASE_URL")

# 3. Logic dự phòng (Fallback)
if not DATABASE_URL:
    # Nếu không tìm thấy .env hoặc biến bị thiếu, mới dùng SQLite để tránh sập app
    DATABASE_URL = "sqlite:///./web_emotion_chat.db"
    print("⚠️  Cảnh báo: Không tìm thấy DATABASE_URL trong .env. Đang dùng SQLite dự phòng.")
else:
    # In ra để bạn kiểm tra xem có đúng là đang dùng postgres không
    print(f"✅ Đang kết nối tới: {DATABASE_URL.split('://')[0]}")

# 4. Cấu hình Engine linh hoạt cho cả Postgres và SQLite
engine_kwargs = {}
if DATABASE_URL.startswith("postgresql"):
    engine_kwargs["pool_pre_ping"] = True
else:
    # Cấu hình đặc thù cho SQLite để chạy mượt với FastAPI
    engine_kwargs["connect_args"] = {"check_same_thread": False}

engine = create_engine(DATABASE_URL, **engine_kwargs)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

mem = Memori(conn=SessionLocal())
# --- Models ---

class Conversation(Base):
    __tablename__ = "conversations"
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(200))
    # Sử dụng datetime.datetime.now để lấy giờ hệ thống máy bạn cho chính xác
    created_at = Column(DateTime, default=datetime.datetime.now)
    
    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan")

class Message(Base):
    __tablename__ = "messages"
    
    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False)
    role = Column(String(20), nullable=False)
    content = Column(Text, nullable=False)
    emotion = Column(String(50))
    ml_detail_emotion = Column(String(50))
    emotion_model_used = Column(String(100), default="Machine Learning")
    timestamp = Column(DateTime, default=datetime.datetime.now)
    
    conversation = relationship("Conversation", back_populates="messages")

# --- Database Utils ---

def init_db():
    """Khởi tạo database và kiểm tra kết nối (đặc biệt quan trọng khi dùng Docker)"""
    MAX_DB_RETRIES = int(os.getenv("DB_CONNECT_MAX_RETRIES", "20"))
    DB_RETRY_SECONDS = float(os.getenv("DB_CONNECT_RETRY_SECONDS", "2"))

    # Chỉ cần chạy vòng lặp retry nếu là PostgreSQL
    if DATABASE_URL.startswith("postgresql"):
        print("Đang kiểm tra kết nối PostgreSQL...")
        for attempt in range(1, MAX_DB_RETRIES + 1):
            try:
                with engine.connect() as conn:
                    conn.execute(text("SELECT 1"))
                print("Kết nối PostgreSQL thành công!")
                break
            except OperationalError:
                if attempt == MAX_DB_RETRIES:
                    print("❌ Lỗi: Không thể kết nối tới PostgreSQL sau nhiều lần thử.")
                    raise
                time.sleep(DB_RETRY_SECONDS)
    
    # Tạo bảng nếu chưa tồn tại
    Base.metadata.create_all(bind=engine)

    mem.config.storage.build()
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()