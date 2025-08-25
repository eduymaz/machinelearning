from sqlalchemy import create_engine, Column, Integer, String, Boolean, DateTime, ForeignKey, Table, MetaData
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
import uuid

# Database connection string
DATABASE_URL = "postgresql://postgres:postgres@localhost:5433/aiapp"

# Create SQLAlchemy engine
engine = create_engine(DATABASE_URL)

# Create declarative base
Base = declarative_base()

# Define User model
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    full_name = Column(String, nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime, nullable=True)
    
    # Relationship with user roles (if needed)
    # roles = relationship("Role", secondary="user_roles", back_populates="users")
    
    def __repr__(self):
        return f"<User(id={self.id}, username={self.username}, email={self.email})>"

# Define other models if needed
# class Role(Base):
#     __tablename__ = "roles"
#     
#     id = Column(Integer, primary_key=True, index=True)
#     name = Column(String, unique=True)
#     description = Column(String, nullable=True)
#     
#     users = relationship("User", secondary="user_roles", back_populates="roles")

# Create all tables in the database
Base.metadata.create_all(bind=engine)

# Create a session
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
db = SessionLocal()

# Sample temporary user data
temp_users = [
    {
        "username": "johndoe",
        "email": "john@example.com",
        "hashed_password": "hashed_password_1",
        "full_name": "John Doe",
        "is_active": True
    },
    {
        "username": "janedoe",
        "email": "jane@example.com",
        "hashed_password": "hashed_password_2",
        "full_name": "Jane Doe",
        "is_active": True
    },
    {
        "username": "admin",
        "email": "admin@example.com",
        "hashed_password": "hashed_admin_password",
        "full_name": "Administrator",
        "is_active": True
    }
]

def persist_users():
    """Persist temporary user data to the database."""
    try:
        # Check if users already exist to avoid duplicates
        for user_data in temp_users:
            existing_user = db.query(User).filter(
                User.username == user_data["username"]
            ).first()
            
            if not existing_user:
                # Create new user object
                new_user = User(
                    username=user_data["username"],
                    email=user_data["email"],
                    hashed_password=user_data["hashed_password"],
                    full_name=user_data["full_name"],
                    is_active=user_data["is_active"],
                    created_at=datetime.utcnow()
                )
                
                # Add user to session
                db.add(new_user)
                print(f"Added user: {new_user.username}")
            else:
                print(f"User {user_data['username']} already exists.")
        
        # Commit the session to persist changes
        db.commit()
        print("All users successfully persisted to database.")
    
    except Exception as e:
        db.rollback()
        print(f"Error persisting users: {e}")
    finally:
        db.close()

def list_users():
    """List all users from the database."""
    try:
        users = db.query(User).all()
        print("\nUsers in database:")
        for user in users:
            print(f"ID: {user.id}, Username: {user.username}, Email: {user.email}")
    except Exception as e:
        print(f"Error listing users: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    # Persist temporary users to database
    persist_users()
    
    # List users from database
    list_users()
