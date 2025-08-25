DOCKER 

IMAGE :

- dependency / OS
- C++ compiler
- c# compiler
- codebase

WSL (Linux / Ubuntu)


```python
# SQLAlchemy Database Connection Example

# Connect to PostgreSQL and persist temporary user data using SQLAlchemy ORM

## Requirements
- PostgreSQL running on localhost:5433
- Database name: aiapp
- Username: postgres
- Password: postgres

## Steps to run
1. Ensure PostgreSQL is running
2. Install requirements: `pip install sqlalchemy psycopg2-binary`
3. Run script: `python db_connection.py`

## Code Implementation
- Created SQLAlchemy models with code-first approach
- Connected to PostgreSQL database
- Persisted temporary user data
- Implemented checks to avoid duplicate entries
- Used proper session management with commit and rollback

## Database Schema
- users table with columns:
  - id (primary key)
  - username (unique)
  - email (unique)
  - hashed_password
  - full_name
  - is_active
  - created_at
  - last_login
```



