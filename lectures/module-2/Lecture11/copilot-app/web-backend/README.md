# Web Backend API Development: Lecture Notes

## 1. Introduction to Web Communication

### Historical Context
In the early days of the internet, communication between systems was complex and non-standardized. 
With the growth of distributed systems, the need for standardized communication methods became essential.

### What is an API?
API (Application Programming Interface) is a set of rules and protocols that allows different software applications to communicate with each other.
- Enables the connection between Mobile Frontend (Flutter) and Web Backend
- Bridges the gap between existing Models and Backend services

## 2. Communication Standards

### JSON as a Communication Language
- JSON (JavaScript Object Notation) became the standard for data interchange
- Lightweight, human-readable format
- Language-agnostic (works across different programming languages like dart, C#, JavaScript, C++)
- Replaced older formats like XML for many applications

## 3. API Architecture

### Endpoints
- Entry points to access the API's functionality
- Example: `/find-plate` or `/find-plates` for vehicle plate lookup
- Each endpoint serves a specific business function

### REST Architecture
- Representational State Transfer
- Stateless communication model
  - Each request from client contains all information needed to complete the request
  - No client context is stored on the server between requests
- Uses standard HTTP methods:
  - GET: Retrieve resources
  - POST: Create resources
  - PUT/PATCH: Update resources
  - DELETE: Remove resources

## 4. API Security

### Authentication & Authorization
- Authentication: Verifies who the user is (identity)
- Authorization: Determines what the authenticated user is allowed to do (permissions)
- Token-based authentication (e.g., JWT) maintains the stateless nature of REST APIs
- Client flow: `/login` → receive token → use token for `/find-plate` and other endpoints

### Statelessness
- Each request is independent
- No session state maintained on server
- Improves scalability and reliability

## 5. API Development Best Practices

### Validation
- Input validation prevents security vulnerabilities
- Data format validation ensures system integrity
- Response validation maintains API contract

### Business Rules
- Encapsulate domain logic
- Separate from presentation layer
- Ensure consistency across different clients

## 6. Client-Server Interaction

### Request-Response Cycle
- Client initiates requests
- Server processes and returns responses
- Success or failure status codes indicate outcome
- Proper error handling improves user experience

### Supported Programming Languages
Modern API development supports multiple programming languages:
- Dart (Flutter)
- C#
- JavaScript
- C++

## 7. Mobile-Backend Integration

### Architecture Overview
- Mobile applications (Flutter) communicate with backend services over the internet
- Use of RESTful APIs to perform operations like login, and data retrieval
- Stateless nature of APIs allows for scalable and efficient communication

### Example Workflow
1. Mobile app requests user login (`/login`)
2. Backend authenticates and returns a token
3. Mobile app uses token to request protected resources (`/find-plate`)
4. Backend validates token and serves the request

### Considerations
- Ensure secure transmission of data (HTTPS)
- Handle token storage and expiration on the client-side
- Regularly update and patch backend services to protect against vulnerabilities

## 8. Security Concepts in API Development

### HASHING (irreversible)
- **Definition**: A one-way function that converts input data into a fixed-length string of characters
- **Properties**:
  - Deterministic: Same input always produces the same output
  - Irreversible: Cannot derive the original input from the hash output
  - Collision-resistant: Different inputs should not produce the same hash
- **Use cases**:
  - Password storage in databases
  - Data integrity verification
  - Digital signatures
- **Example**:
  - Input: abc123 → Hash output: e325ırewfmksdf3042p5452vrek2kr
- **Common algorithms**:
  - SHA-256, SHA-3
  - bcrypt, Argon2 (specifically designed for password hashing)
- **Security best practice**: Always use salting with password hashing to prevent rainbow table attacks

### ENCRYPTION (reversible)
- **Definition**: A two-way function that transforms data to make it unreadable without proper decryption key
- **Properties**:
  - Reversible: Original data can be recovered with the correct key
  - Key-dependent: Security depends on keeping the key secure
- **Types**:
  - Symmetric encryption: Same key for encryption and decryption
  - Asymmetric encryption: Different keys for encryption (public) and decryption (private)
- **Use cases**:
  - Secure data transmission
  - Encrypted storage of sensitive data
  - Secure token generation (JWT)
- **Example**:
  - Encryption: abc123 + key → fngşesgrNBDFKFNGFDSMfsldfö
  - Decryption: fngşesgrNBDFKFNGFDSMfsldfö + key → abc123
- **Common algorithms**:
  - Symmetric: AES, ChaCha20
  - Asymmetric: RSA, ECC

### When to use which?
- Use **hashing** when you need to:
  - Verify data without storing the original (passwords)
  - Check data integrity
  - Create unique identifiers

- Use **encryption** when you need to:
  - Store sensitive data that must be retrieved later
  - Securely transmit data
  - Implement authentication tokens

