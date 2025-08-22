# Model Deployment Architectures: Lecture Notes

## 1. Evolution of Model Deployment Architectures

### Historical Context
As machine learning and AI applications evolved, so did the approaches to deploying models. Initially, models were primarily deployed directly with applications, but with increased model complexity and size, alternative deployment strategies emerged.

## 2. Model Deployment Strategies

### Offline Deployment
- **Definition**: Model file is packaged and distributed directly with the application
- **Historical significance**: The original approach to model deployment when models were smaller
- **Advantages**:
  - Works without internet connection
  - Faster inference (no network latency)
  - More privacy (data stays on device)
- **Disadvantages**:
  - Increases application size
  - Difficult to update models
  - Limited by device hardware capabilities

### Online Deployment
- **Definition**: Model is hosted on a server and accessed via API calls
- **Evolution**: Emerged as models grew in size and complexity
- **Advantages**:
  - Models can be updated without app updates
  - Supports larger models that wouldn't fit on devices
  - Consistent performance regardless of client device
- **Disadvantages**:
  - Requires internet connection
  - Introduces network latency
  - Potential privacy concerns (data leaves device)

## 3. Client-Server Architecture

### Modern Implementation
As applications became more sophisticated, a three-tier architecture emerged:

```
Client 

MOBİL'de : 

Client -----> Web Backend -----> Model
```




