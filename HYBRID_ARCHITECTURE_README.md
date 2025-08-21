# Hybrid Architecture Implementation

This document describes the implementation of a hybrid architecture for SGLang that offloads HTTP response formatting from TokenizerManager to DetokenizerManager processes, improving performance and scalability.

## Overview

The hybrid architecture addresses the bottleneck in the current system where a single TokenizerManager processes all responses from multiple DetokenizerManager processes. By distributing HTTP response formatting across multiple DetokenizerManager workers and using shared memory for coordination, we achieve:

## Implementation Phases

The hybrid architecture was implemented in **5 complete phases**:

### **Phase 1: Multi-Process Configuration** ✅
- Added new server arguments for multi-process TokenizerManager
- Added CLI argument parsing for the new options
- Implemented automatic flag setting based on configuration

### **Phase 2: Shared Memory Infrastructure** ✅
- Created `SharedMemoryManager` for thread-safe coordination
- Implemented request state management and response queuing
- Added automatic cleanup and monitoring capabilities

### **Phase 3: Enhanced Components** ✅
- Enhanced `DetokenizerManager` with HTTP response formatting
- Enhanced `TokenizerManager` with shared memory coordination
- Created `HybridCoordinator` for overall architecture management

### **Phase 4: HTTP Integration** ✅
- Implemented `EnhancedHTTPServer` with hybrid architecture support
- Added FastAPI routes for the enhanced endpoints
- Integrated request routing and response streaming

### **Phase 5: Process Management & Launch** ✅
- Created `EnhancedEngine` for process lifecycle management
- Implemented worker process startup and shutdown
- Added health monitoring and graceful termination

- **Better Performance**: Parallel HTTP response formatting
- **Improved Scalability**: Multiple workers can handle requests simultaneously
- **Request Affinity**: Each request stays with its assigned DetokenizerManager
- **Shared State Management**: Coordinated state management without race conditions

## Architecture Components

### 1. Shared Memory Manager (`SharedMemoryManager`)

**Location**: `python/sglang/srt/managers/shared_memory_manager.py`

**Purpose**: Manages shared memory for request states, response queues, and coordination between processes.

**Key Features**:
- Thread-safe operations with locks
- Automatic cleanup of old requests
- Request state tracking and management
- Response chunk queuing and retrieval

**Usage**:
```python
shared_memory = SharedMemoryManager(max_requests=1000)
shared_memory.register_request("req_123", initial_state)
shared_memory.add_response_chunk("req_123", response_chunk)
```

### 2. Enhanced DetokenizerManager (`EnhancedDetokenizerManager`)

**Location**: `python/sglang/srt/managers/enhanced_detokenizer_manager.py`

**Purpose**: Extends the base DetokenizerManager with HTTP response formatting capabilities.

**Key Features**:
- HTTP response formatting (streaming and non-streaming)
- Request affinity tracking
- Response caching for retransmission
- Integration with shared memory

**Usage**:
```python
enhanced_dm = EnhancedDetokenizerManager(
    server_args,
    port_args,
    shared_memory_manager,
    worker_id=0
)
enhanced_dm.assign_request("req_123")
```

### 3. Enhanced TokenizerManager (`EnhancedTokenizerManager`)

**Location**: `python/sglang/srt/managers/enhanced_tokenizer_manager.py`

**Purpose**: Coordinates with shared memory and enhanced DetokenizerManagers for request handling.

**Key Features**:
- Shared memory coordination
- Request registration and tracking
- Response streaming from shared memory
- Detokenizer assignment management

**Usage**:
```python
enhanced_tm = EnhancedTokenizerManager(
    server_args,
    port_args,
    shared_memory_manager,
    worker_id=0
)
```

### 4. Hybrid Coordinator (`HybridCoordinator`)

**Location**: `python/sglang/srt/managers/hybrid_coordinator.py`

**Purpose**: Manages the overall hybrid architecture and coordinates between workers.

**Key Features**:
- Worker management and registration
- Request routing and load balancing
- Health monitoring and coordination
- Statistics and monitoring

**Usage**:
```python
coordinator = HybridCoordinator(server_args)
coordinator.add_tokenizer_worker(0, tokenizer_worker)
coordinator.add_detokenizer_worker(0, detokenizer_worker)
coordinator.start_coordination()
```

## Configuration

### Server Arguments

The hybrid architecture is controlled by new server arguments:

```python
# Multi-process TokenizerManager
tokenizer_worker_num: int = 1                    # Number of TokenizerManager workers
enable_multi_tokenizer: bool = False             # Enable hybrid mode
detokenizer_processes: int = 1                   # Number of DetokenizerManager processes
detokenizer_load_balance_policy: str = "round_robin"  # Load balancing policy
```

**Load Balancing Policies**:
- `round_robin`: Simple round-robin distribution
- `least_loaded`: Select worker with lowest load
- `weighted`: Weighted selection based on load and other factors

### CLI Arguments

```bash
--tokenizer-worker-num 2                    # 2 TokenizerManager workers
--detokenizer-processes 4                   # 4 DetokenizerManager processes
--detokenizer-load-balance-policy least_loaded  # Use least-loaded policy
```

## Request Flow

### 1. Request Arrival
```
HTTP Request → FastAPI → Enhanced TokenizerManager Worker
```

### 2. Request Registration
```
Enhanced TokenizerManager → Shared Memory Manager
                           → Register request state
                           → Create response queue
```

### 3. Request Routing
```
Hybrid Coordinator → Route to appropriate workers
                  → Assign DetokenizerManager for affinity
                  → Update load balancing
```

### 4. Response Generation
```
Scheduler → Enhanced DetokenizerManager → HTTP formatting → Shared Memory
```

### 5. Response Streaming
```
Enhanced TokenizerManager → Shared Memory → Response chunks → Client
```

## Race Condition Prevention

The implementation includes several mechanisms to prevent race conditions:

### 1. Request Affinity
- Each request is assigned to a specific DetokenizerManager
- No load balancing within a single request's lifecycle
- Prevents chunk ordering issues

### 2. Shared Memory Coordination
- Thread-safe operations with locks
- Atomic state updates
- Consistent request lifecycle management

### 3. Message Passing
- Uses shared memory queues instead of direct method calls
- Asynchronous coordination to avoid blocking
- Clean separation of concerns

## Performance Benefits

### 1. Parallel Processing
- Multiple DetokenizerManager workers format responses simultaneously
- No single bottleneck for HTTP formatting
- Better CPU utilization across workers

### 2. Reduced Latency
- HTTP formatting happens closer to token generation
- Fewer inter-process communication hops
- Faster response delivery to clients

### 3. Improved Throughput
- Better load distribution across workers
- Reduced contention for shared resources
- More efficient resource utilization

## Testing

### Test Script

Run the test script to verify the implementation:

```bash
python test_hybrid_architecture.py
```

The test script demonstrates:
- Worker creation and registration
- Request routing and load balancing
- Response generation and streaming
- Coordination and monitoring

### Expected Output

```
🧪 Hybrid Architecture Test Suite
==================================================
🚀 Starting hybrid architecture test
✅ Server arguments created:
   - Tokenizer workers: 2
   - Detokenizer processes: 3
   - Load balance policy: least_loaded
   - Hybrid mode enabled: True
✅ Port arguments created
✅ Shared memory manager created
✅ Hybrid coordinator created
✅ TokenizerManager worker 0 created and added to coordinator
✅ TokenizerManager worker 1 created and added to coordinator
✅ DetokenizerManager worker 0 created and added to coordinator
✅ DetokenizerManager worker 1 created and added to coordinator
✅ DetokenizerManager worker 2 created and added to coordinator
✅ Coordination started
🔄 Testing request routing...
✅ Request req_001 routed to TM worker 0 and DM worker 0
✅ Request req_002 routed to TM worker 1 and DM worker 1
✅ Request req_003 routed to TM worker 0 and DM worker 2
🔄 Testing response streaming...
📡 Streaming response for request req_001...
   📦 Response chunk: stream_chunk
   📦 Response chunk: stream_chunk
   📦 Response chunk: stream_chunk
   📦 Response chunk: final_response
   ✅ Final response received for req_001
...
🎉 Hybrid architecture test completed successfully!
```

## Integration with Existing Code

### Backward Compatibility

The hybrid architecture is designed to be backward compatible:

- **Default behavior**: Single TokenizerManager (existing behavior)
- **Hybrid mode**: Enable with `--tokenizer-worker-num > 1`
- **Fallback**: Automatically falls back to original implementation if issues occur

### Migration Path

1. **Phase 1**: Deploy with single worker (no changes to existing behavior)
2. **Phase 2**: Enable hybrid mode with monitoring
3. **Phase 3**: Scale up workers based on performance metrics

## Monitoring and Debugging

### Statistics

Access coordination statistics:

```python
stats = coordinator.get_coordination_stats()
print(f"Active requests: {stats['active_requests']}")
print(f"Worker loads: {stats['detokenizer_loads']}")
```

### Worker Health

Monitor individual worker health:

```python
for worker_id, worker in tokenizer_workers.items():
    worker_stats = worker.get_worker_stats()
    print(f"Worker {worker_id}: {worker_stats}")
```

### Logging

Enable debug logging for detailed information:

```python
logging.getLogger("sglang.srt.managers").setLevel(logging.DEBUG)
```

## Future Enhancements

### 1. Dynamic Scaling
- Automatic worker scaling based on load
- Health-based worker replacement
- Load prediction and proactive scaling

### 2. Advanced Load Balancing
- Request size-based routing
- Priority-based scheduling
- Custom load balancing algorithms

### 3. Performance Optimization
- Response caching and compression
- Batch processing optimizations
- Memory usage optimization

## Troubleshooting

### Common Issues

1. **Shared Memory Errors**
   - Check system shared memory limits
   - Verify multiprocessing compatibility
   - Check for memory leaks

2. **Worker Coordination Issues**
   - Verify worker registration
   - Check coordination loop status
   - Monitor worker health

3. **Performance Issues**
   - Monitor worker loads
   - Check request routing efficiency
   - Verify load balancing policy

### Debug Commands

```python
# Check coordination status
print(coordinator.get_coordination_stats())

# Monitor worker health
for worker in coordinator.tokenizer_workers.values():
    print(worker.get_worker_stats())

# Check shared memory status
print(shared_memory_manager.get_request_stats())
```

## Conclusion

The hybrid architecture provides a significant performance improvement over the current single-TokenizerManager design while maintaining backward compatibility and preventing race conditions. The implementation uses proven patterns like shared memory, message passing, and request affinity to create a robust and scalable system.

By distributing HTTP response formatting across multiple DetokenizerManager workers and using coordinated shared memory management, the system can handle higher throughput with lower latency, making it suitable for production deployments with high request volumes.
