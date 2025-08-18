# Multi-Process DetokenizerManager

This document describes the new multi-process DetokenizerManager feature that allows running multiple DetokenizerManager processes in parallel for improved throughput.

## Overview

The multi-process DetokenizerManager feature enables:
- **Parallel processing** of multiple detokenization requests
- **Request affinity** to ensure tokens from the same request are processed by the same DetokenizerManager
- **Load balancing** across multiple DetokenizerManager processes
- **Direct communication** to the load balancer for reduced latency

## Architecture

```
┌─────────────────┐    ┌──────────────────────┐    ┌─────────────────────┐
│   Scheduler     │───▶│ DetokenizerCoordinator│───▶│ DetokenizerManager 0│
│                 │    │                      │    │                     │
└─────────────────┘    │  - Request affinity  │    │  - Direct to LB     │
                       │  - Load balancing    │    └─────────────────────┘
                       └──────────────────────┘    ┌─────────────────────┐
                                                   │ DetokenizerManager 1│
                                                   │                     │
                                                   │  - Direct to LB     │
                                                   └─────────────────────┘
                                                           │
                                                           ▼
                                                   ┌─────────────────────┐
                                                   │   Load Balancer     │
                                                   │                     │
                                                   └─────────────────────┘
```

## Configuration

### Command Line Arguments

```bash
# Launch with 4 DetokenizerManager processes using round-robin load balancing
python -m sglang.launch_server \
    --model-path /path/to/model \
    --detokenizer-processes 4 \
    --detokenizer-load-balance-policy round_robin

# Launch with least-loaded load balancing
python -m sglang.launch_server \
    --model-path /path/to/model \
    --detokenizer-processes 8 \
    --detokenizer-load-balance-policy least_loaded

# Launch with weighted load balancing
python -m sglang.launch_server \
    --model-path /path/to/model \
    --detokenizer-processes 6 \
    --detokenizer-load-balance-policy weighted
```

### Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--detokenizer-processes` | int | 1 | Number of DetokenizerManager processes to run |
| `--detokenizer-load-balance-policy` | str | "round_robin" | Load balancing policy: "round_robin", "least_loaded", or "weighted" |

## Load Balancing Policies

### 1. Round Robin
- **Description**: Distributes requests sequentially across DetokenizerManager processes
- **Pros**: Simple, predictable, even distribution
- **Cons**: Doesn't consider actual load
- **Best for**: Balanced workloads, simple setups

### 2. Least Loaded
- **Description**: Assigns new requests to the DetokenizerManager with the lowest current load
- **Pros**: Adapts to actual load, prevents overloading
- **Cons**: More complex, requires load tracking
- **Best for**: Variable request sizes, performance-critical scenarios

### 3. Weighted
- **Description**: Considers both current load and request size when assigning requests
- **Pros**: Balances load while considering request complexity
- **Cons**: Most complex, requires tuning
- **Best for**: Mixed workloads with varying request sizes

## Request Affinity

**Critical Feature**: Tokens from the same request are always processed by the same DetokenizerManager process to ensure:
- **Serialization**: Proper order of token processing
- **State consistency**: Decode status and incremental updates are maintained
- **Performance**: No cross-process communication overhead for the same request

## Usage Examples

### Basic Usage
```bash
# Launch with 4 DetokenizerManager processes
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --detokenizer-processes 4
```

### PD-Disagg Mode
```bash
# Prefill server
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --disaggregation-mode prefill \
    --detokenizer-processes 2

# Decode server with multi-process DetokenizerManager
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --disaggregation-mode decode \
    --detokenizer-processes 4 \
    --detokenizer-load-balance-policy least_loaded
```

### Performance Tuning
```bash
# High-throughput configuration
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --detokenizer-processes 8 \
    --detokenizer-load-balance-policy weighted \
    --detokenizer-log-interval 50
```

## Monitoring and Debugging

### Logging
Enable detailed logging to monitor performance:
```bash
--enable-detokenizer-logging \
--detokenizer-log-interval 100 \
--log-level debug
```

### Performance Metrics
Each DetokenizerManager process provides:
- Total requests processed
- Total tokens processed
- Average processing time
- Throughput (tokens/second)
- Active request count

### Health Checks
- Process monitoring and automatic restart
- Load distribution statistics
- Request affinity tracking

## Implementation Details

### Components

1. **DetokenizerCoordinator**: Manages request distribution and load balancing
2. **DetokenizerManager**: Individual processes that handle detokenization
3. **PortArgs**: Extended to support multiple IPC channels
4. **Engine**: Modified to launch multiple processes

### Communication Flow

1. **Scheduler** → **DetokenizerCoordinator** (for multi-process mode)
2. **DetokenizerCoordinator** → **DetokenizerManager** (with load balancing)
3. **DetokenizerManager** → **Load Balancer** (direct communication)

### Backward Compatibility

- Single DetokenizerManager mode is preserved
- Existing configurations work unchanged
- New features are opt-in via command line arguments

## Performance Considerations

### Benefits
- **Increased throughput**: Parallel processing of multiple requests
- **Better resource utilization**: Distributes load across multiple processes
- **Reduced latency**: Direct communication to load balancer
- **Scalability**: Easy to add more processes for higher throughput

### Trade-offs
- **Memory usage**: Each process maintains its own decode status
- **IPC overhead**: Coordination between coordinator and processes
- **Complexity**: More complex process management and monitoring

### Recommendations
- Start with 2-4 processes for most use cases
- Use `least_loaded` policy for variable workloads
- Monitor memory usage per process
- Adjust process count based on CPU cores and memory availability

## Troubleshooting

### Common Issues

1. **Port conflicts**: Ensure sufficient port range for multiple processes
2. **Memory pressure**: Monitor per-process memory usage
3. **Load imbalance**: Check load balancing policy and request distribution
4. **Process failures**: Monitor process health and restart behavior

### Debug Commands
```bash
# Check process status
ps aux | grep detokenizer

# Monitor IPC communication
netstat -an | grep <port>

# Check logs for specific process
tail -f logs/detokenizer_<process_id>.log
```

## Future Enhancements

- **Dynamic scaling**: Add/remove processes based on load
- **Advanced load balancing**: Machine learning-based request distribution
- **Shared memory**: Reduce memory overhead across processes
- **Metrics integration**: Prometheus/Grafana dashboards
- **Health monitoring**: Automatic failure detection and recovery
