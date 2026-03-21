CC     := gcc
CFLAGS := -O2 -Wall -Wextra -std=c11
TARGET := profiler
.PHONY: all clean check
all: $(TARGET)
$(TARGET): profiler.c
	$(CC) $(CFLAGS) -o $@ $^
	@echo "Built $(TARGET)"
check:
	@echo "perf_event_paranoid = $$(cat /proc/sys/kernel/perf_event_paranoid 2>/dev/null || echo n/a)"
	@echo "Set to <=1 for hardware counters: sudo sysctl -w kernel.perf_event_paranoid=1"
clean:
	rm -f $(TARGET)
