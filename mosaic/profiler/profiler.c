/*
 * MOSAIC Hardware Counter Profiler
 * Uses perf_event_open() to measure workload fingerprints.
 * Requires: CAP_PERFMON or perf_event_paranoid <= 1
 *
 * Usage: ./profiler <duration_ms> <workload_class> <output.json>
 *        ./profiler 5000 inference_critical /tmp/fp.json
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <errno.h>
#include <time.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <linux/perf_event.h>
#include <asm/unistd.h>

#define MAX_FDS 8
#define NS_PER_MS 1000000LL

typedef struct { uint64_t value, time_enabled, time_running; } perf_read_t;

static int fds[MAX_FDS];
static int n_fds = 0;

static long perf_event_open(struct perf_event_attr *hw, pid_t pid,
                             int cpu, int group_fd, unsigned long flags) {
    return syscall(__NR_perf_event_open, hw, pid, cpu, group_fd, flags);
}

static int open_counter(uint32_t type, uint64_t config, int group_fd) {
    struct perf_event_attr pe = {0};
    pe.type       = type;   pe.size = sizeof(pe);
    pe.config     = config; pe.disabled = (group_fd == -1) ? 1 : 0;
    pe.exclude_hv = 1;
    pe.read_format = PERF_FORMAT_TOTAL_TIME_ENABLED | PERF_FORMAT_TOTAL_TIME_RUNNING;
    int fd = (int)perf_event_open(&pe, 0, -1, group_fd, 0);
    if (fd < 0) fd = (int)perf_event_open(&pe, -1, 0, group_fd, 0);
    return fd;
}

static int setup(void) {
    static const struct { uint32_t type; uint64_t config; } evts[] = {
        {PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS},
        {PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES},
        {PERF_TYPE_HARDWARE, PERF_COUNT_HW_CACHE_MISSES},
        {PERF_TYPE_HARDWARE, PERF_COUNT_HW_CACHE_REFERENCES},
        {PERF_TYPE_HARDWARE, PERF_COUNT_HW_BRANCH_MISSES},
        {PERF_TYPE_HARDWARE, PERF_COUNT_HW_BRANCH_INSTRUCTIONS},
        {PERF_TYPE_HARDWARE, PERF_COUNT_HW_BUS_CYCLES},
    };
    int group = -1, n = (int)(sizeof(evts)/sizeof(evts[0]));
    for (int i = 0; i < n && n_fds < MAX_FDS; i++) {
        int fd = open_counter(evts[i].type, evts[i].config, (i==0)?-1:group);
        if (fd < 0) { fprintf(stderr,"[warn] counter %d unavailable\n",i); continue; }
        if (i == 0) group = fd;
        fds[n_fds++] = fd;
    }
    return (n_fds >= 2) ? 0 : -1;
}

static uint64_t read_counter(int fd) {
    perf_read_t b; ssize_t r = read(fd, &b, sizeof(b));
    if (r != sizeof(b) || b.time_running == 0) return 0;
    return (uint64_t)((double)b.value * (double)b.time_enabled / (double)b.time_running);
}

int main(int argc, char *argv[]) {
    if (argc < 4) {
        fprintf(stderr,"Usage: %s <duration_ms> <workload_class> <output.json>\n",argv[0]);
        return 1;
    }
    uint64_t dur_ms = (uint64_t)atoll(argv[1]);
    if (dur_ms < 100 || dur_ms > 300000) { fprintf(stderr,"dur 100-300000\n"); return 1; }

    if (setup() < 0) return 1;
    ioctl(fds[0], PERF_EVENT_IOC_RESET,  PERF_IOC_FLAG_GROUP);
    ioctl(fds[0], PERF_EVENT_IOC_ENABLE, PERF_IOC_FLAG_GROUP);
    struct timespec ts = {(time_t)(dur_ms/1000), (long)((dur_ms%1000)*NS_PER_MS)};
    nanosleep(&ts, NULL);
    ioctl(fds[0], PERF_EVENT_IOC_DISABLE, PERF_IOC_FLAG_GROUP);

    uint64_t instr=read_counter(fds[0]), cycles=read_counter(fds[1]);
    uint64_t llc_miss=read_counter(fds[2]), llc_ref=read_counter(fds[3]);
    uint64_t br_miss=read_counter(fds[4]),  br_instr=read_counter(fds[5]);

    double ipc      = cycles>0 ? (double)instr/cycles : 0;
    double llc_rate = llc_ref>0 ? (double)llc_miss/llc_ref : 0;
    double bw_gbs   = (double)llc_miss*64.0/(dur_ms/1000.0*1e9);
    double br_rate  = br_instr>0 ? (double)br_miss/br_instr : 0;

    FILE *f = strcmp(argv[3],"-")==0 ? stdout : fopen(argv[3],"w");
    if (!f) { perror("fopen"); return 1; }
    fprintf(f, "{\n"
        "  \"workload_class\": \"%s\",\n"
        "  \"duration_ms\": %lu,\n"
        "  \"ipc\": %.4f,\n"
        "  \"llc_miss_rate\": %.6f,\n"
        "  \"mem_bw_gbs\": %.4f,\n"
        "  \"branch_miss_rate\": %.6f,\n"
        "  \"instructions\": %lu,\n"
        "  \"cycles\": %lu,\n"
        "  \"llc_misses\": %lu,\n"
        "  \"llc_accesses\": %lu\n"
        "}\n",
        argv[2], dur_ms, ipc, llc_rate, bw_gbs, br_rate, instr, cycles, llc_miss, llc_ref);
    if (f != stdout) fclose(f);

    printf("[profiler] %s  IPC=%.3f  LLC_miss=%.4f  MemBW=%.2fGB/s\n",
           argv[2], ipc, llc_rate, bw_gbs);
    for (int i=0;i<n_fds;i++) close(fds[i]);
    return 0;
}
