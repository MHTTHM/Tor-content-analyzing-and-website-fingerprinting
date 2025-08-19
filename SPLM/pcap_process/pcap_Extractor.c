#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <winsock2.h>
#include <windows.h>
#include <pcap.h>
#include "uthash.h"
#include <windows.h>

#pragma pack(push, 1)
typedef struct ip_header {
    BYTE ver_ihl;
    BYTE tos;
    USHORT total_length;
    USHORT ident;
    USHORT flags_fo;
    BYTE ttl;
    BYTE protocol;
    USHORT checksum;
    ULONG src;
    ULONG dst;
} IP_HEADER;

typedef struct tcp_header {
    USHORT src_port;
    USHORT dst_port;
    ULONG seq_num;
    ULONG ack_num;
    USHORT hdr_len_flags;
    USHORT window;
    USHORT checksum;
    USHORT urg_ptr;
} TCP_HEADER;
#pragma pack(pop)

typedef struct {
    double delta;
    int size;
} packet_info;

#pragma pack(push, 1)
typedef struct {
    uint32_t src_ip;
    uint32_t dst_ip;
    uint16_t src_port;
    uint16_t dst_port;
} stream_key;
#pragma pack(pop)

typedef struct stream {
    stream_key key;
    double first_time;
    uint32_t sender_ip;
    uint16_t sender_port;
    packet_info *packets;
    size_t count;
    size_t capacity;
    int positive_count;  // Track count of positive packet sizes
    int negative_count;  // Track count of negative packet sizes
    UT_hash_handle hh;
} stream;

static void normalize_key(stream_key *key) {
    uint32_t host_src_ip = ntohl(key->src_ip);
    uint32_t host_dst_ip = ntohl(key->dst_ip);
    uint16_t host_src_port = ntohs(key->src_port);
    uint16_t host_dst_port = ntohs(key->dst_port);

    if (host_src_ip > host_dst_ip || 
        (host_src_ip == host_dst_ip && host_src_port > host_dst_port)) {
        uint32_t tmp_ip = key->src_ip;
        key->src_ip = key->dst_ip;
        key->dst_ip = tmp_ip;
        uint16_t tmp_port = key->src_port;
        key->src_port = key->dst_port;
        key->dst_port = tmp_port;
    }
}

static void add_packet(stream *s, double time, int size) {
    if (!s) return;

    if (s->count >= s->capacity) {
        size_t new_cap = s->capacity ? s->capacity * 2 : 16;
        packet_info *new_pkts = realloc(s->packets, new_cap * sizeof(packet_info));
        if (!new_pkts) {
            return;
        }
        s->packets = new_pkts;
        s->capacity = new_cap;
    }
    
    if (s->count < s->capacity) {
        s->packets[s->count].delta = time - s->first_time;
        s->packets[s->count].size = size;
        
        // Track packet direction counts
        if (size > 0) {
            s->positive_count++;
        } else if (size < 0) {
            s->negative_count++;
        }
        
        s->count++;
    }
}

// Function to flip packet sizes if needed
static void adjust_stream_direction(stream *s) {
    if (!s || s->count == 0) return;
    
    // If we have more negative packets than positive, flip all sizes
    if (s->negative_count > s->positive_count) {
        for (size_t i = 0; i < s->count; i++) {
            s->packets[i].size = -s->packets[i].size;
        }
    }
}

static int compare_packets(const void *a, const void *b) {
    const packet_info *pa = a;
    const packet_info *pb = b;
    return (pa->delta > pb->delta) - (pa->delta < pb->delta);
}

char* process_pcap(const char *filename) {
    stream *streams = NULL;
    pcap_t *pcap;
    char errbuf[PCAP_ERRBUF_SIZE];
    
    if ((pcap = pcap_open_offline(filename, errbuf)) == NULL) {
        return NULL;
    }

    struct pcap_pkthdr header;
    const u_char *packet;

    while ((packet = pcap_next(pcap, &header))) {
        /* ================== 链路层处理 ================== */
        const int link_type = pcap_datalink(pcap);
        size_t link_header_size = 0;
        
        // 根据链路层类型确定头部大小
        switch (link_type) {
            case DLT_EN10MB:  // Ethernet
                link_header_size = 14;
                break;
            case DLT_NULL:    // Loopback
                link_header_size = 4;
                break;
            default:
                continue;  // 不支持的链路类型
        }
        
        if (header.len <= link_header_size) continue;
        
        /* ================== IP层处理 ================== */
        const u_char *ip_packet = packet + link_header_size;
        const IP_HEADER *ip_hdr = (const IP_HEADER*)ip_packet;
        
        if (ip_hdr->protocol != IPPROTO_TCP) continue;
        
        /* ================== TCP层处理 ================== */
        size_t ip_header_length = (ip_hdr->ver_ihl & 0x0F) * 4;
        const u_char *tcp_packet = ip_packet + ip_header_length;
        const TCP_HEADER *tcp_hdr = (const TCP_HEADER*)tcp_packet;
        
        /* ================== 流识别 ================== */
        stream_key key;
        key.src_ip = ip_hdr->src;
        key.dst_ip = ip_hdr->dst;
        key.src_port = tcp_hdr->src_port;
        key.dst_port = tcp_hdr->dst_port;
        normalize_key(&key);
        
        /* ================== 查找或创建流 ================== */
        stream *s = NULL;
        HASH_FIND(hh, streams, &key, sizeof(stream_key), s);
        
        if (!s) {
            s = malloc(sizeof(stream));
            memset(s, 0, sizeof(stream));
            s->key = key;
            s->first_time = header.ts.tv_sec + header.ts.tv_usec / 1000000.0;
            s->sender_ip = ip_hdr->src;
            s->sender_port = tcp_hdr->src_port;
            s->positive_count = 0;
            s->negative_count = 0;
            HASH_ADD(hh, streams, key, sizeof(stream_key), s);
        }
        
        /* ================== 确定数据包方向 ================== */
        int packet_size = (int)(header.len - link_header_size);
        if (ip_hdr->src == s->sender_ip && tcp_hdr->src_port == s->sender_port) {
            // Outgoing packet (positive size)
            add_packet(s, header.ts.tv_sec + header.ts.tv_usec / 1000000.0, packet_size);
        } else {
            // Incoming packet (negative size)
            add_packet(s, header.ts.tv_sec + header.ts.tv_usec / 1000000.0, -packet_size);
        }
    }
    pcap_close(pcap);

    /* ================== 处理所有流 ================== */
    char *result = NULL;
    size_t total = 0;
    size_t offset = 0;

    stream *s, *tmp;
    HASH_ITER(hh, streams, s, tmp) {
        if (s->count <= 100) continue;
        
        // Adjust packet directions if needed
        adjust_stream_direction(s);

        size_t buf_len = s->count * 30;
        char *buf = malloc(buf_len);
        if (!buf) continue;

        size_t written = 0;
        for (size_t i = 0; i < s->count; i++) {
            written += snprintf(buf + written, buf_len - written, "%.6f\t%d\n", 
                              s->packets[i].delta, s->packets[i].size);
        }

        size_t needed = written + strlen("===END_STREAM===\n");
        char *temp = realloc(result, total + needed + 1);
        if (!temp) {
            free(buf);
            continue;
        }
        result = temp;

        memcpy(result + total, buf, written);
        total += written;
        memcpy(result + total, "===END_STREAM===\n", strlen("===END_STREAM===\n"));
        total += strlen("===END_STREAM===\n");
        result[total] = '\0';

        free(buf);
    }
    
    // 清理哈希表
    HASH_ITER(hh, streams, s, tmp) {
        HASH_DEL(streams, s);
        free(s->packets);
        free(s);
    }
    streams = NULL;

    return result ? result : strdup(""); // 确保返回有效指针
}

void free_result(char *ptr) {
    free(ptr);
}

// 添加DLL导出声明
__declspec(dllexport) char* process_pcap(const char *filename);
__declspec(dllexport) void free_result(char *ptr);