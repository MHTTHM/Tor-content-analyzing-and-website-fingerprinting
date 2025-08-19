#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <winsock2.h>
#include <windows.h>
#include <pcap.h>
#include "uthash.h"
#include <windows.h>


// 调整网络头文件结构
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
    int positive_count;
    int negative_count;
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
    
    // 确保写入位置有效
    if (s->count < s->capacity) {
        s->packets[s->count].delta = time - s->first_time;
        s->packets[s->count].size = size;

        if (size > 0) {
            s->positive_count++;
        } else if (size < 0) {
            s->negative_count++;
        }

        s->count++;
    }
}

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
        // 1. 获取链路层类型
        const int link_type = pcap_datalink(pcap);

        // 2. 计算基础链路头长度
        size_t link_header_len = 0;
        switch (link_type) {
            case DLT_EN10MB:  link_header_len = 14; break; // 以太网
            case DLT_LINUX_SLL: link_header_len = 16; break; // Linux cooked
            default:
                continue;
        }

        // 3. 处理VLAN标签（仅以太网）
        if (link_type == DLT_EN10MB) {
            // 检查以太网类型是否为0x8100（VLAN）
            if (header.caplen >= 16 && ntohs(*(uint16_t*)(packet + 12)) == 0x8100) {
                link_header_len += 4; // 增加VLAN头长度
            }
        }

        /* ================== 基础长度校验 ================== */
        if (header.caplen < link_header_len) {
            continue;
        }

        /* ================== IP层处理 ================== */
        const u_char *ip_pkt = packet + link_header_len;
        IP_HEADER ip;
        
        // 1. IP头长度校验
        if (header.caplen < link_header_len + sizeof(IP_HEADER)) {
            continue;
        }

        // 2. 安全拷贝IP头（避免内存对齐问题）
        memcpy(&ip, ip_pkt, sizeof(IP_HEADER));

        // 3. IP版本校验
        if ((ip.ver_ihl >> 4) != 4) {
            continue;
        }

        /* ================== TCP层处理 ================== */
        // 1. 协议类型校验
        if (ip.protocol != IPPROTO_TCP) continue;

        // 2. TCP头位置校验
        TCP_HEADER tcp;
        const size_t tcp_start = link_header_len + (ip.ver_ihl & 0x0F) * 4;
        if (header.caplen < tcp_start + sizeof(TCP_HEADER)) {
            continue;
        }

        // 3. 安全拷贝TCP头
        memcpy(&tcp, ip_pkt + (ip.ver_ihl & 0x0F) * 4, sizeof(TCP_HEADER));
        
        /* ================== 流处理逻辑 ================== */

        stream_key key;
        memset(&key, 0, sizeof(key));  // 新增初始化
        key.src_ip = ip.src;
        key.dst_ip = ip.dst;
        key.src_port = tcp.src_port;
        key.dst_port = tcp.dst_port;

        normalize_key(&key);

        // 新增过滤echo协议的逻辑
        uint16_t host_src_port = ntohs(key.src_port);
        uint16_t host_dst_port = ntohs(key.dst_port);
        // if (host_src_port == 7 || host_dst_port == 7) {
        //     continue; // 跳过echo协议的包
        // }
        
        stream *s = NULL;
        HASH_FIND(hh, streams, &key, sizeof(stream_key), s);

        if (!s) {
            s = (stream*)malloc(sizeof(stream));
            memset(s, 0, sizeof(stream));
            memcpy(&s->key, &key, sizeof(key));
            s->first_time = header.ts.tv_sec + header.ts.tv_usec / 1e6;
            s->sender_ip = ip.src;
            s->sender_port = tcp.src_port;
            HASH_ADD(hh, streams, key, sizeof(stream_key), s);
        }

        double current_time = header.ts.tv_sec + header.ts.tv_usec / 1e6;
        int is_sender = (ip.src == s->sender_ip) && (tcp.src_port == s->sender_port);
        add_packet(s, current_time, is_sender ? header.len : -header.len);

    }
    pcap_close(pcap);

    char *result = NULL;
    size_t total = 0;
    size_t offset = 0;

    stream *s, *tmp;
    HASH_ITER(hh, streams, s, tmp) {
        if (s->count <= 100) continue;

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