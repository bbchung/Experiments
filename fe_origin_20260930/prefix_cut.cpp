#include <algorithm>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "msg/md_msg.h"

// Diagnostic only: copy a byte-identical source prefix, with native message sizes.
// Production material always reads the original market partitions.
int main(int argc, char **argv)
{
    if (argc != 4) {
        return 2;
    }
    const Timestamp cutoff = std::stoll(argv[3]);
    std::ifstream input(argv[1], std::ios::binary);
    std::ofstream output(argv[2], std::ios::binary | std::ios::trunc);
    if (!input || !output) {
        return 3;
    }
    MsgHeader header{};
    Timestamp last_time = 0;
    size_t count = 0;
    while (input.read(reinterpret_cast<char *>(&header), sizeof(header))) {
        const size_t size = header.type == MsgType::BOOK ? sizeof(Book5) : header.type == MsgType::TRADE ? sizeof(Trade) : 0;
        if (size == 0) {
            throw std::runtime_error("Unknown market message");
        }
        std::vector<char> payload(size);
        if (!input.read(payload.data(), static_cast<std::streamsize>(size))) {
            throw std::runtime_error("Truncated source payload");
        }
        last_time = std::max(last_time, header.time);
        if (last_time > cutoff) {
            break;
        }
        output.write(reinterpret_cast<const char *>(&header), sizeof(header));
        output.write(payload.data(), static_cast<std::streamsize>(size));
        ++count;
    }
    if (!output || count == 0) {
        return 4;
    }
    std::cout << count << '\n';
}
