#include <fstream>
#include <iostream>
#include <stdexcept>

#include "msg/md_msg.h"

// Read-only diagnostic using native message layout and halt-state status rules.
// It emits source evidence only; it does not produce model features or labels.
int main(int argc, char **argv)
{
    if (argc != 2) {
        return 2;
    }
    std::ifstream input(argv[1], std::ios::binary);
    if (!input) {
        return 3;
    }
    MsgHeader header{};
    Timestamp last_receive = 0;
    size_t records = 0, regressions = 0;
    bool halted = false;
    std::cout << "transition,receive_time,update_time,seq,type,status_mask\n";
    auto inspect = [&](const auto &message, Timestamp exchange_time) {
        if (header.time < last_receive) {
            ++regressions;
        }
        last_receive = std::max(last_receive, header.time);
        ++records;
        const bool next_halted = is_set(message.status_mask, Status::TRIAL) || is_set(message.status_mask, Status::SUSPEND);
        if (next_halted != halted) {
            std::cout << (next_halted ? "halt" : "resume") << ',' << header.time << ',' << exchange_time << ',' << header.seq << ','
                      << static_cast<char>(header.type) << ',' << message.status_mask << '\n';
        }
        halted = next_halted;
    };
    while (input.read(reinterpret_cast<char *>(&header), sizeof(header))) {
        if (header.type == MsgType::BOOK) {
            Book5 message{};
            if (!input.read(reinterpret_cast<char *>(&message), sizeof(message))) {
                throw std::runtime_error("Truncated native book");
            }
            inspect(message, message.update_time);
        } else if (header.type == MsgType::TRADE) {
            Trade message{};
            if (!input.read(reinterpret_cast<char *>(&message), sizeof(message))) {
                throw std::runtime_error("Truncated native trade");
            }
            inspect(message, message.trade_time);
        } else {
            throw std::runtime_error("Unknown native market message");
        }
    }
    std::cerr << "records=" << records << " receive_regressions=" << regressions << '\n';
}
